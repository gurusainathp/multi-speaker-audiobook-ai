"""
src/sfx/sfx_generator.py
────────────────────────────────────────────────────────────────
Step 6 of the audiobook pipeline: manifest sfx/ambience → audio files.

Reads the manifest written by tts_generator.py, finds all segments
where type == "sfx" or type == "ambience" and file == "", generates
audio for each using gpt-4o-mini-tts, writes the MP3, and updates
the manifest's "file" field so the merge step picks it up normally.

Design note — why the same TTS model for SFX:
    OpenAI does not yet expose a dedicated sfx/audio generation model
    via the standard API. gpt-4o-mini-tts can produce short sound
    effect descriptions when prompted correctly ("generate the sound of
    a creaking door"). Results are limited compared to a dedicated sfx
    library, but the pipeline stays self-contained and cost is minimal.
    When OpenAI releases a dedicated audio generation endpoint this
    module can be upgraded by changing MODEL and _build_sfx_prompt()
    without touching anything else.

Output:
    data/audio_segments/<stem>/0005.mp3   ← generated, manifest updated
    data/audio_segments/<stem>/0012.mp3   ← generated, manifest updated

Usage (import):
    from src.sfx.sfx_generator import generate_sfx

    generate_sfx(stem="story")
    generate_sfx(stem="story", dry_run=True)

Usage (CLI):
    python src/sfx/sfx_generator.py --stem story
    python src/sfx/sfx_generator.py --stem story --dry-run
    python src/sfx/sfx_generator.py --stem story --no-resume
"""

# ── Path bootstrap ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL          = "gpt-4o-mini-tts"    # swap to dedicated audio model when available
OUTPUT_FORMAT  = "mp3"
FALLBACK_VOICE = "alloy"              # voice param required by API even for sfx
RETRY_LIMIT    = 3
RETRY_DELAY    = 2.0
SEGMENTS_DIR   = Path("data/audio_segments")

# SFX types that this module handles
SFX_TYPES = {"sfx", "ambience"}

# ── Prompt templates per sound category ──────────────────────────────────────
# gpt-4o-mini-tts produces better sfx results with explicit category context.
# Each entry maps a sound name keyword to an instruction prefix.
SFX_PROMPT_PREFIXES: dict[str, str] = {
    # Environment
    "wind":        "Generate the ambient sound of",
    "rain":        "Generate the ambient sound of",
    "thunder":     "Generate the sound effect of",
    "fire":        "Generate the ambient sound of",
    "water":       "Generate the ambient sound of",
    "ocean":       "Generate the ambient sound of",
    "crowd":       "Generate the ambient sound of",
    "forest":      "Generate the ambient sound of",
    "night":       "Generate the ambient sound of",
    "storm":       "Generate the ambient sound of",
    # Interior
    "door":        "Generate the sound effect of a",
    "footsteps":   "Generate the sound effect of",
    "clock":       "Generate the sound effect of a",
    "glass":       "Generate the sound effect of",
    "book":        "Generate the sound effect of",
    "paper":       "Generate the sound effect of",
    "chair":       "Generate the sound effect of a",
    "keyboard":    "Generate the sound effect of",
    # Events
    "thunder":     "Generate the dramatic sound effect of",
    "explosion":   "Generate the sound effect of a distant",
    "crash":       "Generate the sound effect of a",
    "bell":        "Generate the sound effect of a",
    "phone":       "Generate the sound effect of a",
    "alarm":       "Generate the sound effect of an",
}

DEFAULT_SFX_PREFIX = "Generate the sound effect of"
DEFAULT_AMB_PREFIX = "Generate the ambient background sound of"


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def generate_sfx(
    stem: str = "story",
    dry_run: bool = False,
    resume: bool = True,
) -> list[Path]:
    """
    Generate audio for all sfx/ambience entries in the manifest.

    Reads:   data/audio_segments/<stem>/manifest.json
    Writes:  data/audio_segments/<stem>/<index>.mp3  for each sfx/ambience
    Updates: manifest.json with the generated filename per segment

    Args:
        stem:    Story stem name — must match the subfolder under audio_segments/.
        dry_run: If True, skips API calls and writes empty placeholder files.
        resume:  If True, skips sfx segments whose file already exists.

    Returns:
        List of Paths to the generated SFX audio files.

    Raises:
        FileNotFoundError: If manifest.json does not exist.
        EnvironmentError:  If OPENAI_API_KEY is not set (and dry_run=False).
        RuntimeError:      If an API call fails after all retries.
    """
    if not dry_run:
        _check_api_key()

    manifest_path, manifest = _load_manifest(stem)
    segments = manifest.get("segments", [])

    sfx_segments = [s for s in segments if s.get("type", "") in SFX_TYPES]
    total_sfx    = len(sfx_segments)

    if total_sfx == 0:
        logger.info(f"[generate_sfx] No sfx/ambience segments found in manifest for '{stem}'.")
        return []

    logger.info(
        f"[generate_sfx] Found {total_sfx} sfx/ambience segment(s) in manifest. "
        f"Stem: '{stem}'  Dry-run: {dry_run}"
    )

    audio_dir   = SEGMENTS_DIR / stem
    generated   = 0
    skipped     = 0
    sfx_paths:  list[Path] = []
    manifest_dirty = False   # track whether manifest needs saving

    for sfx_seg in sfx_segments:
        seg_index  = sfx_seg.get("index", 0)
        seg_type   = sfx_seg.get("type", "sfx")
        sound_name = str(sfx_seg.get("sound", "")).strip()
        existing_file = str(sfx_seg.get("file", "")).strip()

        if not sound_name:
            logger.warning(
                f"[generate_sfx] Segment {seg_index}: no 'sound' field — skipping."
            )
            continue

        # Filename matches the segment index for consistent ordering
        filename = f"{seg_index:04d}.{OUTPUT_FORMAT}"
        out_path = audio_dir / filename

        # ── Resume: skip if file already exists and has content ──────────────
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            logger.info(
                f"[generate_sfx] [{seg_index:04d}] Skipping '{sound_name}' "
                f"(file already exists: {filename})"
            )
            sfx_paths.append(out_path)
            skipped += 1
            # Ensure manifest has the filename even if set from a previous run
            if existing_file != filename:
                sfx_seg["file"] = filename
                manifest_dirty = True
            continue

        # ── Build prompt ─────────────────────────────────────────────────────
        sfx_prompt = _build_sfx_prompt(sound_name, seg_type)

        logger.info(
            f"[generate_sfx] [{seg_index:04d}] [{seg_type.upper():<10}] "
            f"'{sound_name}'  →  prompt: '{sfx_prompt[:60]}...'"
        )

        # ── Generate audio ───────────────────────────────────────────────────
        if dry_run:
            out_path.write_bytes(b"")
        else:
            _call_tts(prompt=sfx_prompt, out_path=out_path)

        sfx_paths.append(out_path)
        generated += 1

        # ── Update manifest entry with filename ──────────────────────────────
        sfx_seg["file"] = filename
        manifest_dirty = True

    # ── Save updated manifest ─────────────────────────────────────────────────
    if manifest_dirty:
        _save_manifest(manifest_path, manifest)
        logger.info(f"[generate_sfx] Manifest updated: {manifest_path}")

    logger.info(
        f"[generate_sfx] Complete. "
        f"Generated: {generated}  Skipped: {skipped}  Total: {len(sfx_paths)}"
    )

    return sfx_paths


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ════════════════════════════════════════════════════════════════════════════

def _build_sfx_prompt(sound_name: str, seg_type: str) -> str:
    """
    Build a natural-language prompt for gpt-4o-mini-tts to generate a sound.

    Converts snake_case sound names like "door_creak" or "night_wind" into
    a full instruction sentence that steers the model toward the right output.

    Format:
        "<prefix> <human-readable sound description>."

    Examples:
        "door_creak"      → "Generate the sound effect of a door creaking slowly."
        "thunder_rumble"  → "Generate the dramatic sound effect of thunder rumbling."
        "night_wind"      → "Generate the ambient background sound of wind at night."
        "footsteps_stone" → "Generate the sound effect of footsteps on stone."
        "rain_soft"       → "Generate the ambient sound of soft rain falling."
    """
    # Convert snake_case to human-readable: "night_wind" → "night wind"
    readable = sound_name.replace("_", " ").strip()

    # Pick prefix based on seg_type and sound keyword
    if seg_type == "ambience":
        prefix = DEFAULT_AMB_PREFIX
    else:
        # Check if any keyword in the sound name matches our prefix table
        prefix = DEFAULT_SFX_PREFIX
        for keyword, kw_prefix in SFX_PROMPT_PREFIXES.items():
            if keyword in sound_name.lower():
                prefix = kw_prefix
                break

    return f"{prefix} {readable}."


# ════════════════════════════════════════════════════════════════════════════
# TTS API CALL
# ════════════════════════════════════════════════════════════════════════════

def _call_tts(prompt: str, out_path: Path) -> None:
    """
    Call gpt-4o-mini-tts with the sfx prompt and write to out_path.
    Retries up to RETRY_LIMIT times on transient failures.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Run: pip install openai")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    last_error: Exception | None = None

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.audio.speech.create(
                model=MODEL,
                voice=FALLBACK_VOICE,
                input=prompt,
                response_format=OUTPUT_FORMAT,
            )
            response.stream_to_file(str(out_path))
            return  # success

        except Exception as e:
            last_error = e
            if attempt < RETRY_LIMIT:
                logger.warning(
                    f"[_call_tts] Attempt {attempt}/{RETRY_LIMIT} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[_call_tts] All {RETRY_LIMIT} attempts failed.")

    raise RuntimeError(
        f"SFX generation failed after {RETRY_LIMIT} attempts. "
        f"Last error: {last_error}\n"
        f"Prompt: {prompt}"
    )


# ════════════════════════════════════════════════════════════════════════════
# MANIFEST I/O
# ════════════════════════════════════════════════════════════════════════════

def _load_manifest(stem: str) -> tuple[Path, dict]:
    """Load manifest.json for the given stem. Returns (path, dict)."""
    manifest_path = SEGMENTS_DIR / stem / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run Step 4 (tts_generator.py) first."
        )

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse manifest.json: {e}")

    return manifest_path, data


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write the updated manifest back to disk."""
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _check_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate audio for sfx/ambience segments in a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/sfx/sfx_generator.py --stem story\n"
            "  python src/sfx/sfx_generator.py --stem story --dry-run\n"
            "  python src/sfx/sfx_generator.py --stem story --no-resume -v\n"
        ),
    )
    p.add_argument(
        "--stem", "-s", required=True,
        help="Story stem name (e.g. 'story' → reads data/audio_segments/story/).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Write empty placeholder files without calling the API.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Regenerate all sfx even if files already exist.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    try:
        sfx_paths = generate_sfx(
            stem=args.stem,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
    except (FileNotFoundError, EnvironmentError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    if sfx_paths:
        print(f"\n── SFX generation complete ─────────────────────────────")
        print(f"  Generated : {len(sfx_paths)} file(s)")
        print(f"  Location  : {SEGMENTS_DIR / args.stem}/")
        print(f"  Manifest  : {SEGMENTS_DIR / args.stem / 'manifest.json'} (updated)")
        if args.dry_run:
            print("  ⚠️  Dry-run — no real audio was generated.")
        print()
    else:
        print("\n  No sfx/ambience segments found in manifest.\n")


if __name__ == "__main__":
    main()