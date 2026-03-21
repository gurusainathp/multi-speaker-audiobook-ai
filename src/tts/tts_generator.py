"""
src/tts/tts_generator.py
────────────────────────────────────────────────────────────────
Step 4 of the audiobook pipeline: clean script → audio segments.

Takes the cleaned list[dict] from clean_script() and generates one
MP3 file per segment using the OpenAI TTS API.

Output layout:
    data/audio_segments/<stem>/
        0001.mp3    ← segment 1
        0002.mp3    ← segment 2
        ...

Each file is named with a zero-padded 4-digit index so they sort
correctly in any file manager or audio tool.

A manifest file is also written alongside the audio:
    data/audio_segments/<stem>/manifest.json
This records the segment metadata (speaker, emotion, voice, text)
matched to its filename — used by the merge step.

Usage (import):
    from src.tts.tts_generator import generate_audio

    audio_files = generate_audio(clean_script, stem="story")
    # returns list of Path objects in order

Usage (CLI):
    python src/tts/tts_generator.py -i data/scripts/story_clean.json
    python src/tts/tts_generator.py -i data/scripts/story_clean.json --dry-run
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

TTS_MODEL      = "gpt-4o-mini-tts"
OUTPUT_FORMAT  = "mp3"
DEFAULT_SPEED  = 1.0          # 0.25–4.0; 1.0 is natural
RETRY_LIMIT    = 3            # Retry failed API calls up to this many times
RETRY_DELAY    = 2.0          # Seconds to wait between retries
BASE_OUTPUT_DIR = Path("data/audio_segments")

# Fallback voice if a segment has no tts_voice field
FALLBACK_VOICE = "alloy"

# Valid OpenAI TTS voices
VALID_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def generate_audio(
    script: list,
    stem: str = "story",
    speed: float = DEFAULT_SPEED,
    dry_run: bool = False,
    resume: bool = True,
) -> list[Path]:
    """
    Generate one MP3 audio file per segment of the cleaned script.

    Args:
        script:  Output of clean_script() — list of segment dicts.
                 Each dict must have: text, tts_voice (or voice).
        stem:    Used as the output subfolder name under data/audio_segments/.
        speed:   TTS playback speed (0.25–4.0). Default 1.0.
        dry_run: If True, skips API calls and creates empty placeholder files.
                 Useful for testing the file-structure logic.
        resume:  If True, skips segments whose output file already exists.
                 Allows resuming interrupted runs without re-billing.

    Returns:
        Ordered list of Path objects pointing to the generated MP3 files.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set (and dry_run=False).
        ValueError:       If the script is empty or malformed.
        RuntimeError:     If a segment fails after all retries.
    """
    if not script:
        raise ValueError("Script is empty — nothing to generate audio for.")

    if not dry_run:
        _check_api_key()

    speed = _clamp_speed(speed)

    output_dir = BASE_OUTPUT_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(script)
    logger.info(f"[generate_audio] {total} segment(s) → {output_dir}")
    logger.info(f"[generate_audio] Model: {TTS_MODEL}  Speed: {speed}  Dry-run: {dry_run}")

    audio_paths: list[Path] = []
    skipped = 0
    generated = 0

    for i, seg in enumerate(script, start=1):
        filename = f"{i:04d}.{OUTPUT_FORMAT}"
        out_path = output_dir / filename

        # ── Resume: skip if already exists ──────────────────────────────────
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"[generate_audio] [{i:04d}/{total}] Skipping (already exists): {filename}")
            audio_paths.append(out_path)
            skipped += 1
            continue

        # ── Resolve voice ────────────────────────────────────────────────────
        voice = _resolve_voice(seg)

        # ── Resolve text ─────────────────────────────────────────────────────
        text = str(seg.get("text", "")).strip()
        if not text:
            logger.warning(f"[generate_audio] [{i:04d}/{total}] Empty text — skipping segment.")
            continue

        speaker = seg.get("speaker", "Unknown")
        emotion = seg.get("emotion", "neutral")

        logger.info(
            f"[generate_audio] [{i:04d}/{total}] "
            f"{speaker:<12} ({emotion:<13})  voice={voice:<8}  "
            f'"{text[:45]}{"..." if len(text) > 45 else ""}"'
        )

        # ── Generate audio ───────────────────────────────────────────────────
        if dry_run:
            out_path.write_bytes(b"")   # empty placeholder
        else:
            _generate_segment(text=text, voice=voice, speed=speed, out_path=out_path)

        audio_paths.append(out_path)
        generated += 1

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = _write_manifest(script, audio_paths, output_dir, stem)

    logger.info(
        f"[generate_audio] Done. "
        f"Generated: {generated}  Skipped: {skipped}  "
        f"Total files: {len(audio_paths)}"
    )
    logger.info(f"[generate_audio] Manifest: {manifest_path}")

    return audio_paths


# ════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _generate_segment(
    text: str,
    voice: str,
    speed: float,
    out_path: Path,
) -> None:
    """
    Call the OpenAI TTS API and write the audio to out_path.
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
                model=TTS_MODEL,
                voice=voice,
                input=text,
                speed=speed,
                response_format=OUTPUT_FORMAT,
            )
            # Stream the audio bytes directly to disk
            response.stream_to_file(str(out_path))
            return  # success

        except Exception as e:
            last_error = e
            if attempt < RETRY_LIMIT:
                logger.warning(
                    f"[_generate_segment] Attempt {attempt}/{RETRY_LIMIT} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[_generate_segment] All {RETRY_LIMIT} attempts failed.")

    raise RuntimeError(
        f"TTS generation failed after {RETRY_LIMIT} attempts. "
        f"Last error: {last_error}"
    )


def _resolve_voice(seg: dict) -> str:
    """
    Get the TTS voice for a segment.

    Priority:
      1. seg["tts_voice"]  — set by json_cleaner (preferred)
      2. seg["voice"]      — logical slot; map it to a real voice name
      3. FALLBACK_VOICE    — safety net
    """
    # Best case: cleaner already resolved to an actual TTS voice name
    tts_voice = str(seg.get("tts_voice", "")).strip().lower()
    if tts_voice in VALID_VOICES:
        return tts_voice

    # Fallback: logical slot like "male1", "female2", "narrator"
    # Re-apply the same VOICE_MAP the cleaner uses
    VOICE_MAP = {
        "narrator": "onyx",
        "male1":    "echo",
        "male2":    "fable",
        "male3":    "onyx",
        "female1":  "nova",
        "female2":  "shimmer",
        "female3":  "alloy",
        "neutral1": "alloy",
        "neutral2": "echo",
    }
    voice_slot = str(seg.get("voice", "")).strip().lower()
    if voice_slot in VOICE_MAP:
        return VOICE_MAP[voice_slot]

    logger.warning(
        f"[_resolve_voice] No valid voice found for speaker "
        f"'{seg.get('speaker', '?')}' — using fallback '{FALLBACK_VOICE}'."
    )
    return FALLBACK_VOICE


def _clamp_speed(speed: float) -> float:
    """Clamp speed to the OpenAI-supported range 0.25–4.0."""
    clamped = max(0.25, min(4.0, speed))
    if clamped != speed:
        logger.warning(f"[_clamp_speed] Speed {speed} clamped to {clamped}.")
    return clamped


def _write_manifest(
    script: list,
    audio_paths: list[Path],
    output_dir: Path,
    stem: str,
) -> Path:
    """
    Write a manifest.json alongside the audio files.

    The manifest maps each audio file to its segment metadata.
    The merge step reads this to know the correct order and pauses.

    Format:
    {
      "stem": "story",
      "total_segments": 42,
      "segments": [
        {
          "index": 1,
          "file": "0001.mp3",
          "speaker": "Narrator",
          "type": "narration",
          "emotion": "neutral",
          "voice": "narrator",
          "tts_voice": "onyx",
          "text": "The sun had already set."
        },
        ...
      ]
    }
    """
    segments_meta = []

    # Pair each path with its original script segment
    # audio_paths may be shorter than script if some segments were skipped
    path_iter = iter(audio_paths)

    for i, seg in enumerate(script, start=1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue  # was skipped in main loop

        try:
            file_path = next(path_iter)
        except StopIteration:
            break

        segments_meta.append({
            "index":     i,
            "file":      file_path.name,
            "speaker":   seg.get("speaker", "Narrator"),
            "type":      seg.get("type", "narration"),
            "emotion":   seg.get("emotion", "neutral"),
            "voice":     seg.get("voice", ""),
            "tts_voice": seg.get("tts_voice", ""),
            "text":      text,
        })

    manifest = {
        "stem":           stem,
        "total_segments": len(segments_meta),
        "tts_model":      TTS_MODEL,
        "format":         OUTPUT_FORMAT,
        "segments":       segments_meta,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


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
        description="Generate TTS audio segments from a cleaned script JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/tts/tts_generator.py -i data/scripts/story_clean.json\n"
            "  python src/tts/tts_generator.py -i data/scripts/story_clean.json --dry-run\n"
            "  python src/tts/tts_generator.py -i data/scripts/story_clean.json --speed 0.95\n"
            "  python src/tts/tts_generator.py -i data/scripts/story_clean.json --no-resume\n"
        ),
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to cleaned script JSON (output of json_cleaner.py).",
    )
    p.add_argument(
        "--speed", type=float, default=DEFAULT_SPEED,
        help=f"TTS playback speed (0.25–4.0). Default: {DEFAULT_SPEED}.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls; create empty placeholder files. For testing only.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Regenerate all segments even if output files already exist.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    json_path = Path(args.input)
    if not json_path.exists():
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    try:
        script = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse script JSON: {e}")
        sys.exit(1)

    # Derive stem from filename (e.g. "story_clean" → "story")
    stem = json_path.stem.replace("_clean", "").replace("_raw", "")

    try:
        audio_paths = generate_audio(
            script=script,
            stem=stem,
            speed=args.speed,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
    except (EnvironmentError, RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    print(f"\n── Audio generation complete ───────────────────────────")
    print(f"  Segments generated : {len(audio_paths)}")
    print(f"  Output folder      : {BASE_OUTPUT_DIR / stem}/")
    print(f"  Manifest           : {BASE_OUTPUT_DIR / stem / 'manifest.json'}")
    if args.dry_run:
        print("  ⚠️  Dry-run mode — no real audio was generated.")
    print()


if __name__ == "__main__":
    main()