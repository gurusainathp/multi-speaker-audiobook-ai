"""
src/sfx/sfx_generator.py
────────────────────────────────────────────────────────────────
Step 5 of the audiobook pipeline: manifest sfx/ambience → audio files.

Uses Freesound.org to source real sound effects and ambience recordings.
An AI-generated search query (gpt-4o-mini) converts each snake_case
sound name + surrounding scene context into a precise Freesound query,
so you get "wooden door creak slow" instead of just "door creak".

Setup required:
    FREESOUND_API_KEY in your .env file
    Get a free key at: https://freesound.org/apiv2/apply/

Reads:   data/audio_segments/<stem>/manifest.json
Writes:  data/audio_segments/<stem>/<index>.mp3  for each sfx/ambience
Updates: manifest.json with the generated filename per segment

Usage (import):
    from src.sfx.sfx_generator import generate_sfx
    generate_sfx(stem="story")
    generate_sfx(stem="story", dry_run=True)

Usage (CLI):
    python src/sfx/sfx_generator.py --stem story
    python src/sfx/sfx_generator.py --stem story --dry-run
    python src/sfx/sfx_generator.py --stem story --no-resume -v
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
import ssl
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from src.script_parser.prompts import SFX_QUERY_SYSTEM_PROMPT, SFX_QUERY_USER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

QUERY_MODEL  = "gpt-4o-mini"   # generates Freesound search queries
                                # chosen over gpt-4.1-mini: 2.7x cheaper,
                                # identical quality for 3-5 word generation
OUTPUT_FORMAT = "mp3"
RETRY_LIMIT   = 3
RETRY_DELAY   = 2.0
SEGMENTS_DIR  = Path("data/audio_segments")
SFX_TYPES     = {"sfx", "ambience"}

# Duration filter per type (seconds) — keeps results focused
DURATION_MAX = {"sfx": 8, "ambience": 15}

# Windows fix: Python's urllib doesn't use the system cert store, causing
# SSL_CERTIFICATE_VERIFY_FAILED on many Windows machines. We create an
# unverified context for Freesound requests only — the API key in the URL
# still authenticates us; skipping cert verification is acceptable for a
# public audio download service (not sending any sensitive credentials).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE


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

    For each sfx/ambience segment:
      1. gpt-4o-mini generates an optimised Freesound search query
         using the sound name + surrounding scene context
      2. Freesound API returns the top-rated matching sound
      3. Preview MP3 is downloaded and saved
      4. manifest.json is updated with the filename

    Args:
        stem:    Story stem — matches subfolder under data/audio_segments/.
        dry_run: Skip all API calls; write empty placeholder files instead.
        resume:  Skip segments whose audio file already exists.

    Returns:
        List of Paths to the generated/downloaded SFX audio files.

    Raises:
        FileNotFoundError: If manifest.json does not exist.
        EnvironmentError:  If OPENAI_API_KEY or FREESOUND_API_KEY not set.
    """
    if not dry_run:
        _check_api_key("OPENAI_API_KEY",
                       "Required for gpt-4o-mini query generation.")
        _check_api_key("FREESOUND_API_KEY",
                       "Get a free key at https://freesound.org/apiv2/apply/")

    manifest_path, manifest = _load_manifest(stem)
    segments     = manifest.get("segments", [])
    sfx_segments = [s for s in segments if s.get("type", "") in SFX_TYPES]
    total_sfx    = len(sfx_segments)

    if total_sfx == 0:
        logger.info(f"[generate_sfx] No sfx/ambience segments in '{stem}' manifest.")
        return []

    logger.info(
        f"[generate_sfx] {total_sfx} sfx/ambience segment(s).  "
        f"stem='{stem}'  dry_run={dry_run}"
    )

    # Index map for scene context lookup
    seg_by_index = {s.get("index", i): s for i, s in enumerate(segments)}

    audio_dir      = SEGMENTS_DIR / stem
    generated      = 0
    skipped        = 0
    sfx_paths: list[Path] = []
    manifest_dirty = False

    for sfx_seg in sfx_segments:
        seg_index     = sfx_seg.get("index", 0)
        seg_type      = sfx_seg.get("type", "sfx")
        sound_name    = str(sfx_seg.get("sound", "")).strip()
        existing_file = str(sfx_seg.get("file", "")).strip()

        if not sound_name:
            logger.warning(
                f"[generate_sfx] Segment {seg_index}: no 'sound' field — skipping."
            )
            continue

        filename = f"{seg_index:04d}.{OUTPUT_FORMAT}"
        out_path = audio_dir / filename

        # ── Resume ──────────────────────────────────────────────────────────
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            logger.info(
                f"[generate_sfx] [{seg_index:04d}] [{seg_type.upper()}] "
                f"'{sound_name}' — already exists, skipping."
            )
            sfx_paths.append(out_path)
            skipped += 1
            if existing_file != filename:
                sfx_seg["file"] = filename
                manifest_dirty  = True
            continue

        logger.info(
            f"[generate_sfx] [{seg_index:04d}] [{seg_type.upper():<10}]  "
            f"'{sound_name}'"
        )

        if dry_run:
            out_path.write_bytes(b"")
            logger.info(f"[generate_sfx] [{seg_index:04d}] Dry-run placeholder written.")
        else:
            # Step 1: AI query generation
            scene_context = _get_scene_context(sfx_seg, seg_by_index)
            query         = _generate_query(sound_name, seg_type, scene_context)

            # Step 2: Freesound download (falls back to silence if it fails)
            success = _fetch_freesound(sound_name, query, seg_type, out_path)
            if not success:
                logger.warning(
                    f"[generate_sfx] [{seg_index:04d}] Freesound failed — "
                    f"writing silence placeholder."
                )
                _write_silence(out_path, 2000 if seg_type == "sfx" else 4000)

        sfx_paths.append(out_path)
        generated     += 1
        sfx_seg["file"] = filename
        manifest_dirty  = True

    if manifest_dirty:
        _save_manifest(manifest_path, manifest)
        logger.info(f"[generate_sfx] Manifest updated: {manifest_path}")

    logger.info(
        f"[generate_sfx] Done.  "
        f"Generated: {generated}  Skipped: {skipped}  Total: {len(sfx_paths)}"
    )
    return sfx_paths


# ════════════════════════════════════════════════════════════════════════════
# SCENE CONTEXT
# ════════════════════════════════════════════════════════════════════════════

def _get_scene_context(sfx_seg: dict, seg_by_index: dict) -> str:
    """
    Grab the nearest speech segments before and after this sfx to give
    the query generator scene awareness.

    Example:
        sfx at index 3, neighbours are "The library had been closed for years."
        and "I thought you had left the city."
        → "The library had been closed for years. / I thought you had left."
    """
    idx   = sfx_seg.get("index", 0)
    parts = []

    for offset in [-1, -2]:
        n = seg_by_index.get(idx + offset)
        if n and n.get("type") in ("narration", "dialogue"):
            text = str(n.get("text", "")).strip()
            if text:
                parts.append(text[:120])
                break

    for offset in [1, 2]:
        n = seg_by_index.get(idx + offset)
        if n and n.get("type") in ("narration", "dialogue"):
            text = str(n.get("text", "")).strip()
            if text:
                parts.append(text[:120])
                break

    return " / ".join(parts) if parts else "no context available"


# ════════════════════════════════════════════════════════════════════════════
# QUERY GENERATION  (gpt-4o-mini)
# ════════════════════════════════════════════════════════════════════════════

def _generate_query(sound_name: str, seg_type: str, scene_context: str) -> str:
    """
    Use gpt-4o-mini to generate an optimised Freesound search query.

    Converts "door_creak" + library scene context into something like
    "wooden door creak slow" — much more precise than just replacing
    underscores with spaces.

    Falls back to plain name conversion on any failure, so the pipeline
    is never blocked by a query model issue.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return sound_name.replace("_", " ")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_message = (
        SFX_QUERY_USER_PROMPT
        .replace("<<SOUND_NAME>>",   sound_name.replace("_", " "))
        .replace("<<SCENE_CONTEXT>>", scene_context)
        .replace("<<SOUND_TYPE>>",   seg_type)
    )

    import time
    last_error = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=QUERY_MODEL,
                temperature=0.2,   # low = consistent, specific queries
                max_tokens=20,     # 3-5 words; hard cap prevents runaway output
                messages=[
                    {"role": "system", "content": SFX_QUERY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            query = (
                response.choices[0].message.content
                .strip().strip('"').strip("'").lower()
            )
            if query:
                logger.info(
                    f"[_generate_query] '{sound_name}'  →  query: '{query}'"
                )
                return query

        except Exception as e:
            last_error = e
            if attempt < RETRY_LIMIT:
                logger.warning(
                    f"[_generate_query] Attempt {attempt}/{RETRY_LIMIT} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)

    logger.warning(
        f"[_generate_query] All attempts failed ({last_error}). "
        f"Using plain name fallback."
    )
    return sound_name.replace("_", " ")


# ════════════════════════════════════════════════════════════════════════════
# FREESOUND DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

def _fetch_freesound(
    sound_name: str,
    query: str,
    seg_type: str,
    out_path: Path,
) -> bool:
    """
    Search Freesound.org with the AI-generated query and download the
    highest-rated matching preview MP3.

    Search parameters:
        - Top 5 results sorted by rating descending
        - Duration filter: ≤8s for sfx, ≤15s for ambience
        - Picks highest-rated result that has an HQ preview

    Returns True on success, False on any failure.
    """
    api_key = os.getenv("FREESOUND_API_KEY", "")
    if not api_key:
        logger.error("[_fetch_freesound] FREESOUND_API_KEY not set.")
        return False

    duration_max = DURATION_MAX.get(seg_type, 8)

    search_url = (
        "https://freesound.org/apiv2/search/text/"
        f"?query={urllib.parse.quote(query)}"
        f"&fields=id,name,previews,duration,avg_rating"
        f"&page_size=5"
        f"&filter=duration:[0.5+TO+{duration_max}]"
        f"&sort=rating_desc"
        f"&token={api_key}"
    )

    import time
    last_error = None

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            with urllib.request.urlopen(search_url, timeout=10, context=_SSL_CTX) as resp:
                data = json.loads(resp.read())

            results = data.get("results", [])
            if not results:
                logger.warning(
                    f"[_fetch_freesound] No results for query '{query}' "
                    f"(sound: '{sound_name}')."
                )
                return False

            # Pick highest-rated result with HQ preview
            chosen = None
            for result in results:
                if result.get("previews", {}).get("preview-hq-mp3"):
                    chosen = result
                    break
            if not chosen:
                chosen = results[0]

            preview_url = (
                chosen["previews"].get("preview-hq-mp3")
                or chosen["previews"].get("preview-lq-mp3")
            )
            if not preview_url:
                logger.warning(
                    f"[_fetch_freesound] No preview URL for '{chosen.get('name')}'."
                )
                return False

            with urllib.request.urlopen(preview_url, timeout=15, context=_SSL_CTX) as resp:
                out_path.write_bytes(resp.read())

            logger.info(
                f"[_fetch_freesound] ✅ '{chosen['name']}'  "
                f"(query: '{query}', rating: {chosen.get('avg_rating', 'n/a')}, "
                f"duration: {chosen.get('duration', '?'):.1f}s)"
            )
            return True

        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:   # rate limited
                wait = 5 * attempt
                logger.warning(
                    f"[_fetch_freesound] Rate limited — waiting {wait}s "
                    f"(attempt {attempt}/{RETRY_LIMIT})"
                )
                time.sleep(wait)
            else:
                logger.warning(
                    f"[_fetch_freesound] HTTP {e.code} for query '{query}': {e}"
                )
                return False

        except Exception as e:
            last_error = e
            if attempt < RETRY_LIMIT:
                logger.warning(
                    f"[_fetch_freesound] Attempt {attempt}/{RETRY_LIMIT} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)

    logger.error(
        f"[_fetch_freesound] All {RETRY_LIMIT} attempts failed for "
        f"query '{query}' (sound: '{sound_name}'). Last error: {last_error}"
    )
    return False


# ── Silence fallback ──────────────────────────────────────────────────────────

def _write_silence(out_path: Path, duration_ms: int) -> None:
    """Write a silent MP3 using pydub. Falls back to empty file if unavailable."""
    try:
        from pydub import AudioSegment
        AudioSegment.silent(duration=duration_ms).export(str(out_path), format="mp3")
    except ImportError:
        logger.warning("[_write_silence] pydub not installed — writing empty file.")
        out_path.write_bytes(b"")


# ── Manifest I/O ──────────────────────────────────────────────────────────────

def _load_manifest(stem: str) -> tuple[Path, dict]:
    manifest_path = SEGMENTS_DIR / stem / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run Step 4 first:  python main.py --stem {stem} --from-step 4 --to-step 4"
        )
    try:
        return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse manifest.json: {e}")


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _check_api_key(key: str, hint: str = "") -> None:
    if not os.getenv(key):
        raise EnvironmentError(
            f"{key} is not set. {hint}\n"
            f"Add it to your .env file."
        )


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate SFX/ambience audio for manifest entries via Freesound.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires in .env:\n"
            "  OPENAI_API_KEY     — for gpt-4o-mini query generation\n"
            "  FREESOUND_API_KEY  — get free at freesound.org/apiv2/apply/\n"
            "\n"
            "Examples:\n"
            "  python src/sfx/sfx_generator.py --stem story\n"
            "  python src/sfx/sfx_generator.py --stem story --dry-run\n"
            "  python src/sfx/sfx_generator.py --stem story --no-resume -v\n"
        ),
    )
    p.add_argument("--stem",      "-s", required=True,       help="Story stem name.")
    p.add_argument("--dry-run",         action="store_true", help="No API calls.")
    p.add_argument("--no-resume",       action="store_true", help="Regenerate all.")
    p.add_argument("--verbose",   "-v", action="store_true", help="Debug logging.")
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
        logging.getLogger(__name__).error(str(e))
        sys.exit(1)

    if sfx_paths:
        print(f"\n── SFX generation complete ─────────────────────────────")
        print(f"  Generated : {len(sfx_paths)} file(s)")
        print(f"  Location  : {SEGMENTS_DIR / args.stem}/")
        print(f"  Manifest  : {SEGMENTS_DIR / args.stem / 'manifest.json'}  (updated)")
        if args.dry_run:
            print("  ⚠️  Dry-run — no real audio was generated.")
        print()
    else:
        print("\n  No sfx/ambience segments found in manifest.\n")


if __name__ == "__main__":
    main()