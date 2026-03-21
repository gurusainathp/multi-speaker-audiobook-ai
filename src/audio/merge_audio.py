"""
src/audio/merge_audio.py
────────────────────────────────────────────────────────────────
Step 5 of the audiobook pipeline: audio segments → final audiobook.

Reads manifest.json (written by tts_generator.py), loads each MP3
in manifest order, inserts context-aware silence between segments,
and exports one final MP3.

Pause logic (smart pauses):
    Pause duration is chosen based on the NEXT segment's context,
    not the current one — so the silence feels like breathing room
    before the next line begins.

    Base rules:
        narration → narration   :  500 ms  (scene beat)
        narration → dialogue    :  350 ms  (narrator hands off to character)
        dialogue  → narration   :  500 ms  (character finishes, narrator resumes)
        dialogue  → dialogue    :  300 ms  (conversation back-and-forth)

    Emotion modifiers (applied to the CURRENT segment's emotion):
        sad / melancholic       :  +300 ms  (let sorrow breathe)
        contemplative / bitter  :  +200 ms  (thoughtful weight)
        tense / angry           :  +100 ms  (urgency but still a beat)
        gentle / hopeful        :  +100 ms  (softness needs space)
        happy / excited         :   -50 ms  (energy keeps moving)

    Speaker-change bonus:
        If the speaker changes between two segments: +100 ms
        (helps the listener register it's a different voice)

Output:
    data/final/<stem>.mp3

Usage (import):
    from src.audio.merge_audio import merge_audio

    output_path = merge_audio(stem="story")
    output_path = merge_audio(stem="story", fixed_pause_ms=400)  # override smart pauses

Usage (CLI):
    python src/audio/merge_audio.py --stem story
    python src/audio/merge_audio.py --stem story --fixed-pause 400
    python src/audio/merge_audio.py --stem story --dry-run
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

logger = logging.getLogger(__name__)

# ── Directory constants ───────────────────────────────────────────────────────

SEGMENTS_DIR = Path("data/audio_segments")
FINAL_DIR    = Path("data/final")

# ── Pause table (milliseconds) ────────────────────────────────────────────────
# Key: (current_type, next_type)
BASE_PAUSE_MS: dict[tuple[str, str], int] = {
    ("narration", "narration"): 500,
    ("narration", "dialogue"):  350,
    ("dialogue",  "narration"): 500,
    ("dialogue",  "dialogue"):  300,
}
DEFAULT_PAUSE_MS = 400   # fallback if type combo not in table

# Additive modifiers based on CURRENT segment's emotion
EMOTION_MODIFIER_MS: dict[str, int] = {
    "sad":           300,
    "melancholic":   300,
    "contemplative": 200,
    "bitter":        200,
    "tense":         100,
    "angry":         100,
    "gentle":        100,
    "hopeful":       100,
    "happy":         -50,
    "excited":       -50,
}

SPEAKER_CHANGE_BONUS_MS = 100   # extra pause when the speaker changes
MIN_PAUSE_MS            = 150   # never go below this, even after subtractions


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def merge_audio(
    stem: str = "story",
    fixed_pause_ms: int | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Merge all audio segments for a given stem into one final MP3.

    Reads:  data/audio_segments/<stem>/manifest.json
    Writes: data/final/<stem>.mp3

    Args:
        stem:           The story stem (matches the subfolder under audio_segments/).
        fixed_pause_ms: If set, use a fixed silence (ms) between every segment
                        instead of smart pauses. Useful for quick testing.
        output_dir:     Override the output directory. Default: data/final/.
        dry_run:        If True, loads and validates everything but writes no file.

    Returns:
        Path to the final MP3 (or the expected path if dry_run=True).

    Raises:
        FileNotFoundError: If manifest.json or any audio segment is missing.
        ImportError:       If pydub is not installed.
        RuntimeError:      If no valid audio segments are found.
    """
    _check_pydub()

    from pydub import AudioSegment

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = _load_manifest(stem)
    segments_meta = manifest.get("segments", [])

    if not segments_meta:
        raise RuntimeError(f"Manifest for '{stem}' contains no segments.")

    total = len(segments_meta)
    logger.info(f"[merge_audio] Merging {total} segment(s) for stem '{stem}'")

    # ── Load audio segments ───────────────────────────────────────────────────
    audio_dir = SEGMENTS_DIR / stem
    loaded: list[tuple[AudioSegment, dict]] = []   # (audio, meta)

    for i, meta in enumerate(segments_meta):
        filename = meta.get("file", "")
        file_path = audio_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Audio segment missing: {file_path}\n"
                f"Run Step 4 (tts_generator.py) to generate it."
            )

        if file_path.stat().st_size == 0:
            logger.warning(
                f"[merge_audio] [{i+1:04d}/{total}] {filename} is empty "
                "(dry-run placeholder?) — inserting silence instead."
            )
            # Insert a 1-second silence placeholder so the merge doesn't crash
            segment_audio = AudioSegment.silent(duration=1000)
        else:
            logger.info(
                f"[merge_audio] [{i+1:04d}/{total}] Loading {filename}  "
                f"({meta.get('speaker','?')}, {meta.get('emotion','?')})"
            )
            segment_audio = AudioSegment.from_mp3(str(file_path))

        loaded.append((segment_audio, meta))

    logger.info(f"[merge_audio] All {total} segments loaded. Building final audio...")

    # ── Assemble with smart pauses ────────────────────────────────────────────
    final: AudioSegment = AudioSegment.empty()
    total_pause_ms = 0

    for i, (audio, meta) in enumerate(loaded):
        final += audio

        # Don't add a pause after the very last segment
        if i == len(loaded) - 1:
            break

        next_meta = loaded[i + 1][1]

        if fixed_pause_ms is not None:
            pause_ms = fixed_pause_ms
        else:
            pause_ms = _calculate_pause(meta, next_meta)

        total_pause_ms += pause_ms
        final += AudioSegment.silent(duration=pause_ms)

        logger.debug(
            f"[merge_audio] Pause after seg {i+1}: {pause_ms} ms  "
            f"({meta.get('type','?')} → {next_meta.get('type','?')}, "
            f"emotion={meta.get('emotion','?')})"
        )

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir = output_dir or FINAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{stem}.mp3"

    duration_sec = len(final) / 1000
    logger.info(
        f"[merge_audio] Total duration: {_format_duration(len(final))}  "
        f"({len(final):,} ms)  |  Silence added: {total_pause_ms:,} ms"
    )

    if dry_run:
        logger.info(f"[merge_audio] DRY-RUN — skipping file write. Would write: {output_path}")
    else:
        logger.info(f"[merge_audio] Exporting → {output_path}")
        final.export(str(output_path), format="mp3", bitrate="192k")
        logger.info(f"[merge_audio] ✅ Export complete: {output_path}")

    return output_path


# ════════════════════════════════════════════════════════════════════════════
# SMART PAUSE LOGIC
# ════════════════════════════════════════════════════════════════════════════

def _calculate_pause(current: dict, next_seg: dict) -> int:
    """
    Calculate the silence gap (ms) to insert after `current` segment,
    before `next_seg` begins.

    Uses a base pause from the type-transition table, then applies
    additive modifiers from the current segment's emotion and whether
    the speaker changes.
    """
    current_type = current.get("type", "narration")
    next_type    = next_seg.get("type", "narration")
    emotion      = current.get("emotion", "neutral")
    current_spk  = current.get("speaker", "")
    next_spk     = next_seg.get("speaker", "")

    # Base pause from transition table
    pause = BASE_PAUSE_MS.get((current_type, next_type), DEFAULT_PAUSE_MS)

    # Emotion modifier
    pause += EMOTION_MODIFIER_MS.get(emotion, 0)

    # Speaker change bonus
    if current_spk and next_spk and current_spk != next_spk:
        pause += SPEAKER_CHANGE_BONUS_MS

    # Never go below minimum
    return max(pause, MIN_PAUSE_MS)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _load_manifest(stem: str) -> dict:
    """Load and return the manifest.json for the given stem."""
    manifest_path = SEGMENTS_DIR / stem / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run Step 4 (tts_generator.py --input data/scripts/{stem}_clean.json) first."
        )

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse manifest.json: {e}")


def _check_pydub() -> None:
    """Raise a clear error if pydub is not installed."""
    try:
        import pydub  # noqa: F401
    except ImportError:
        raise ImportError(
            "pydub is required for audio merging.\n"
            "Run: pip install pydub\n"
            "Also ensure ffmpeg is installed on your system:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def _format_duration(ms: int) -> str:
    """Format milliseconds as HH:MM:SS for human-readable logging."""
    total_sec = ms // 1000
    hours     = total_sec // 3600
    minutes   = (total_sec % 3600) // 60
    seconds   = total_sec % 60
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge TTS audio segments into a final audiobook MP3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/audio/merge_audio.py --stem story\n"
            "  python src/audio/merge_audio.py --stem story --fixed-pause 400\n"
            "  python src/audio/merge_audio.py --stem story --dry-run -v\n"
            "  python src/audio/merge_audio.py --stem story --output-dir data/exports\n"
        ),
    )
    p.add_argument(
        "--stem", "-s", required=True,
        help="Story stem name (e.g. 'story' → reads data/audio_segments/story/).",
    )
    p.add_argument(
        "--fixed-pause", type=int, default=None,
        metavar="MS",
        help="Use a fixed pause (ms) between all segments instead of smart pauses.",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory. Default: data/final/.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate everything but do not write the output file.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging (shows each pause calculation).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    out_dir = Path(args.output_dir) if args.output_dir else None

    try:
        output_path = merge_audio(
            stem=args.stem,
            fixed_pause_ms=args.fixed_pause,
            output_dir=out_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ImportError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    if not args.dry_run:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n── Merge complete ──────────────────────────────────────")
        print(f"  Output  : {output_path}")
        print(f"  Size    : {size_mb:.2f} MB")
        print()
    else:
        print(f"\n── Dry-run complete — no file written ──────────────────")
        print(f"  Would write: {output_path}")
        print()


if __name__ == "__main__":
    main()