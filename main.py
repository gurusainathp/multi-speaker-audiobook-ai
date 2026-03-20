"""
main.py
────────────────────────────────────────────────────────────────
Top-level pipeline orchestrator for the audiobook generator.

Run from the project root. This is the single command to kick off
the full pipeline. Each step will be added here as it is built.

Current steps:
    [1] Load + extract text from input file  ✅

Planned steps (will be wired in as they are implemented):
    [2] Dialogue / speaker / emotion detection
    [3] JSON cleanup and validation
    [4] TTS audio generation
    [5] Audio merge → final audiobook

Usage:
    python main.py --input data/input_pdfs/story.pdf
    python main.py --input data/input_pdfs/story.pdf --verbose
    python main.py --input book.epub --no-save
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Logging setup (before any src imports so all modules inherit it) ─────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=level,
    )

# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_1_load_text(input_path: str, save: bool) -> str:
    """
    Step 1 — Load and extract clean text from the input file.
    Supports: PDF, TXT, DOCX, EPUB, HTML, MD, RTF.
    """
    from src.io.text_loader import load_text
    return load_text(input_path, save=save)


# ─────────────────────────────────────────────────────────────────────────────
# Stub placeholders — these will be replaced when each step is implemented.
# ─────────────────────────────────────────────────────────────────────────────

def step_2_detect_dialogue(text: str) -> list:
    """Step 2 — Detect dialogue, speakers, and emotions (gpt-4o-mini)."""
    # TODO: implement in src/script_parser/dialogue_detector.py
    logging.getLogger(__name__).warning(
        "[Step 2] Dialogue detection not yet implemented — skipping."
    )
    return []


def step_3_clean_script(raw_script: list) -> list:
    """Step 3 — Validate and clean the structured JSON script (gpt-4.1-nano)."""
    # TODO: implement in src/script_parser/json_cleaner.py
    logging.getLogger(__name__).warning(
        "[Step 3] JSON cleaning not yet implemented — skipping."
    )
    return []


def step_4_generate_audio(script: list) -> list:
    """Step 4 — Generate per-segment audio files via OpenAI TTS."""
    # TODO: implement in src/tts/tts_generator.py
    logging.getLogger(__name__).warning(
        "[Step 4] TTS generation not yet implemented — skipping."
    )
    return []


def step_5_merge_audio(audio_files: list, output_path: str) -> str:
    """Step 5 — Merge audio segments into the final audiobook MP3."""
    # TODO: implement in src/audio/merge_audio.py
    logging.getLogger(__name__).warning(
        "[Step 5] Audio merge not yet implemented — skipping."
    )
    return ""


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, save: bool, verbose: bool) -> None:
    logger = logging.getLogger(__name__)

    print("\n" + "═" * 55)
    print("  🎙  Multi-Speaker Audiobook Generator")
    print("═" * 55)

    # ── Step 1 ───────────────────────────────────────────────
    print("\n[Step 1/5] Extracting text...")
    try:
        text = step_1_load_text(input_path, save=save)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)

    output_stem = Path(input_path).stem
    output_file = Path("data/extracted_text") / f"{output_stem}.txt"

    print(f"  ✅ Extracted {len(text):,} characters")
    if save:
        print(f"  💾 Saved to: {output_file}")

    # ── Step 2 ───────────────────────────────────────────────
    print("\n[Step 2/5] Detecting dialogue and speakers...")
    raw_script = step_2_detect_dialogue(text)

    # ── Step 3 ───────────────────────────────────────────────
    print("\n[Step 3/5] Cleaning and validating script JSON...")
    clean_script = step_3_clean_script(raw_script)

    # ── Step 4 ───────────────────────────────────────────────
    print("\n[Step 4/5] Generating TTS audio segments...")
    audio_files = step_4_generate_audio(clean_script)

    # ── Step 5 ───────────────────────────────────────────────
    final_output = f"data/final_audio/{output_stem}.mp3"
    print("\n[Step 5/5] Merging audio into final audiobook...")
    step_5_merge_audio(audio_files, final_output)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Pipeline complete.")
    print(f"  Input:  {input_path}")
    if save:
        print(f"  Text:   {output_file}")
    print("─" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Speaker Emotional Audiobook Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input data/input_pdfs/story.pdf\n"
            "  python main.py --input data/input_pdfs/story.pdf --verbose\n"
            "  python main.py --input book.epub --no-save\n"
        ),
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input file (PDF, TXT, DOCX, EPUB, HTML, MD, RTF).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the extracted text file to disk.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)
    run_pipeline(
        input_path=args.input,
        save=not args.no_save,
        verbose=args.verbose,
    )