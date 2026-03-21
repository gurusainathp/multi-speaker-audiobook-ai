"""
main.py
────────────────────────────────────────────────────────────────
Top-level pipeline orchestrator for the audiobook generator.
Always run from the project root.

Current steps:
    [1] Load + extract text from input file   ✅
    [2] Detect dialogue / speakers / emotions  ✅
    [3] Clean script + assign voices           ✅

Planned steps:
    [4] TTS audio generation
    [5] Audio merge → final audiobook

Usage:
    python main.py --input data/input_files/story.pdf
    python main.py --input data/input_files/story.pdf --verbose
    python main.py --input data/input_files/story.pdf --skip-llm
"""

import argparse
import logging
import sys
from pathlib import Path


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_1_load_text(input_path: str, save: bool) -> str:
    """Step 1 — Load and extract clean text from the input file."""
    from src.io.text_loader import load_text
    return load_text(input_path, save=save)


def step_2_detect_dialogue(text: str, stem: str, save: bool) -> list:
    """Step 2 — Detect dialogue, speakers, and emotions via gpt-4o-mini."""
    from src.script_parser.dialogue_detector import detect_dialogue
    return detect_dialogue(text, save=save, stem=stem)


def step_3_clean_script(raw_script: list, stem: str, save: bool, skip_llm: bool) -> list:
    """Step 3 — Validate, clean, and assign voices via gpt-4.1-nano."""
    from src.script_parser.json_cleaner import clean_script
    return clean_script(raw_script, save=save, stem=stem, skip_llm=skip_llm)


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

def run_pipeline(input_path: str, save: bool, skip_llm: bool, verbose: bool) -> None:
    logger = logging.getLogger(__name__)
    stem = Path(input_path).stem

    print("\n" + "═" * 57)
    print("  🎙  Multi-Speaker Audiobook Generator")
    print("═" * 57)

    # ── Step 1 ───────────────────────────────────────────────
    print("\n[Step 1/5] Extracting text...")
    try:
        text = step_1_load_text(input_path, save=save)
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(str(e))
        sys.exit(1)

    output_txt = Path("data/extracted_text") / f"{stem}.txt"
    print(f"  ✅ Extracted {len(text):,} characters")
    if save:
        print(f"  💾 Saved  →  {output_txt}")

    # ── Step 2 ───────────────────────────────────────────────
    if skip_llm:
        print("\n[Step 2/5] Skipping dialogue detection (--skip-llm).")
        raw_script = []
    else:
        print(f"\n[Step 2/5] Detecting dialogue and speakers  (gpt-4o-mini)...")
        try:
            raw_script = step_2_detect_dialogue(text, stem=stem, save=save)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)

        output_raw = Path("data/scripts") / f"{stem}_raw.json"
        print(f"  ✅ Detected {len(raw_script)} segment(s)")
        if save:
            print(f"  💾 Saved  →  {output_raw}")

        print()
        for seg in raw_script[:3]:
            tag = f"[{seg.get('type','?').upper():<10}]"
            spk = f"{seg.get('speaker','?'):<12}"
            emo = f"({seg.get('emotion','?'):<14})"
            txt = seg.get('text','')[:52] + ("..." if len(seg.get('text','')) > 52 else "")
            print(f"    {tag} {spk} {emo}  {txt}")
        if len(raw_script) > 3:
            print(f"    ... and {len(raw_script) - 3} more segment(s)")

    # ── Step 3 ───────────────────────────────────────────────
    if skip_llm:
        print("\n[Step 3/5] Skipping script cleaning (--skip-llm).")
        clean = []
    else:
        print(f"\n[Step 3/5] Cleaning script + assigning voices  (gpt-4.1-nano)...")
        try:
            clean = step_3_clean_script(raw_script, stem=stem, save=save, skip_llm=False)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)

        output_clean = Path("data/scripts") / f"{stem}_clean.json"
        print(f"  ✅ Cleaned {len(clean)} segment(s)")
        if save:
            print(f"  💾 Saved  →  {output_clean}")

        # Voice cast summary
        seen: dict[str, str] = {}
        for seg in clean:
            sp = seg.get("speaker", "?")
            if sp not in seen:
                seen[sp] = seg.get("tts_voice", "?")
        print()
        print("  🎭 Voice cast:")
        for sp, v in seen.items():
            print(f"      {sp:<16}  →  {v}")

    # ── Step 4 ───────────────────────────────────────────────
    print("\n[Step 4/5] Generating TTS audio segments...")
    audio_files = step_4_generate_audio(clean)

    # ── Step 5 ───────────────────────────────────────────────
    final_output = f"data/final_audio/{stem}.mp3"
    print("\n[Step 5/5] Merging audio into final audiobook...")
    step_5_merge_audio(audio_files, final_output)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "─" * 57)
    print("  Pipeline run complete.")
    print(f"  Input  : {input_path}")
    if save and not skip_llm:
        print(f"  Text   : {output_txt}")
        if raw_script:
            print(f"  Raw    : {Path('data/scripts') / f'{stem}_raw.json'}")
        if clean:
            print(f"  Clean  : {Path('data/scripts') / f'{stem}_clean.json'}")
    print("─" * 57 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Speaker Emotional Audiobook Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input data/input_files/story.pdf\n"
            "  python main.py --input data/input_files/story.pdf --verbose\n"
            "  python main.py --input data/input_files/story.pdf --skip-llm\n"
            "  python main.py --input book.epub --no-save\n"
        ),
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to input file (PDF, TXT, DOCX, EPUB, HTML, MD, RTF).",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Do not save any output files to disk.",
    )
    p.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Steps 2 and 3 (no OpenAI calls). Useful for testing Step 1 only.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)
    run_pipeline(
        input_path=args.input,
        save=not args.no_save,
        skip_llm=args.skip_llm,
        verbose=args.verbose,
    )