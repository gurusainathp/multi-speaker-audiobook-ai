"""
main.py
────────────────────────────────────────────────────────────────
Top-level pipeline orchestrator for the audiobook generator.
Always run from the project root.

Pipeline steps:
    1  load      — Extract text from input file
    2  detect    — Detect dialogue / speakers / emotions  (gpt-4o-mini)
    3  clean     — Validate script + assign voices        (gpt-4.1-nano)
    4  tts       — Generate TTS audio segments            (gpt-4o-mini-tts)
    5  merge     — Merge segments into final audiobook    (pydub)

Step control:
    --from-step N   Start at step N  (load saved output of step N-1 from disk)
    --to-step   N   Stop after step N (save and exit)

    These can be combined to run any single step or range:
        --from-step 5 --to-step 5    run merge only
        --from-step 2 --to-step 3    run detect + clean only
        --to-step 1                  extract text only

    When jumping into a mid-pipeline step, main.py loads the required
    input automatically from the standard output location of the
    previous step — no extra flags needed.

Usage:
    # Full pipeline
    python main.py --input data/input_files/story.pdf

    # Run a single step
    python main.py --stem story --from-step 5 --to-step 5
    python main.py --input data/input_files/story.pdf --to-step 1

    # Run a range
    python main.py --stem story --from-step 2 --to-step 3

    # With modifiers
    python main.py --input data/input_files/story.pdf --dry-run
    python main.py --input data/input_files/story.pdf --speed 0.95
    python main.py --stem story --from-step 4 --to-step 5 --fixed-pause 400
"""

import argparse
import json
import logging
import sys
from pathlib import Path


# ── Constants — standard file locations for each step's output ───────────────

def _txt_path(stem: str)        -> Path: return Path("data/extracted_text") / f"{stem}.txt"
def _raw_json_path(stem: str)   -> Path: return Path("data/scripts") / f"{stem}_raw.json"
def _clean_json_path(stem: str) -> Path: return Path("data/scripts") / f"{stem}_clean.json"
def _audio_dir(stem: str)       -> Path: return Path("data/audio_segments") / stem
def _manifest_path(stem: str)   -> Path: return _audio_dir(stem) / "manifest.json"
def _final_path(stem: str)      -> Path: return Path("data/final") / f"{stem}.mp3"

STEP_NAMES = {1: "load", 2: "detect", 3: "clean", 4: "tts", 5: "merge"}


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


# ── Individual step wrappers ──────────────────────────────────────────────────

def step_1_load_text(input_path: str, save: bool) -> str:
    from src.io.text_loader import load_text
    return load_text(input_path, save=save)


def step_2_detect_dialogue(text: str, stem: str, save: bool) -> list:
    from src.script_parser.dialogue_detector import detect_dialogue
    return detect_dialogue(text, save=save, stem=stem)


def step_3_clean_script(raw_script: list, stem: str, save: bool) -> list:
    from src.script_parser.json_cleaner import clean_script
    return clean_script(raw_script, save=save, stem=stem, skip_llm=False)


def step_4_generate_audio(
    script: list, stem: str, speed: float, dry_run: bool, resume: bool
) -> list[Path]:
    from src.tts.tts_generator import generate_audio
    return generate_audio(script, stem=stem, speed=speed, dry_run=dry_run, resume=resume)


def step_5_merge_audio(stem: str, fixed_pause_ms: int | None, dry_run: bool) -> Path:
    from src.audio.merge_audio import merge_audio
    return merge_audio(stem=stem, fixed_pause_ms=fixed_pause_ms, dry_run=dry_run)


# ── Disk loaders (used when jumping into a mid-pipeline step) ─────────────────

def _load_text_from_disk(stem: str) -> str:
    path = _txt_path(stem)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected extracted text at: {path}\n"
            f"Run step 1 first:  python main.py --input <file> --to-step 1"
        )
    return path.read_text(encoding="utf-8")


def _load_raw_script_from_disk(stem: str) -> list:
    path = _raw_json_path(stem)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected raw script at: {path}\n"
            f"Run step 2 first:  python main.py --stem {stem} --from-step 2 --to-step 2"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_clean_script_from_disk(stem: str) -> list:
    path = _clean_json_path(stem)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected clean script at: {path}\n"
            f"Run step 3 first:  python main.py --stem {stem} --from-step 3 --to-step 3"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_audio_paths_from_disk(stem: str) -> list[Path]:
    manifest = _manifest_path(stem)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Expected manifest at: {manifest}\n"
            f"Run step 4 first:  python main.py --stem {stem} --from-step 4 --to-step 4"
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    audio_dir = _audio_dir(stem)
    return [audio_dir / seg["file"] for seg in data.get("segments", [])]


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    stem: str,
    input_path: str | None,
    from_step: int,
    to_step: int,
    save: bool,
    dry_run: bool,
    speed: float,
    no_resume: bool,
    fixed_pause_ms: int | None,
    verbose: bool,
) -> None:
    logger = logging.getLogger(__name__)

    # Validate step range
    if from_step > to_step:
        logger.error(f"--from-step ({from_step}) cannot be greater than --to-step ({to_step}).")
        sys.exit(1)

    # Step 1 requires --input; mid-pipeline steps use --stem
    if from_step == 1 and not input_path:
        logger.error("--input is required when starting from step 1.")
        sys.exit(1)
    if from_step > 1 and not stem:
        logger.error("--stem is required when starting from step 2 or later.")
        sys.exit(1)

    # Derive stem from input_path if not explicitly given
    if input_path and not stem:
        stem = Path(input_path).stem

    # Build the step range label for the header
    step_range = (
        f"Step {from_step} only"
        if from_step == to_step
        else f"Steps {from_step}–{to_step}"
    )
    step_labels = " → ".join(
        STEP_NAMES[i] for i in range(from_step, to_step + 1)
    )

    print("\n" + "═" * 57)
    print("  🎙  Multi-Speaker Audiobook Generator")
    print(f"  Running: {step_range}  ({step_labels})")
    print("═" * 57)

    # Carry variables — populated as steps complete or loaded from disk
    text:         str         = ""
    raw_script:   list        = []
    clean:        list        = []
    audio_paths:  list[Path]  = []
    final_path:   Path | None = None

    # ─────────────────────────────────────────────────────────
    # STEP 1 — Load text
    # ─────────────────────────────────────────────────────────
    if from_step <= 1 <= to_step:
        print(f"\n[Step 1/5 — load] Extracting text from: {input_path}")
        try:
            text = step_1_load_text(input_path, save=save)
        except (FileNotFoundError, ValueError, ImportError) as e:
            logger.error(str(e))
            sys.exit(1)

        print(f"  ✅ Extracted {len(text):,} characters")
        if save:
            print(f"  💾 Saved  →  {_txt_path(stem)}")

    elif from_step > 1 and to_step >= 2:
        # Step 2+ needs the text — load it from disk
        try:
            text = _load_text_from_disk(stem)
            logger.info(f"[resume] Loaded text from disk: {_txt_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if to_step == 1:
        _print_summary(stem, from_step, to_step, input_path, text, [], [], [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 2 — Detect dialogue
    # ─────────────────────────────────────────────────────────
    if from_step <= 2 <= to_step:
        # Need text — either from step 1 above, or load from disk
        if not text:
            try:
                text = _load_text_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e))
                sys.exit(1)

        print(f"\n[Step 2/5 — detect] Detecting dialogue, speakers, emotions  (gpt-4o-mini)...")
        try:
            raw_script = step_2_detect_dialogue(text, stem=stem, save=save)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)

        print(f"  ✅ Detected {len(raw_script)} segment(s)")
        if save:
            print(f"  💾 Saved  →  {_raw_json_path(stem)}")
        print()
        for seg in raw_script[:3]:
            tag = f"[{seg.get('type','?').upper():<10}]"
            spk = f"{seg.get('speaker','?'):<12}"
            emo = f"({seg.get('emotion','?'):<14})"
            txt = seg.get('text','')[:52] + ("..." if len(seg.get('text','')) > 52 else "")
            print(f"    {tag} {spk} {emo}  {txt}")
        if len(raw_script) > 3:
            print(f"    ... and {len(raw_script) - 3} more segment(s)")

    elif from_step > 2 and to_step >= 3:
        try:
            raw_script = _load_raw_script_from_disk(stem)
            logger.info(f"[resume] Loaded raw script from disk: {_raw_json_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if to_step == 2:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, [], [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 3 — Clean script
    # ─────────────────────────────────────────────────────────
    if from_step <= 3 <= to_step:
        if not raw_script:
            try:
                raw_script = _load_raw_script_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e))
                sys.exit(1)

        print(f"\n[Step 3/5 — clean] Cleaning script + assigning voices  (gpt-4.1-nano)...")
        try:
            clean = step_3_clean_script(raw_script, stem=stem, save=save)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)

        print(f"  ✅ Cleaned {len(clean)} segment(s)")
        if save:
            print(f"  💾 Saved  →  {_clean_json_path(stem)}")
        seen: dict[str, str] = {}
        for seg in clean:
            sp = seg.get("speaker", "?")
            if sp not in seen:
                seen[sp] = seg.get("tts_voice", "?")
        print()
        print("  🎭 Voice cast:")
        for sp, v in seen.items():
            print(f"      {sp:<16}  →  {v}")

    elif from_step > 3 and to_step >= 4:
        try:
            clean = _load_clean_script_from_disk(stem)
            logger.info(f"[resume] Loaded clean script from disk: {_clean_json_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if to_step == 3:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 4 — TTS generation
    # ─────────────────────────────────────────────────────────
    if from_step <= 4 <= to_step:
        if not clean:
            try:
                clean = _load_clean_script_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e))
                sys.exit(1)

        dry_tag = "  ⚠️  DRY-RUN" if dry_run else ""
        print(f"\n[Step 4/5 — tts] Generating TTS audio segments  (gpt-4o-mini-tts)...{dry_tag}")
        try:
            audio_paths = step_4_generate_audio(
                clean, stem=stem, speed=speed,
                dry_run=dry_run, resume=not no_resume,
            )
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)

        audio_dir = _audio_dir(stem)
        print(f"  ✅ Generated {len(audio_paths)} audio file(s)")
        print(f"  💾 Saved  →  {audio_dir}/")
        print(f"  📋 Manifest →  {_manifest_path(stem)}")

    elif from_step > 4 and to_step >= 5:
        try:
            audio_paths = _load_audio_paths_from_disk(stem)
            logger.info(f"[resume] Loaded {len(audio_paths)} audio path(s) from manifest.")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if to_step == 4:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, audio_paths, None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 5 — Merge
    # ─────────────────────────────────────────────────────────
    if from_step <= 5 <= to_step:
        pause_desc = f"{fixed_pause_ms} ms fixed" if fixed_pause_ms else "smart pauses"
        dry_tag    = "  ⚠️  DRY-RUN" if dry_run else ""
        print(f"\n[Step 5/5 — merge] Merging audio  ({pause_desc})...{dry_tag}")
        try:
            final_path = step_5_merge_audio(
                stem=stem,
                fixed_pause_ms=fixed_pause_ms,
                dry_run=dry_run,
            )
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            logger.error(str(e))
            sys.exit(1)

        if not dry_run and final_path.exists():
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Audiobook created!")
            print(f"  💾 Saved  →  {final_path}  ({size_mb:.2f} MB)")
        else:
            print(f"  ✅ Dry-run — would write: {final_path}")

    _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, audio_paths, final_path, save, dry_run)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(
    stem, from_step, to_step, input_path,
    text, raw_script, clean, audio_paths, final_path,
    save, dry_run,
):
    print("\n" + "═" * 57)
    print("  ✅ Done.")
    if input_path:
        print(f"  Input    : {input_path}")
    print(f"  Stem     : {stem}")
    print(f"  Steps run: {from_step}–{to_step}")

    if text and from_step <= 1:
        print(f"  Text     : {_txt_path(stem)}")
    if raw_script and from_step <= 2:
        print(f"  Raw      : {_raw_json_path(stem)}")
    if clean and from_step <= 3:
        print(f"  Clean    : {_clean_json_path(stem)}")
    if audio_paths and from_step <= 4:
        print(f"  Segments : {_audio_dir(stem)}/  ({len(audio_paths)} files)")
    if final_path and not dry_run and final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"  🎧 Final : {final_path}  ({size_mb:.2f} MB)")
    print("═" * 57 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Speaker Emotional Audiobook Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Step names:  1=load  2=detect  3=clean  4=tts  5=merge\n"
            "\n"
            "Examples:\n"
            "  # Full pipeline\n"
            "  python main.py --input data/input_files/story.pdf\n"
            "\n"
            "  # Extract text only\n"
            "  python main.py --input data/input_files/story.pdf --to-step 1\n"
            "\n"
            "  # Run detect + clean only (text already extracted)\n"
            "  python main.py --stem story --from-step 2 --to-step 3\n"
            "\n"
            "  # Re-run merge only (TTS already done)\n"
            "  python main.py --stem story --from-step 5\n"
            "\n"
            "  # Re-run TTS + merge with slower speed\n"
            "  python main.py --stem story --from-step 4 --speed 0.9\n"
            "\n"
            "  # Dry-run full pipeline (no TTS billing)\n"
            "  python main.py --input data/input_files/story.pdf --dry-run\n"
        ),
    )

    # ── Input (one of these required depending on from_step)
    src = p.add_argument_group("Input (source)")
    src.add_argument(
        "--input", "-i",
        help="Path to input file. Required when --from-step=1 (default).",
    )
    src.add_argument(
        "--stem", "-s",
        help="Story stem name (e.g. 'story'). Required when --from-step > 1.",
    )

    # ── Step control
    steps = p.add_argument_group("Step control")
    steps.add_argument(
        "--from-step", type=int, default=1, metavar="N", choices=range(1, 6),
        help="Start pipeline at step N (1–5). Default: 1.",
    )
    steps.add_argument(
        "--to-step", type=int, default=5, metavar="N", choices=range(1, 6),
        help="Stop pipeline after step N (1–5). Default: 5.",
    )

    # ── Step 4 options
    tts_opts = p.add_argument_group("TTS options (Step 4)")
    tts_opts.add_argument(
        "--speed", type=float, default=1.0,
        help="TTS playback speed (0.25–4.0). Default: 1.0.",
    )
    tts_opts.add_argument(
        "--no-resume", action="store_true",
        help="Regenerate all audio segments even if files already exist.",
    )
    tts_opts.add_argument(
        "--dry-run", action="store_true",
        help="Simulate Steps 4–5 without real API calls or file writes.",
    )

    # ── Step 5 options
    merge_opts = p.add_argument_group("Merge options (Step 5)")
    merge_opts.add_argument(
        "--fixed-pause", type=int, default=None, metavar="MS",
        help="Fixed pause (ms) between segments instead of smart pauses.",
    )

    # ── General
    general = p.add_argument_group("General")
    general.add_argument(
        "--no-save", action="store_true",
        help="Do not save intermediate files to disk.",
    )
    general.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    # Validate: need --input for step 1, --stem for step 2+
    if args.from_step == 1 and not args.input:
        print("[ERROR] --input is required when starting from step 1.")
        sys.exit(1)
    if args.from_step > 1 and not args.stem:
        print("[ERROR] --stem is required when --from-step > 1.")
        sys.exit(1)

    run_pipeline(
        stem=args.stem or "",
        input_path=args.input,
        from_step=args.from_step,
        to_step=args.to_step,
        save=not args.no_save,
        dry_run=args.dry_run,
        speed=args.speed,
        no_resume=args.no_resume,
        fixed_pause_ms=args.fixed_pause,
        verbose=args.verbose,
    )