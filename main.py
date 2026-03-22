"""
main.py
────────────────────────────────────────────────────────────────
Top-level pipeline orchestrator for the audiobook generator.
Always run from the project root.

Pipeline steps:
    1  load      — Extract text from input file
    2  detect    — Detect dialogue / speakers / emotions  (gpt-4o-mini)
    3  clean     — Validate, enrich, assign voices        (gpt-4o)
    4  tts       — Generate TTS audio segments            (gpt-4o-mini-tts)
    5  sfx       — Generate SFX / ambience audio          (gpt-4o-mini-tts)
    6  merge     — Merge all segments into final audiobook (pydub)

Step control:
    --from-step N   Start at step N  (loads saved output of step N-1 from disk)
    --to-step   N   Stop after step N (save and exit)

    Combine for any single step or range:
        --from-step 6 --to-step 6    merge only
        --from-step 5 --to-step 5    sfx only
        --from-step 2 --to-step 3    detect + clean only
        --to-step 1                  extract text only

Usage:
    # Full pipeline
    python main.py --input data/input_files/story.pdf

    # Single step
    python main.py --stem story --from-step 6
    python main.py --stem story --from-step 5 --to-step 5

    # Range
    python main.py --stem story --from-step 4 --to-step 5

    # Modifiers
    python main.py --input data/input_files/story.pdf --dry-run
    python main.py --input data/input_files/story.pdf --speed 0.95
    python main.py --stem story --from-step 6 --fixed-pause 400
"""

import argparse
import json
import logging
import sys
from pathlib import Path


# ── Step output locations ─────────────────────────────────────────────────────

def _txt_path(stem)        -> Path: return Path("data/extracted_text") / f"{stem}.txt"
def _raw_json_path(stem)   -> Path: return Path("data/scripts") / f"{stem}_raw.json"
def _clean_json_path(stem) -> Path: return Path("data/scripts") / f"{stem}_clean.json"
def _audio_dir(stem)       -> Path: return Path("data/audio_segments") / stem
def _manifest_path(stem)   -> Path: return _audio_dir(stem) / "manifest.json"
def _final_path(stem)      -> Path: return Path("data/final") / f"{stem}.mp3"

STEP_NAMES = {1: "load", 2: "detect", 3: "clean", 4: "tts", 5: "sfx", 6: "merge"}


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


# ── Step wrappers ─────────────────────────────────────────────────────────────

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


def step_5_generate_sfx(stem: str, dry_run: bool, resume: bool) -> list[Path]:
    from src.sfx.sfx_generator import generate_sfx
    return generate_sfx(stem=stem, dry_run=dry_run, resume=resume)


def step_6_merge_audio(stem: str, fixed_pause_ms: int | None, dry_run: bool) -> Path:
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
    return [
        audio_dir / seg["file"]
        for seg in data.get("segments", [])
        if seg.get("file")
    ]


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

    if from_step > to_step:
        logger.error(f"--from-step ({from_step}) cannot be greater than --to-step ({to_step}).")
        sys.exit(1)
    if from_step == 1 and not input_path:
        logger.error("--input is required when starting from step 1.")
        sys.exit(1)
    if from_step > 1 and not stem:
        logger.error("--stem is required when starting from step 2 or later.")
        sys.exit(1)

    if input_path and not stem:
        stem = Path(input_path).stem

    step_range = (
        f"Step {from_step} only"
        if from_step == to_step
        else f"Steps {from_step}–{to_step}"
    )
    step_labels = " → ".join(STEP_NAMES[i] for i in range(from_step, to_step + 1))

    print("\n" + "═" * 60)
    print("  🎙  Multi-Speaker Audiobook Generator")
    print(f"  Running: {step_range}  ({step_labels})")
    print("═" * 60)

    # carry variables
    text:        str        = ""
    raw_script:  list       = []
    clean:       list       = []
    audio_paths: list[Path] = []
    sfx_paths:   list[Path] = []
    final_path:  Path | None = None

    # ─────────────────────────────────────────────────────────
    # STEP 1 — Load text
    # ─────────────────────────────────────────────────────────
    if from_step <= 1 <= to_step:
        print(f"\n[Step 1/6 — load] Extracting text from: {input_path}")
        try:
            text = step_1_load_text(input_path, save=save)
        except (FileNotFoundError, ValueError, ImportError) as e:
            logger.error(str(e)); sys.exit(1)

        print(f"  ✅ Extracted {len(text):,} characters")
        if save:
            print(f"  💾 Saved  →  {_txt_path(stem)}")

    elif from_step > 1 and to_step >= 2:
        try:
            text = _load_text_from_disk(stem)
            logger.info(f"[resume] Loaded text from: {_txt_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e)); sys.exit(1)

    if to_step == 1:
        _print_summary(stem, from_step, to_step, input_path, text, [], [], [], [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 2 — Detect dialogue
    # ─────────────────────────────────────────────────────────
    if from_step <= 2 <= to_step:
        if not text:
            try:
                text = _load_text_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e)); sys.exit(1)

        print(f"\n[Step 2/6 — detect] Detecting dialogue, speakers, emotions  (gpt-4o-mini)...")
        try:
            raw_script = step_2_detect_dialogue(text, stem=stem, save=save)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e)); sys.exit(1)

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
            logger.info(f"[resume] Loaded raw script from: {_raw_json_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e)); sys.exit(1)

    if to_step == 2:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, [], [], [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 3 — Clean script
    # ─────────────────────────────────────────────────────────
    if from_step <= 3 <= to_step:
        if not raw_script:
            try:
                raw_script = _load_raw_script_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e)); sys.exit(1)

        print(f"\n[Step 3/6 — clean] Enriching script + assigning voices  (gpt-4o)...")
        try:
            clean = step_3_clean_script(raw_script, stem=stem, save=save)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e)); sys.exit(1)

        speech_segs = [s for s in clean if s.get("type") in ("narration", "dialogue")]
        sfx_segs    = [s for s in clean if s.get("type") in ("sfx", "ambience")]
        print(f"  ✅ {len(speech_segs)} speech  +  {len(sfx_segs)} sfx/ambience  =  {len(clean)} total segments")
        if save:
            print(f"  💾 Saved  →  {_clean_json_path(stem)}")

        seen: dict[str, str] = {}
        for seg in clean:
            sp = seg.get("speaker", "?")
            if sp not in seen and seg.get("tts_voice"):
                seen[sp] = seg.get("tts_voice", "?")
        if seen:
            print()
            print("  🎭 Voice cast:")
            for sp, v in seen.items():
                print(f"      {sp:<16}  →  {v}")

        if sfx_segs:
            print()
            print("  🔊 SFX / ambience:")
            for seg in sfx_segs:
                print(f"      [{seg['type'].upper():<10}]  {seg.get('sound','?')}")

    elif from_step > 3 and to_step >= 4:
        try:
            clean = _load_clean_script_from_disk(stem)
            logger.info(f"[resume] Loaded clean script from: {_clean_json_path(stem)}")
        except FileNotFoundError as e:
            logger.error(str(e)); sys.exit(1)

    if to_step == 3:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, [], [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 4 — TTS generation
    # ─────────────────────────────────────────────────────────
    if from_step <= 4 <= to_step:
        if not clean:
            try:
                clean = _load_clean_script_from_disk(stem)
            except FileNotFoundError as e:
                logger.error(str(e)); sys.exit(1)

        dry_tag = "  ⚠️  DRY-RUN" if dry_run else ""
        print(f"\n[Step 4/6 — tts] Generating speech segments  (gpt-4o-mini-tts)...{dry_tag}")
        try:
            audio_paths = step_4_generate_audio(
                clean, stem=stem, speed=speed,
                dry_run=dry_run, resume=not no_resume,
            )
        except (EnvironmentError, RuntimeError, ValueError) as e:
            logger.error(str(e)); sys.exit(1)

        print(f"  ✅ Generated {len(audio_paths)} speech file(s)")
        print(f"  💾 Saved  →  {_audio_dir(stem)}/")
        print(f"  📋 Manifest →  {_manifest_path(stem)}")

    elif from_step > 4 and to_step >= 5:
        try:
            audio_paths = _load_audio_paths_from_disk(stem)
            logger.info(f"[resume] Loaded {len(audio_paths)} audio path(s) from manifest.")
        except FileNotFoundError as e:
            logger.error(str(e)); sys.exit(1)

    if to_step == 4:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, audio_paths, [], None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 5 — SFX generation
    # ─────────────────────────────────────────────────────────
    if from_step <= 5 <= to_step:
        dry_tag = "  ⚠️  DRY-RUN" if dry_run else ""
        print(f"\n[Step 5/6 — sfx] Generating SFX / ambience audio  (gpt-4o-mini-tts)...{dry_tag}")
        try:
            sfx_paths = step_5_generate_sfx(
                stem=stem,
                dry_run=dry_run,
                resume=not no_resume,
            )
        except (FileNotFoundError, EnvironmentError, RuntimeError) as e:
            logger.error(str(e)); sys.exit(1)

        if sfx_paths:
            print(f"  ✅ Generated {len(sfx_paths)} SFX file(s)")
            print(f"  📋 Manifest updated  →  {_manifest_path(stem)}")
        else:
            print(f"  ✅ No sfx/ambience segments found — nothing to generate.")

    if to_step == 5:
        _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, audio_paths, sfx_paths, None, save, dry_run)
        return

    # ─────────────────────────────────────────────────────────
    # STEP 6 — Merge
    # ─────────────────────────────────────────────────────────
    if from_step <= 6 <= to_step:
        pause_desc = f"{fixed_pause_ms} ms fixed" if fixed_pause_ms else "smart pauses"
        dry_tag    = "  ⚠️  DRY-RUN" if dry_run else ""
        print(f"\n[Step 6/6 — merge] Merging all audio  ({pause_desc})...{dry_tag}")
        try:
            final_path = step_6_merge_audio(
                stem=stem,
                fixed_pause_ms=fixed_pause_ms,
                dry_run=dry_run,
            )
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            logger.error(str(e)); sys.exit(1)

        if not dry_run and final_path.exists():
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Audiobook created!")
            print(f"  🎧 Saved  →  {final_path}  ({size_mb:.2f} MB)")
        else:
            print(f"  ✅ Dry-run — would write: {final_path}")

    _print_summary(stem, from_step, to_step, input_path, text, raw_script, clean, audio_paths, sfx_paths, final_path, save, dry_run)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    stem, from_step, to_step, input_path,
    text, raw_script, clean, audio_paths, sfx_paths, final_path,
    save, dry_run,
):
    print("\n" + "═" * 60)
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
        speech = sum(1 for s in clean if s.get("type") in ("narration","dialogue"))
        sfx    = sum(1 for s in clean if s.get("type") in ("sfx","ambience"))
        print(f"  Clean    : {_clean_json_path(stem)}  ({speech} speech + {sfx} sfx)")
    if audio_paths and from_step <= 4:
        print(f"  Speech   : {_audio_dir(stem)}/  ({len(audio_paths)} files)")
    if sfx_paths and from_step <= 5:
        print(f"  SFX      : {_audio_dir(stem)}/  ({len(sfx_paths)} files)")
    if final_path and not dry_run and final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"  🎧 Final : {final_path}  ({size_mb:.2f} MB)")
    print("═" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Speaker Emotional Audiobook Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Step names:  1=load  2=detect  3=clean  4=tts  5=sfx  6=merge\n"
            "\n"
            "Examples:\n"
            "  # Full pipeline\n"
            "  python main.py --input data/input_files/story.pdf\n"
            "\n"
            "  # Extract text only\n"
            "  python main.py --input data/input_files/story.pdf --to-step 1\n"
            "\n"
            "  # Re-run SFX + merge only\n"
            "  python main.py --stem story --from-step 5\n"
            "\n"
            "  # Re-run merge only with fixed pauses\n"
            "  python main.py --stem story --from-step 6 --fixed-pause 400\n"
            "\n"
            "  # Dry-run full pipeline\n"
            "  python main.py --input data/input_files/story.pdf --dry-run\n"
        ),
    )

    src = p.add_argument_group("Input (source)")
    src.add_argument("--input", "-i", help="Path to input file. Required when --from-step=1.")
    src.add_argument("--stem",  "-s", help="Story stem name. Required when --from-step > 1.")

    steps = p.add_argument_group("Step control")
    steps.add_argument("--from-step", type=int, default=1, metavar="N", choices=range(1, 7),
                       help="Start pipeline at step N (1–6). Default: 1.")
    steps.add_argument("--to-step",   type=int, default=6, metavar="N", choices=range(1, 7),
                       help="Stop pipeline after step N (1–6). Default: 6.")

    tts_opts = p.add_argument_group("TTS options (Step 4)")
    tts_opts.add_argument("--speed",     type=float, default=1.0,
                          help="TTS playback speed (0.25–4.0). Default: 1.0.")
    tts_opts.add_argument("--no-resume", action="store_true",
                          help="Regenerate all segments even if files already exist.")
    tts_opts.add_argument("--dry-run",   action="store_true",
                          help="Simulate Steps 4–6 without real API calls or file writes.")

    merge_opts = p.add_argument_group("Merge options (Step 6)")
    merge_opts.add_argument("--fixed-pause", type=int, default=None, metavar="MS",
                            help="Fixed pause (ms) between segments instead of smart pauses.")

    general = p.add_argument_group("General")
    general.add_argument("--no-save", action="store_true",
                         help="Do not save intermediate files to disk.")
    general.add_argument("--verbose", "-v", action="store_true",
                         help="Enable debug logging.")

    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

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