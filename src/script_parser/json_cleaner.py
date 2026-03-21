"""
src/script_parser/json_cleaner.py
────────────────────────────────────────────────────────────────
Step 3 of the audiobook pipeline: raw script JSON → clean, TTS-safe script.

Takes the raw list[dict] from detect_dialogue() and:
  - Fixes schema errors (missing fields, invalid values)
  - Assigns a consistent "voice" field per character
  - Guarantees every segment is safe to pass directly to TTS

Two-pass design:
  Pass 1 (local)  — fast, free Python validation. Catches obvious issues
                    without spending an API call.
  Pass 2 (LLM)   — gpt-4.1-nano cleans anything Pass 1 can't fix:
                    ambiguous genders, inferred emotions, structural fixes.
  Pass 3 (local)  — final schema re-validation after the LLM pass.
                    Guarantees output is always safe for TTS regardless
                    of what the model returned.

Usage (import):
    from src.script_parser.json_cleaner import clean_script

    raw  = detect_dialogue(text)
    clean = clean_script(raw)          # list[dict] with "voice" field added

Usage (CLI):
    python src/script_parser/json_cleaner.py -i data/scripts/story_raw.json
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

from dotenv import load_dotenv

from src.script_parser.prompts import CLEANER_SYSTEM_PROMPT, CLEANER_USER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL       = "gpt-4.1-nano"
MAX_TOKENS  = 4096
TEMPERATURE = 0.1          # Near-zero: we want deterministic schema fixes
OUTPUT_DIR  = Path("data/scripts")

VALID_TYPES = {"narration", "dialogue"}
VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "excited", "melancholic", "contemplative",
    "tense", "gentle", "bitter", "hopeful",
}

# OpenAI TTS voice names — mapped from logical voice slots
# These are the actual values TTS will use in Step 4
VOICE_MAP = {
    "narrator":  "onyx",
    "male1":     "echo",
    "male2":     "fable",
    "male3":     "onyx",
    "female1":   "nova",
    "female2":   "shimmer",
    "female3":   "alloy",
    "neutral1":  "alloy",
    "neutral2":  "echo",
}


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def clean_script(
    raw_script: list,
    save: bool = True,
    stem: str = "story",
    skip_llm: bool = False,
) -> list:
    """
    Validate, repair, and enrich a raw script list.

    Args:
        raw_script: Output of detect_dialogue() — list of segment dicts.
        save:       If True, saves cleaned script to data/scripts/.
        stem:       Filename stem (e.g. "story" → "story_clean.json").
        skip_llm:   If True, run local validation only (no API call).
                    Useful for testing or when offline.

    Returns:
        Cleaned list of dicts. Each dict is guaranteed to have:
        speaker, type, emotion, text, voice, tts_voice

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set (and skip_llm=False).
        RuntimeError:     If the API call fails.
    """
    if not raw_script:
        raise ValueError("raw_script is empty — nothing to clean.")

    logger.info(f"[clean_script] Input: {len(raw_script)} segment(s)")

    # ── Pass 1: local pre-validation ─────────────────────────────────────────
    pre_validated = [
        seg for i, seg in enumerate(
            (_local_validate(s, i) for i, s in enumerate(raw_script))
        ) if seg is not None
    ]
    logger.info(f"[clean_script] After local pre-validation: {len(pre_validated)} segment(s)")

    # ── Pass 2: LLM cleanup (assigns voice, fixes ambiguities) ───────────────
    if skip_llm:
        logger.info("[clean_script] Skipping LLM pass (skip_llm=True).")
        llm_cleaned = pre_validated
    else:
        _check_api_key()
        logger.info(f"[clean_script] Sending to {MODEL} for cleanup and voice assignment...")
        llm_cleaned = _call_openai(pre_validated)
        logger.info(f"[clean_script] After LLM pass: {len(llm_cleaned)} segment(s)")

    # ── Pass 3: local post-validation + tts_voice injection ──────────────────
    # This guarantees TTS safety regardless of what the LLM returned.
    final = _post_validate_all(llm_cleaned)
    logger.info(f"[clean_script] Final clean segment count: {len(final)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if save:
        out = _save_script(final, stem=f"{stem}_clean")
        logger.info(f"[clean_script] Saved cleaned script to: {out}")

    return final


# ════════════════════════════════════════════════════════════════════════════
# PASS 1 — Local pre-validation
# ════════════════════════════════════════════════════════════════════════════

def _local_validate(seg: dict, index: int) -> dict | None:
    """
    Cheap local check before sending to the LLM.
    Fixes obvious issues; drops unrecoverable segments.
    Does NOT assign voice — that's the LLM's job.
    """
    if not isinstance(seg, dict):
        logger.warning(f"[pre-validate] Segment {index}: not a dict — dropped.")
        return None

    text = str(seg.get("text", "")).strip()
    if not text:
        logger.warning(f"[pre-validate] Segment {index}: empty text — dropped.")
        return None

    speaker = str(seg.get("speaker", "Narrator")).strip() or "Narrator"

    seg_type = str(seg.get("type", "")).lower()
    if seg_type not in VALID_TYPES:
        seg_type = "dialogue" if speaker != "Narrator" else "narration"

    emotion = str(seg.get("emotion", "neutral")).lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    # Pass through existing voice if already set (e.g. from a previous run)
    result = {
        "speaker": speaker,
        "type":    seg_type,
        "emotion": emotion,
        "text":    text,
    }
    if "voice" in seg and seg["voice"]:
        result["voice"] = str(seg["voice"])

    return result


# ════════════════════════════════════════════════════════════════════════════
# PASS 2 — LLM cleanup
# ════════════════════════════════════════════════════════════════════════════

def _call_openai(script: list) -> list:
    """
    Send the pre-validated script to gpt-4.1-nano for cleanup.
    Returns the parsed, corrected list.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Run: pip install openai")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    script_str = json.dumps(script, indent=2, ensure_ascii=False)
    user_message = CLEANER_USER_PROMPT.replace("<<SCRIPT>>", script_str)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": CLEANER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    raw_content = response.choices[0].message.content.strip()
    logger.debug(f"[_call_openai] Raw response preview:\n{raw_content[:300]}")

    return _parse_json_response(raw_content)


def _parse_json_response(content: str) -> list:
    """Parse model response — strips fences, handles dict wrapping."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[_parse_json_response] JSON parse failed: {e}")
        logger.error(f"[_parse_json_response] Raw:\n{content[:600]}")
        raise RuntimeError(
            f"gpt-4.1-nano returned unparseable JSON. Error: {e}\n"
            f"Preview: {content[:200]}"
        )

    if isinstance(parsed, dict):
        parsed = parsed.get("segments", [parsed])
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected JSON array, got: {type(parsed).__name__}")

    return parsed


# ════════════════════════════════════════════════════════════════════════════
# PASS 3 — Post-validation + tts_voice injection
# ════════════════════════════════════════════════════════════════════════════

def _post_validate_all(script: list) -> list:
    """
    Final local pass after the LLM.
    - Re-validates all fields
    - Ensures "voice" is set for every segment (local fallback if LLM missed it)
    - Injects "tts_voice" — the actual OpenAI TTS voice name Step 4 will use
    - Enforces voice consistency: same speaker always gets same voice
    """
    # Build a speaker→voice map from whatever the LLM assigned
    # so we can be consistent for speakers the LLM already handled
    speaker_voice_map: dict[str, str] = {"Narrator": "narrator"}
    male_counter   = 1
    female_counter = 1
    neutral_counter = 1

    def _assign_voice(speaker: str, existing_voice: str | None) -> str:
        nonlocal male_counter, female_counter, neutral_counter

        if speaker in speaker_voice_map:
            return speaker_voice_map[speaker]

        # Use the LLM's assignment if it looks valid
        if existing_voice and existing_voice in VOICE_MAP:
            speaker_voice_map[speaker] = existing_voice
            return existing_voice

        # Local fallback: derive from voice hint in the LLM response
        if existing_voice:
            v = existing_voice.lower()
            if v.startswith("male"):
                slot = f"male{male_counter}"
                if male_counter < 3:
                    male_counter += 1
                speaker_voice_map[speaker] = slot
                return slot
            if v.startswith("female"):
                slot = f"female{female_counter}"
                if female_counter < 3:
                    female_counter += 1
                speaker_voice_map[speaker] = slot
                return slot

        # Final fallback: guess from speaker name conventions
        # (very rough — LLM should have handled this, but we never crash)
        slot = f"neutral{neutral_counter}"
        if neutral_counter < 2:
            neutral_counter += 1
        speaker_voice_map[speaker] = slot
        return slot

    validated = []
    for i, seg in enumerate(script):
        seg = _local_validate(seg, i)   # re-run local validation
        if seg is None:
            continue

        # Resolve voice slot
        existing_voice = seg.get("voice") or None
        voice_slot = _assign_voice(seg["speaker"], existing_voice)
        seg["voice"] = voice_slot

        # Inject the actual TTS voice name for Step 4
        seg["tts_voice"] = VOICE_MAP.get(voice_slot, "alloy")

        validated.append(seg)

    return validated


# ── File I/O ──────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )


def _save_script(script: list, stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{stem}.json"
    out.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean and validate a raw script JSON for TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/script_parser/json_cleaner.py -i data/scripts/story_raw.json\n"
            "  python src/script_parser/json_cleaner.py -i story_raw.json --skip-llm\n"
            "  python src/script_parser/json_cleaner.py -i story_raw.json --no-save -v\n"
        ),
    )
    p.add_argument("--input",    "-i", required=True,       help="Path to raw script JSON file.")
    p.add_argument("--no-save",        action="store_true", help="Skip saving output JSON.")
    p.add_argument("--skip-llm",       action="store_true", help="Local validation only, no API call.")
    p.add_argument("--verbose",  "-v", action="store_true", help="Enable debug logging.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    json_path = Path(args.input)
    if not json_path.exists():
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    try:
        raw_script = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input JSON: {e}")
        sys.exit(1)

    try:
        cleaned = clean_script(
            raw_script,
            save=not args.no_save,
            stem=json_path.stem.replace("_raw", ""),
            skip_llm=args.skip_llm,
        )
    except (EnvironmentError, RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Print preview table
    print(f"\n── Cleaned script preview (first 6 segments) ─────────")
    for seg in cleaned[:6]:
        tag  = f"[{seg['type'].upper():<10}]"
        spk  = f"{seg['speaker']:<12}"
        emo  = f"({seg['emotion']:<14})"
        voice = f"voice={seg.get('tts_voice','?'):<8}"
        txt  = seg['text'][:50] + ("..." if len(seg['text']) > 50 else "")
        print(f"  {tag} {spk} {emo} {voice}  {txt}")
    print(f"───────────────────────────────────────────────────────")
    print(f"  Total segments: {len(cleaned)}\n")

    # Print voice cast
    seen: dict[str, str] = {}
    for seg in cleaned:
        sp = seg["speaker"]
        if sp not in seen:
            seen[sp] = f"{seg.get('voice','?')}  →  {seg.get('tts_voice','?')}"
    print("── Voice cast ─────────────────────────────────────────")
    for sp, v in seen.items():
        print(f"  {sp:<16}  {v}")
    print()


if __name__ == "__main__":
    main()