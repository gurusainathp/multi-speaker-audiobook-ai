"""
src/script_parser/json_cleaner.py
────────────────────────────────────────────────────────────────
Step 3 of the audiobook pipeline: raw script JSON → enriched, TTS-safe script.

Schema v2 — each segment now carries:
    id, type, speaker, emotion, style, intensity,
    pause_after, sound, text, voice, tts_voice

New segment types: sfx, ambience (in addition to narration, dialogue)

Three-pass design:
  Pass 1 (local)  — cheap pre-validation before the API call
  Pass 2 (LLM)   — gpt-4o enriches with style, intensity, pause_after,
                    inserts sfx/ambience, assigns voices
  Pass 3 (local)  — post-validation guarantees schema compliance
                    regardless of model output

Usage (import):
    from src.script_parser.json_cleaner import clean_script

    raw   = detect_dialogue(text)
    clean = clean_script(raw)   # list[dict] — full schema v2

Usage (CLI):
    python src/script_parser/json_cleaner.py -i data/scripts/story_raw.json
    python src/script_parser/json_cleaner.py -i story_raw.json --skip-llm
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
import uuid

from dotenv import load_dotenv

from src.script_parser.prompts import (
    CLEANER_SYSTEM_PROMPT,
    CLEANER_USER_PROMPT,
    SCHEMA_DEFAULTS,
    VALID_TYPES,
    VALID_EMOTIONS,
    TEXT_TYPES,
    SOUND_TYPES,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL       = "gpt-4.1"          # same capability tier as gpt-4o, ~20% cheaper
                                  # ($2.00/$8.00 vs $2.50/$10.00 per 1M tokens)
                                  # DO NOT drop to mini — style/sfx placement
                                  # requires narrative judgment that mini loses
MAX_TOKENS  = 8192              # schema v2 segments are larger
TEMPERATURE = 0.2               # slight creativity for style/sfx, still structured
OUTPUT_DIR  = Path("data/scripts")

# OpenAI TTS voice names — mapped from logical voice slots
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
    Validate, repair, and enrich a raw script list to schema v2.

    Args:
        raw_script: Output of detect_dialogue() — list of segment dicts.
        save:       If True, saves cleaned script to data/scripts/.
        stem:       Filename stem (e.g. "story" → "story_clean.json").
        skip_llm:   If True, run local validation only (no API call).

    Returns:
        Cleaned list of dicts. Each dict is guaranteed to have all 10
        schema v2 fields: id, type, speaker, emotion, style, intensity,
        pause_after, sound, text, voice, tts_voice

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set (and skip_llm=False).
        RuntimeError:     If the API call fails.
    """
    if not raw_script:
        raise ValueError("raw_script is empty — nothing to clean.")

    logger.info(f"[clean_script] Input: {len(raw_script)} segment(s)")

    # ── Pass 1: local pre-validation ─────────────────────────────────────────
    pre_validated = [
        seg for seg in
        (_pre_validate(s, i) for i, s in enumerate(raw_script))
        if seg is not None
    ]
    logger.info(f"[clean_script] After pre-validation: {len(pre_validated)} segment(s)")

    # ── Pass 2: LLM enrichment ────────────────────────────────────────────────
    if skip_llm:
        logger.info("[clean_script] Skipping LLM pass (skip_llm=True).")
        llm_result = pre_validated
    else:
        _check_api_key()
        logger.info(f"[clean_script] Sending to {MODEL} for enrichment...")
        llm_result = _call_openai(pre_validated)
        logger.info(f"[clean_script] After LLM pass: {len(llm_result)} segment(s)")

    # ── Pass 3: post-validation + voice injection ─────────────────────────────
    final = _post_validate_all(llm_result)
    logger.info(f"[clean_script] Final segment count: {len(final)}")

    if save:
        out = _save_script(final, stem=f"{stem}_clean")
        logger.info(f"[clean_script] Saved to: {out}")

    return final


# ════════════════════════════════════════════════════════════════════════════
# PASS 1 — Local pre-validation (before LLM call)
# ════════════════════════════════════════════════════════════════════════════

def _pre_validate(seg: dict, index: int) -> dict | None:
    """
    Cheap local check before sending to the LLM.
    Applies schema defaults, fixes obvious type/emotion errors.
    Does NOT assign voice, style, intensity — that's the LLM's job.
    """
    if not isinstance(seg, dict):
        logger.warning(f"[pre-validate] Segment {index}: not a dict — dropped.")
        return None

    seg_type = str(seg.get("type", "narration")).lower()
    if seg_type not in VALID_TYPES:
        seg_type = "narration"

    # Validate type-specific required fields
    if seg_type in TEXT_TYPES:
        text = str(seg.get("text", "")).strip()
        if not text:
            logger.warning(f"[pre-validate] Segment {index}: {seg_type} has empty text — dropped.")
            return None
    else:
        text = ""

    if seg_type in SOUND_TYPES:
        sound = str(seg.get("sound", "")).strip()
        # Don't drop sfx/ambience without sound — LLM will fill it in
    else:
        sound = ""

    speaker = str(seg.get("speaker", "Narrator")).strip() or "Narrator"

    emotion = str(seg.get("emotion", "neutral")).lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    # Pass through any enrichment fields that may already be set
    result = {
        "id":          str(seg.get("id", "")).strip() or "",
        "type":        seg_type,
        "speaker":     speaker,
        "emotion":     emotion,
        "style":       str(seg.get("style", "")).strip(),
        "intensity":   _clamp_intensity(seg.get("intensity", 0.5)),
        "pause_after": _clamp_pause(seg.get("pause_after", SCHEMA_DEFAULTS["pause_after"])),
        "sound":       str(seg.get("sound", sound)).strip(),
        "text":        text,
        "voice":       str(seg.get("voice", "")).strip(),
        "tts_voice":   str(seg.get("tts_voice", "")).strip(),
    }
    return result


# ════════════════════════════════════════════════════════════════════════════
# PASS 2 — LLM enrichment
# ════════════════════════════════════════════════════════════════════════════

def _call_openai(script: list) -> list:
    """Send the pre-validated script to gpt-4o for enrichment."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Run: pip install openai")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    script_str   = json.dumps(script, indent=2, ensure_ascii=False)
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
    logger.debug(f"[_call_openai] Response preview:\n{raw_content[:400]}")
    return _parse_json_response(raw_content)


def _parse_json_response(content: str) -> list:
    """Parse model response — strips fences, handles dict wrapping."""
    content = content.strip()
    if content.startswith("```"):
        lines   = content.split("\n")
        content = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[_parse_json_response] JSON parse failed: {e}")
        logger.error(f"[_parse_json_response] Raw:\n{content[:800]}")
        raise RuntimeError(
            f"gpt-4o returned unparseable JSON.\nError: {e}\nPreview: {content[:300]}"
        )

    if isinstance(parsed, dict):
        parsed = parsed.get("segments", [parsed])
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected JSON array, got: {type(parsed).__name__}")

    return parsed


# ════════════════════════════════════════════════════════════════════════════
# PASS 3 — Post-validation + voice injection
# ════════════════════════════════════════════════════════════════════════════

def _post_validate_all(script: list) -> list:
    """
    Final local pass after the LLM.
    - Guarantees all 10 schema fields present on every segment
    - Generates missing IDs
    - Enforces voice consistency: same speaker → same voice
    - Injects tts_voice from VOICE_MAP
    - Clamps numeric fields to valid ranges
    - Enforces type-specific field rules (text ↔ sound)
    """
    speaker_voice_map: dict[str, str] = {"Narrator": "narrator"}
    male_counter    = 1
    female_counter  = 1
    neutral_counter = 1

    def _assign_voice(speaker: str, existing_voice: str) -> str:
        nonlocal male_counter, female_counter, neutral_counter

        if speaker in speaker_voice_map:
            return speaker_voice_map[speaker]

        if existing_voice and existing_voice in VOICE_MAP:
            speaker_voice_map[speaker] = existing_voice
            return existing_voice

        if existing_voice:
            v = existing_voice.lower()
            if v.startswith("male"):
                slot = f"male{min(male_counter, 3)}"
                male_counter += 1
                speaker_voice_map[speaker] = slot
                return slot
            if v.startswith("female"):
                slot = f"female{min(female_counter, 3)}"
                female_counter += 1
                speaker_voice_map[speaker] = slot
                return slot

        slot = f"neutral{min(neutral_counter, 2)}"
        neutral_counter += 1
        speaker_voice_map[speaker] = slot
        return slot

    validated = []
    for i, seg in enumerate(script):
        seg = _pre_validate(seg, i)
        if seg is None:
            continue

        seg_type = seg["type"]

        # ── Generate ID if missing ────────────────────────────────────────
        if not seg["id"]:
            seg["id"] = uuid.uuid4().hex[:8]

        # ── Enforce type-specific field rules ─────────────────────────────
        if seg_type in TEXT_TYPES:
            # narration/dialogue must have text, must NOT have sound
            if not seg["text"]:
                logger.warning(f"[post-validate] Seg {i} ({seg_type}): empty text — dropped.")
                continue
            seg["sound"] = ""

        elif seg_type in SOUND_TYPES:
            # sfx/ambience must have sound, text must be empty
            seg["text"]    = ""
            seg["speaker"] = "Narrator"
            seg["emotion"] = "neutral"
            seg["style"]   = ""
            if not seg["sound"]:
                logger.warning(f"[post-validate] Seg {i} ({seg_type}): no sound name — skipping.")
                continue

        # ── Voice assignment (only for speech segments) ───────────────────
        if seg_type in TEXT_TYPES:
            existing_voice = seg.get("voice") or ""
            voice_slot     = _assign_voice(seg["speaker"], existing_voice)
            seg["voice"]   = voice_slot
            seg["tts_voice"] = VOICE_MAP.get(voice_slot, "alloy")
        else:
            seg["voice"]     = ""
            seg["tts_voice"] = ""

        # ── Clamp numeric fields ──────────────────────────────────────────
        seg["intensity"]   = _clamp_intensity(seg.get("intensity", 0.5))
        seg["pause_after"] = _clamp_pause(seg.get("pause_after", SCHEMA_DEFAULTS["pause_after"]))

        # ── Ensure all 10 schema fields exist ────────────────────────────
        for field, default in SCHEMA_DEFAULTS.items():
            if field not in seg:
                seg[field] = default

        # ── Output in canonical field order ──────────────────────────────
        validated.append({
            "id":          seg["id"],
            "type":        seg["type"],
            "speaker":     seg["speaker"],
            "emotion":     seg["emotion"],
            "style":       seg["style"],
            "intensity":   seg["intensity"],
            "pause_after": seg["pause_after"],
            "sound":       seg["sound"],
            "text":        seg["text"],
            "voice":       seg["voice"],
            "tts_voice":   seg["tts_voice"],
        })

    return validated


# ── Numeric clamps ────────────────────────────────────────────────────────────

def _clamp_intensity(val) -> float:
    try:
        return round(max(0.1, min(1.0, float(val))), 2)
    except (TypeError, ValueError):
        return 0.5


def _clamp_pause(val) -> int:
    try:
        return max(0, min(5000, int(val)))
    except (TypeError, ValueError):
        return SCHEMA_DEFAULTS["pause_after"]


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
        description="Clean and enrich a raw script JSON to schema v2 (style, intensity, sfx, etc.).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/script_parser/json_cleaner.py -i data/scripts/story_raw.json\n"
            "  python src/script_parser/json_cleaner.py -i story_raw.json --skip-llm\n"
            "  python src/script_parser/json_cleaner.py -i story_raw.json --no-save -v\n"
        ),
    )
    p.add_argument("--input",    "-i", required=True,       help="Path to raw script JSON.")
    p.add_argument("--no-save",        action="store_true", help="Skip saving output JSON.")
    p.add_argument("--skip-llm",       action="store_true", help="Local validation only.")
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

    # Preview table
    print(f"\n── Cleaned script preview (first 8 segments) ─────────────────────")
    for seg in cleaned[:8]:
        type_tag = f"[{seg['type'].upper():<10}]"
        spk      = f"{seg['speaker']:<12}"
        emo      = f"({seg['emotion']:<14})"
        style    = f"style={seg['style'][:20]:<22}" if seg['style'] else f"{'':26}"
        intens   = f"i={seg['intensity']:.1f}"
        pause    = f"p={seg['pause_after']}ms"
        tts      = f"tts={seg.get('tts_voice','—'):<8}" if seg['type'] in ('narration','dialogue') else f"sfx={seg.get('sound','—'):<8}"
        content  = (seg['text'] or seg['sound'])[:45] + ("..." if len(seg.get('text','') or seg.get('sound','')) > 45 else "")
        print(f"  {type_tag} {spk} {emo}  {style} {intens}  {pause}  {tts}  {content}")

    print(f"──────────────────────────────────────────────────────────────────────")
    print(f"  Total segments: {len(cleaned)}")

    # Voice cast
    seen: dict[str, str] = {}
    for seg in cleaned:
        sp = seg["speaker"]
        if sp not in seen and seg["tts_voice"]:
            seen[sp] = f"{seg['voice']}  →  {seg['tts_voice']}"
    if seen:
        print("\n── Voice cast ──────────────────────────────────────────────────────")
        for sp, v in seen.items():
            print(f"  {sp:<16}  {v}")

    # SFX summary
    sfx_segs = [s for s in cleaned if s["type"] in ("sfx", "ambience")]
    if sfx_segs:
        print(f"\n── Sound effects / ambience ({len(sfx_segs)} segments) ──────────────────")
        for seg in sfx_segs:
            print(f"  [{seg['type'].upper():<10}]  {seg['sound']}")
    print()


if __name__ == "__main__":
    main()