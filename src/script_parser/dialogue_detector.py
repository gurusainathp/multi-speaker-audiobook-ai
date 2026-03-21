"""
src/script_parser/dialogue_detector.py
────────────────────────────────────────────────────────────────
Step 2 of the audiobook pipeline: clean text → structured script.

Sends story text to gpt-4o-mini and receives a JSON list of
script segments, each with speaker, emotion, type, and text.

Usage (import):
    from src.script_parser.dialogue_detector import detect_dialogue

    with open("data/extracted_text/story.txt") as f:
        text = f.read()

    script = detect_dialogue(text)   # List[dict]

Usage (CLI):
    python src/script_parser/dialogue_detector.py \\
        --input data/extracted_text/story.txt
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

from src.script_parser.prompts import DIALOGUE_SYSTEM_PROMPT, DIALOGUE_USER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL          = "gpt-4o-mini"
MAX_TOKENS     = 4096          # Response budget — enough for ~2 pages of script
TEMPERATURE    = 0.3           # Low temp = consistent structure, less hallucination
CHUNK_SIZE     = 3000          # Characters per chunk when splitting long texts
OUTPUT_DIR     = Path("data/scripts")


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def detect_dialogue(text: str, save: bool = True, stem: str = "story") -> list:
    """
    Convert clean story text into a structured JSON script.

    Sends the text to gpt-4o-mini with a structured prompt.
    For long texts, splits into chunks and merges the results.

    Args:
        text: Clean plain-text story string (output of load_text).
        save: If True, saves the raw script JSON to data/scripts/.
        stem: Output filename stem (e.g. "story" → "story_raw.json").

    Returns:
        List of dicts, each with keys: speaker, type, emotion, text.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        RuntimeError:     If the API call fails or returns unparseable JSON.
    """
    _check_api_key()

    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    logger.info(f"[detect_dialogue] Input length: {len(text):,} chars")

    # Split into chunks if text is long
    chunks = _split_text(text, CHUNK_SIZE)
    logger.info(f"[detect_dialogue] Processing {len(chunks)} chunk(s) via {MODEL}")

    all_segments: list = []

    for i, chunk in enumerate(chunks, start=1):
        logger.info(f"[detect_dialogue] Chunk {i}/{len(chunks)} ({len(chunk):,} chars)...")
        segments = _call_openai(chunk)
        all_segments.extend(segments)

    logger.info(f"[detect_dialogue] Total segments: {len(all_segments)}")

    if save:
        out = _save_script(all_segments, stem=f"{stem}_raw")
        logger.info(f"[detect_dialogue] Saved raw script to: {out}")

    return all_segments


# ════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _check_api_key() -> None:
    """Raise a clear error if OPENAI_API_KEY is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )


def _split_text(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks of roughly chunk_size characters,
    breaking only at paragraph boundaries (double newlines).

    This prevents a character's speech from being cut mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n separator
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _call_openai(text_chunk: str) -> list:
    """
    Send one chunk to gpt-4o-mini and return parsed list of segments.

    Uses the new openai >= 1.0 client interface.
    Retries once on JSON parse failure with a stricter follow-up prompt.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required. Run: pip install openai"
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_message = DIALOGUE_USER_PROMPT.replace("<<TEXT>>", text_chunk)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    raw_content = response.choices[0].message.content.strip()

    logger.debug(f"[_call_openai] Raw response ({len(raw_content)} chars):\n{raw_content[:300]}...")

    return _parse_json_response(raw_content)


def _parse_json_response(content: str) -> list:
    """
    Parse the model's response into a Python list.

    Handles common model quirks:
    - Response wrapped in ```json ... ``` fences
    - Leading/trailing whitespace
    - Model returning a dict instead of a list (wraps it)
    """
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Drop first line (```json or ```) and last line (```)
        content = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[_parse_json_response] JSON parse failed: {e}")
        logger.error(f"[_parse_json_response] Raw content:\n{content[:500]}")
        raise RuntimeError(
            f"Failed to parse model response as JSON. "
            f"Parse error: {e}\n"
            f"Response preview: {content[:200]}"
        )

    # Model occasionally returns a dict instead of a list — wrap it
    if isinstance(parsed, dict):
        if "segments" in parsed:
            parsed = parsed["segments"]
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Expected a JSON array, got: {type(parsed).__name__}"
        )

    # Validate and normalise each segment
    validated = []
    for i, seg in enumerate(parsed):
        seg = _validate_segment(seg, index=i)
        if seg:
            validated.append(seg)

    return validated


def _validate_segment(seg: dict, index: int) -> dict | None:
    """
    Ensure a segment has all required fields with valid values.
    Fills in safe defaults rather than crashing on minor model slippage.
    Returns None if the segment is unrecoverable (e.g. empty text).
    """
    VALID_TYPES    = {"narration", "dialogue"}
    VALID_EMOTIONS = {
        "neutral", "happy", "sad", "angry", "fearful", "surprised",
        "disgusted", "excited", "melancholic", "contemplative",
        "tense", "gentle", "bitter", "hopeful",
    }

    if not isinstance(seg, dict):
        logger.warning(f"[_validate_segment] Segment {index} is not a dict — skipping.")
        return None

    text = str(seg.get("text", "")).strip()
    if not text:
        logger.warning(f"[_validate_segment] Segment {index} has empty text — skipping.")
        return None

    speaker = str(seg.get("speaker", "Narrator")).strip() or "Narrator"

    seg_type = str(seg.get("type", "narration")).lower()
    if seg_type not in VALID_TYPES:
        seg_type = "dialogue" if speaker != "Narrator" else "narration"

    emotion = str(seg.get("emotion", "neutral")).lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    return {
        "speaker": speaker,
        "type":    seg_type,
        "emotion": emotion,
        "text":    text,
    }


def _save_script(script: list, stem: str) -> Path:
    """Save script JSON to data/scripts/<stem>.json."""
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
        description="Detect dialogue, speakers, and emotions from extracted story text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/script_parser/dialogue_detector.py -i data/extracted_text/story.txt\n"
            "  python src/script_parser/dialogue_detector.py -i story.txt --no-save -v\n"
        ),
    )
    p.add_argument("--input",   "-i", required=True,       help="Path to extracted .txt file.")
    p.add_argument("--no-save",       action="store_true", help="Skip saving output JSON.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    txt_path = Path(args.input)
    if not txt_path.exists():
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    text = txt_path.read_text(encoding="utf-8")

    try:
        script = detect_dialogue(text, save=not args.no_save, stem=txt_path.stem)
    except (EnvironmentError, RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Print a preview table
    print(f"\n── Script preview (first 6 segments) ─────────────────")
    for seg in script[:6]:
        tag = f"[{seg['type'].upper():<10}]"
        spk = f"{seg['speaker']:<12}"
        emo = f"({seg['emotion']:<14})"
        txt = seg['text'][:60] + ("..." if len(seg['text']) > 60 else "")
        print(f"  {tag} {spk} {emo}  {txt}")
    print(f"──────────────────────────────────────────────────────")
    print(f"  Total segments: {len(script)}\n")


if __name__ == "__main__":
    main()