"""
src/script_parser/prompts.py
────────────────────────────────────────────────────────────────
Central prompt definitions for the script parser.

All LLM prompts are defined here as module-level constants.
No prompt strings should live inside detector or cleaner modules.

IMPORTANT — substitution convention:
    Prompts use <<PLACEHOLDER>> tokens instead of {placeholder}.
    This avoids Python's str.format() choking on the JSON example
    blocks inside the prompt (which also contain curly braces).

    In dialogue_detector.py:
        prompt = DIALOGUE_USER_PROMPT.replace("<<TEXT>>", text_chunk)

    In json_cleaner.py:
        prompt = CLEANER_USER_PROMPT.replace("<<SCRIPT>>", script_str)
"""

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dialogue / Speaker / Emotion Detection
# Used by: dialogue_detector.py  →  gpt-4o-mini
# ════════════════════════════════════════════════════════════════════════════

DIALOGUE_SYSTEM_PROMPT = """\
You are a script analyst for an audiobook production system.
Your job is to convert raw story text into a structured JSON script.
You must return ONLY a valid JSON array — no explanation, no markdown fences, no preamble.
"""

DIALOGUE_USER_PROMPT = """\
Convert the following story text into a structured JSON script for audiobook narration.

Rules:
1. Split the text into individual segments — one segment per spoken line or per narration beat.
2. For each segment, identify:
   - "speaker": The character speaking. Use "Narrator" for narration or description.
   - "type": Either "narration" or "dialogue".
   - "emotion": The emotional tone. Choose from:
       neutral, happy, sad, angry, fearful, surprised, disgusted,
       excited, melancholic, contemplative, tense, gentle, bitter, hopeful
   - "text": The exact text of the segment. Do not paraphrase.
3. Keep narration segments short (1-3 sentences). Split long narration blocks.
4. Every piece of the original text must appear exactly once across all segments.
5. Do not add, remove, or rephrase any text.
6. Output ONLY a valid JSON array. No markdown. No explanation. No code fences.

Example output format:
[
  {"speaker": "Narrator", "type": "narration", "emotion": "neutral", "text": "The sun had already set."},
  {"speaker": "Elena", "type": "dialogue", "emotion": "sad", "text": "I don't think I can do this anymore."},
  {"speaker": "Narrator", "type": "narration", "emotion": "tense", "text": "Marcus looked away."},
  {"speaker": "Marcus", "type": "dialogue", "emotion": "gentle", "text": "You don't have to. Not alone."}
]

Story text:
<<TEXT>>
"""


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — JSON Cleanup and Validation
# Used by: json_cleaner.py  →  gpt-4.1-nano
# ════════════════════════════════════════════════════════════════════════════

CLEANER_SYSTEM_PROMPT = """\
You are a JSON validator and formatter for an audiobook pipeline.
You receive a JSON array of script segments and return a corrected, complete version.
You must return ONLY valid JSON — no explanation, no markdown, no code fences.
"""

CLEANER_USER_PROMPT = """\
You are given a JSON script array for an audiobook. Fix any issues and return the corrected array.

Validation rules:
1. Every object must have exactly these four fields: "speaker", "type", "emotion", "text".
2. "type" must be either "narration" or "dialogue". Fix any other values.
3. "emotion" must be one of:
   neutral, happy, sad, angry, fearful, surprised, disgusted,
   excited, melancholic, contemplative, tense, gentle, bitter, hopeful
   If missing or invalid, infer from context.
4. "speaker" must never be empty. Use "Narrator" if unknown or missing.
5. "text" must never be empty. Remove the segment if text is empty.
6. Add a "voice" field to each segment based on these rules:
   - "Narrator"        -> "voice": "narrator"
   - Male characters   -> "voice": "male1", "male2", etc. (assign consistently per character)
   - Female characters -> "voice": "female1", "female2", etc.
   - Unknown gender    -> "voice": "neutral1"
7. Keep character-to-voice assignments consistent throughout the entire array.
8. Do not change any "text" field content.
9. Output ONLY a valid JSON array. No markdown. No explanation.

Input script:
<<SCRIPT>>
"""