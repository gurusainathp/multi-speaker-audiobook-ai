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

Schema version: 2.0
    Added: id, style, intensity, pause_after, sound
    Added types: sfx, ambience
"""

import uuid


def _new_id() -> str:
    """Generate a short unique segment ID."""
    return uuid.uuid4().hex[:8]


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dialogue / Speaker / Emotion Detection
# Model: gpt-4o-mini
# ════════════════════════════════════════════════════════════════════════════

DIALOGUE_SYSTEM_PROMPT = """\
You are a script analyst for a cinematic audiobook production system.
Your job is to convert raw story text into a structured JSON script.
You must return ONLY a valid JSON array — no explanation, no markdown fences, no preamble.
"""

DIALOGUE_USER_PROMPT = """\
Convert the following story text into a structured JSON script for cinematic audiobook narration.

Rules:
1. Split the text into individual segments — one segment per spoken line or per narration beat.
2. For each segment, identify:
   - "speaker": The character speaking. Use "Narrator" for all narration and description.
   - "type": One of: "narration", "dialogue"
   - "emotion": The emotional tone. Choose from:
       neutral, happy, sad, angry, fearful, surprised, disgusted,
       excited, melancholic, contemplative, tense, gentle, bitter, hopeful
   - "text": The exact text of the segment. Do not paraphrase or alter.
3. Keep narration segments short (1–3 sentences). Split long narration blocks.
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
# STEP 3 — JSON Cleanup, Schema Upgrade, Voice + Acting Assignment
# Model: gpt-4o  (upgraded from gpt-4.1-nano — richer schema needs it)
# ════════════════════════════════════════════════════════════════════════════

CLEANER_SYSTEM_PROMPT = """\
You are a senior audiobook script editor and sound designer for a cinematic production system.
You receive a raw JSON script array and return a fully enriched, production-ready version.
You must return ONLY a valid JSON array — no explanation, no markdown fences, no preamble.
"""

CLEANER_USER_PROMPT = """\
You are given a raw JSON script array for a cinematic audiobook. Your job is to:
  1. Fix all schema errors
  2. Enrich every segment with acting direction, pacing, and intensity
  3. Insert sound effect and ambience segments where they would enhance the scene
  4. Assign voices consistently per character

────────────────────────────────────────────────────────────────
FULL SCHEMA — every output segment must use this exact structure:
────────────────────────────────────────────────────────────────
{
  "id":          "<8-char unique hex string>",
  "type":        "narration | dialogue | sfx | ambience",
  "speaker":     "<character name or Narrator>",
  "emotion":     "<emotion>",
  "style":       "<acting direction>",
  "intensity":   <0.1 – 1.0>,
  "pause_after": <milliseconds>,
  "sound":       "<sfx/ambience sound name or empty string>",
  "text":        "<exact text or empty string for sfx/ambience>",
  "voice":       "<voice slot>",
  "tts_voice":   "<openai tts voice name>"
}

────────────────────────────────────────────────────────────────
FIELD RULES:
────────────────────────────────────────────────────────────────
type:
  - "narration"  → spoken by Narrator
  - "dialogue"   → spoken by a character
  - "sfx"        → a sound effect event (door creak, thunder, footsteps)
  - "ambience"   → a background sound layer (wind, rain, crowd noise)

text:
  - Required for narration and dialogue. Must be exact original text.
  - Must be empty string "" for sfx and ambience.

sound:
  - Required for sfx and ambience. Use a short snake_case name.
  - Examples: "door_creak", "thunder_rumble", "rain_soft", "fire_crackling",
    "footsteps_stone", "glass_breaking", "night_wind", "crowd_murmur"
  - Must be empty string "" for narration and dialogue.

speaker:
  - Use "Narrator" for narration, sfx, and ambience.
  - Never leave empty.

emotion:
  - Required for narration and dialogue. Choose from:
    neutral, happy, sad, angry, fearful, surprised, disgusted,
    excited, melancholic, contemplative, tense, gentle, bitter, hopeful
  - Use "neutral" for sfx and ambience.

style (acting direction for dialogue and narration):
  - Describe how the line should be delivered. Be specific.
  - Examples: "slow, whispered", "sharp and clipped", "breathless urgency",
    "dry and flat", "warm but tired", "rising panic", "cold certainty"
  - Use "" for sfx and ambience.

intensity:
  - Float between 0.1 (very quiet/soft) and 1.0 (full emotional peak).
  - Calibrate to the moment: a whispered confession = 0.3, a scream = 0.9.
  - Use 0.5 as default for neutral segments.

pause_after:
  - Silence in milliseconds to insert AFTER this segment before the next.
  - Required for all types. Suggested values:
      Dialogue back-and-forth:      250 – 400 ms
      Narration beat:               400 – 600 ms
      After emotional dialogue:     500 – 800 ms
      After sfx (before speech):    300 – 700 ms
      After ambience insert:        0 ms (ambience plays under, not before)
  - Use your judgement based on emotional weight.

voice / tts_voice (for narration and dialogue only):
  - Assign consistently: same speaker always gets the same voice.
  - Narrator → voice: "narrator", tts_voice: "onyx"
  - Male characters → voice: "male1"/"male2"/"male3", tts_voice: "echo"/"fable"/"onyx"
  - Female characters → voice: "female1"/"female2"/"female3", tts_voice: "nova"/"shimmer"/"alloy"
  - Unknown gender → voice: "neutral1", tts_voice: "alloy"
  - sfx/ambience → voice: "", tts_voice: ""

────────────────────────────────────────────────────────────────
SFX AND AMBIENCE INSERTION GUIDELINES:
────────────────────────────────────────────────────────────────
- Insert sfx segments BEFORE the action they accompany (e.g. door creak before "He entered")
- Insert ambience segments at the START of a new scene or location
- Do not over-insert: 1–2 sfx per scene is enough. Less is more.
- Only insert sfx/ambience that are strongly implied by the text.

────────────────────────────────────────────────────────────────
VALIDATION:
────────────────────────────────────────────────────────────────
- Never leave "id" empty — generate a unique 8-char hex string per segment.
- Never change any "text" field content.
- Every segment must have all 10 fields, even if value is "".
- Output ONLY the JSON array. No markdown. No explanation.

Input script:
<<SCRIPT>>
"""


# ════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFAULTS — used by local validation passes in json_cleaner.py
# ════════════════════════════════════════════════════════════════════════════

SCHEMA_DEFAULTS = {
    "id":          "",          # filled by post-validator if missing
    "type":        "narration",
    "speaker":     "Narrator",
    "emotion":     "neutral",
    "style":       "",
    "intensity":   0.5,
    "pause_after": 400,
    "sound":       "",
    "text":        "",
    "voice":       "",
    "tts_voice":   "",
}

# Types that require a "text" field
TEXT_TYPES = {"narration", "dialogue"}

# Types that require a "sound" field
SOUND_TYPES = {"sfx", "ambience"}

# All valid segment types
VALID_TYPES = {"narration", "dialogue", "sfx", "ambience"}

# All valid emotions
VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "excited", "melancholic", "contemplative",
    "tense", "gentle", "bitter", "hopeful",
}