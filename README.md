# 🎙️ Multi-Speaker Emotional Audiobook Generator

> Transform any PDF story into a fully voiced, multi-speaker audiobook — complete with distinct character voices and emotion-aware narration.

---

## 📖 Overview

This project is an end-to-end AI pipeline that takes a PDF as input and produces a finished audiobook as output. It automatically detects narration, dialogue, speaker identity, and emotional tone — then synthesizes realistic audio for each segment using OpenAI's TTS API.

No model training. No heavy local models. Just a clean, modular pipeline built on top of well-maintained APIs and libraries.

---

## 🔁 Pipeline

```
PDF
 ↓  pdfplumber / pymupdf       — extract raw text (free, local)
 ↓  gpt-4o-mini                — detect dialogue, speakers, emotions → structured JSON
 ↓  gpt-4.1-nano               — clean and validate JSON, assign voice types
 ↓  gpt-4o-mini-tts            — generate audio per segment with emotion
 ↓  pydub / ffmpeg             — merge segments, add pauses, export
 ↓
Final Audiobook (.mp3)
```

---

## ✨ Features

- **PDF Parsing** — extracts clean text from any story PDF
- **Dialogue Detection** — separates narration from character speech
- **Speaker Identification** — assigns lines to named characters
- **Emotion Detection** — tags each segment (happy, angry, sad, neutral, etc.)
- **Multi-Voice TTS** — different voice profiles per character and narrator
- **Audio Merging** — combines all segments with natural pacing and pauses
- **Configurable Voices** — voice assignments editable via `configs/voices.json`

---

## 🗂️ Project Structure

```
pdf-emotional-audiobook-generator/
├── README.md
├── requirements.txt
├── .env                        # API keys (never commit this)
├── .gitignore
│
├── data/
│   ├── input_pdfs/             # Drop your story PDFs here
│   ├── extracted_text/         # Raw text output from PDF parser
│   ├── scripts/                # Structured JSON scripts
│   ├── audio_segments/         # Individual TTS audio files
│   └── final_audio/            # Finished audiobook exports
│
├── src/
│   ├── pdf_parser/
│   │   └── parse_pdf.py        # PDF → plain text
│   ├── script_parser/
│   │   ├── dialogue_detector.py  # text → structured JSON (gpt-4o-mini)
│   │   └── json_cleaner.py       # JSON validation pass (gpt-4.1-nano)
│   ├── tts/
│   │   └── tts_generator.py    # JSON segments → audio files (OpenAI TTS)
│   ├── audio/
│   │   └── merge_audio.py      # Merge segments → final audiobook
│   ├── utils/
│   │   └── file_utils.py       # Shared helpers
│   └── main.py                 # Orchestrates the full pipeline
│
├── tests/
│   └── test_pipeline.py
├── configs/
│   └── voices.json             # Voice assignments per character/emotion
└── notebooks/
    └── experiments.ipynb       # Scratchpad for testing steps in isolation
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pdf-emotional-audiobook-generator.git
cd pdf-emotional-audiobook-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You will also need **ffmpeg** installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 3. Set up your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

A single OpenAI API key covers all three models used in this pipeline (`gpt-4o-mini`, `gpt-4.1-nano`, and `gpt-4o-mini-tts`).

---

## 🚀 Usage

### Run the full pipeline

```bash
python src/main.py --input data/input_pdfs/story.pdf
```

### Run individual steps

```bash
# Step 1: Parse PDF only
python src/pdf_parser/parse_pdf.py --input data/input_pdfs/story.pdf

# Step 2: Detect dialogue and structure
python src/script_parser/dialogue_detector.py --input data/extracted_text/story.txt

# Step 3: Clean and validate JSON
python src/script_parser/json_cleaner.py --input data/scripts/story_raw.json

# Step 4: Generate audio segments
python src/tts/tts_generator.py --input data/scripts/story_clean.json

# Step 5: Merge into final audiobook
python src/audio/merge_audio.py --input data/audio_segments/ --output data/final_audio/audiobook.mp3
```

---

## 🧩 Structured Script Format

After Step 2, each line of the story is represented as a JSON object:

```json
[
  {
    "speaker": "Narrator",
    "emotion": "neutral",
    "voice": "narrator",
    "text": "John walked into the room.",
    "type": "narration"
  },
  {
    "speaker": "John",
    "emotion": "happy",
    "voice": "male1",
    "text": "Hello Mary",
    "type": "dialogue"
  },
  {
    "speaker": "Mary",
    "emotion": "angry",
    "voice": "female1",
    "text": "Why are you here?",
    "type": "dialogue"
  }
]
```

Voice assignments are loaded from `configs/voices.json` and can be customized per project.

---

## 💰 Estimated API Costs

| Content Length | Estimated Cost |
|----------------|---------------|
| Short test (~10 lines) | ~$0.02 |
| Short story (~5 min audio) | ~$0.10 |
| Medium story (~30 min audio) | ~$0.50 |
| Full novel | ~$4–6 |

The bulk of cost comes from TTS generation. LLM parsing steps are negligible (< $0.002 per run). You can run many parsing iterations for pennies while fine-tuning prompts before committing to TTS.

---

## 🔧 Configuration

Edit `configs/voices.json` to customize voice assignments:

```json
{
  "Narrator": "onyx",
  "default_male": "echo",
  "default_female": "nova",
  "characters": {
    "John": "echo",
    "Mary": "shimmer"
  }
}
```

Available OpenAI TTS voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

---

## 🗺️ Roadmap / Optional Features

- [ ] Background music/ambience mixing
- [ ] Sound effects on key events
- [ ] Web UI (Streamlit or React/FastAPI)
- [ ] Per-character speed and pitch control
- [ ] Custom emotion intensity levels
- [ ] Multi-language support
- [ ] Save/load project state
- [ ] Batch processing for multiple PDFs

---

## 🤝 Team

| Role | Responsibility |
|------|---------------|
| Person 1 | PDF parsing and text cleaning |
| Person 2 | OpenAI parsing and JSON structuring |
| Person 3 | TTS generation and audio output |
| Person 4 | Audio merging and UI |

---

## 📋 Requirements

See `requirements.txt` for the full list. Core dependencies:

- `openai` — LLM + TTS API calls
- `pdfplumber` / `pymupdf` — PDF text extraction
- `pydub` — audio segment manipulation
- `python-dotenv` — environment variable management
- `ffmpeg` — audio encoding (system dependency)

---

## ⚠️ Notes

- Never commit your `.env` file or any API keys
- The `data/` directory is excluded from version control by default (see `.gitignore`)
- This project uses APIs only — no local model training or GPU required
- Pipeline steps are designed to be run independently for easier debugging and development

---

## 📄 License

MIT License — feel free to fork, extend, and build on this.