"""
src/io/text_loader.py
────────────────────────────────────────────────────────────────
Universal file loader for the audiobook pipeline.

This is the ONLY entry point the rest of the pipeline should use.
Never call individual parsers directly — always call load_text().

Supported formats:
    .pdf    — pdfplumber
    .txt    — plain read with encoding detection (chardet)
    .docx   — python-docx
    .epub   — ebooklib + beautifulsoup4  [optional]
    .html   — beautifulsoup4             [optional]
    .md     — plain read (markdown is plain text)
    .rtf    — striprtf                   [optional]

Usage (import from main.py or pipeline):
    from src.io.text_loader import load_text
    text = load_text("data/input_pdfs/story.pdf")

Usage (CLI, run directly):
    python src/io/text_loader.py --input data/input_pdfs/story.pdf
    python src/io/text_loader.py -i story.docx --no-save --verbose
"""

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensures `from src.utils...` works whether this file is:
#   - run directly:  python src/io/text_loader.py
#   - imported from: main.py at the project root
# Must be at the very top, before any src.* imports.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging

from src.utils.text_cleaner import clean_text, clean_page

logger = logging.getLogger(__name__)

# ── Output directory ─────────────────────────────────────────────────────────

OUTPUT_DIR = Path("data/extracted_text")

# ── Supported extensions ─────────────────────────────────────────────────────

SUPPORTED = {".pdf", ".txt", ".docx", ".epub", ".html", ".htm", ".md", ".rtf"}


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def load_text(file_path: str, save: bool = True) -> str:
    """
    Load and return clean text from any supported file format.

    This is the single entry point for all file loading in the pipeline.
    Internally routes to the correct private parser based on file extension,
    then runs the result through text_cleaner before returning.

    Args:
        file_path: Path to the input file.
        save:      If True, saves extracted text to data/extracted_text/.

    Returns:
        Clean plain-text string of the full document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the extension is unsupported or extraction fails.
        ImportError:       If a required optional library is not installed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED))}"
        )

    logger.info(f"[load_text] Loading '{path.name}'  (format: {ext})")

    # ── Route to correct parser ──────────────────────────────────────────────
    if ext == ".pdf":
        raw = _load_pdf(path)
    elif ext == ".txt":
        raw = _load_txt(path)
    elif ext == ".docx":
        raw = _load_docx(path)
    elif ext == ".epub":
        raw = _load_epub(path)
    elif ext in (".html", ".htm"):
        raw = _load_html(path)
    elif ext == ".md":
        raw = _load_md(path)
    elif ext == ".rtf":
        raw = _load_rtf(path)
    else:
        raise ValueError(f"No parser implemented for: {ext}")

    if not raw or not raw.strip():
        raise ValueError(
            f"No text could be extracted from: {path.name}. "
            "The file may be empty, encrypted, or image-only."
        )

    # ── Clean ────────────────────────────────────────────────────────────────
    text = clean_text(raw)

    logger.info(
        f"[load_text] Extracted {len(text):,} characters from '{path.name}'"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    if save:
        out = _save_text(text, stem=path.stem)
        logger.info(f"[load_text] Saved to: {out}")

    return text


# ════════════════════════════════════════════════════════════════════════════
# PRIVATE PARSERS
# ════════════════════════════════════════════════════════════════════════════

def _load_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.
    Applies light per-page cleaning before joining pages.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF loading. "
            "Run: pip install pdfplumber"
        )

    pages_text = []

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        logger.info(f"[_load_pdf] {total} page(s) found.")

        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text()
            if raw:
                cleaned = clean_page(raw)
                if cleaned:
                    pages_text.append(cleaned)
            else:
                logger.warning(
                    f"[_load_pdf] Page {i}/{total} yielded no text "
                    "(image-only or blank)."
                )

    if not pages_text:
        raise ValueError(
            "No text extracted from PDF. "
            "It may be a scanned document — consider OCR."
        )

    return "\n\n".join(pages_text)


def _load_txt(path: Path) -> str:
    """
    Load a plain-text file.
    Tries UTF-8 first; falls back to chardet encoding detection
    so Windows-1252 / Latin-1 files don't crash.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass

    try:
        import chardet
    except ImportError:
        raise ImportError(
            "chardet is required to load non-UTF-8 text files. "
            "Run: pip install chardet"
        )

    raw_bytes = path.read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding") or "latin-1"
    confidence = detected.get("confidence", 0)

    logger.info(
        f"[_load_txt] Detected encoding: {encoding} "
        f"(confidence: {confidence:.0%})"
    )

    return raw_bytes.decode(encoding, errors="replace")


def _load_docx(path: Path) -> str:
    """
    Extract text from a .docx file using python-docx.
    Preserves paragraph boundaries.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for .docx loading. "
            "Run: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

    if not paragraphs:
        raise ValueError(f"No text found in docx file: {path.name}")

    return "\n\n".join(paragraphs)


def _load_epub(path: Path) -> str:
    """
    Extract text from an .epub file using ebooklib + BeautifulSoup.
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "ebooklib and beautifulsoup4 are required for .epub loading. "
            "Run: pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(str(path))
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "head", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        if text:
            chapters.append(text)

    if not chapters:
        raise ValueError(f"No text found in epub file: {path.name}")

    return "\n\n".join(chapters)


def _load_html(path: Path) -> str:
    """
    Extract readable text from an HTML file using BeautifulSoup.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for .html loading. "
            "Run: pip install beautifulsoup4 lxml"
        )

    raw = _load_txt(path)
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style", "head", "nav", "footer", "header"]):
        tag.decompose()

    return soup.get_text(separator="\n").strip()


def _load_md(path: Path) -> str:
    """Load Markdown as plain text — it's already human-readable."""
    return _load_txt(path)


def _load_rtf(path: Path) -> str:
    """Strip RTF markup using striprtf."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise ImportError(
            "striprtf is required for .rtf loading. "
            "Run: pip install striprtf"
        )
    return rtf_to_text(_load_txt(path))


# ── File I/O helper ───────────────────────────────────────────────────────────

def _save_text(text: str, stem: str) -> Path:
    """Save to data/extracted_text/<stem>.txt, creating dirs as needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{stem}.txt"
    out.write_text(text, encoding="utf-8")
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s  %(message)s", level=level)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load and extract text from PDF, TXT, DOCX, EPUB, HTML, MD, or RTF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/io/text_loader.py -i data/input_pdfs/story.pdf\n"
            "  python src/io/text_loader.py -i story.docx --no-save\n"
            "  python src/io/text_loader.py -i book.epub -v\n"
        ),
    )
    p.add_argument("--input",   "-i", required=True,       help="Path to the input file.")
    p.add_argument("--no-save",       action="store_true", help="Skip saving output .txt file.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    _setup_logging(args.verbose)

    try:
        text = load_text(args.input, save=not args.no_save)
        preview = " ".join(text[:400].split())
        print(f"\n── Preview (first 400 chars) ──────────────────────────")
        print(preview)
        print(f"───────────────────────────────────────────────────────")
        print(f"Total characters: {len(text):,}\n")
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()