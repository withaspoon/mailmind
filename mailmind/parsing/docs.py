from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class OCRConfig:
    langs: str = "eng"  # e.g., "eng+spa+swe"
    min_chars_per_page: int = 120  # OCR PDFs when below this per page
    max_pages_ocr: int = 200  # safety limit


def _pdftotext_available() -> bool:
    return shutil.which("pdftotext") is not None


def pdf_extract_text(path: Path) -> Tuple[str, int]:
    """Extract text from a PDF using pdfplumber or pdftotext.

    Returns (text, pages). If extraction fails, returns ("", 0).
    """
    # Try pdfplumber first
    try:
        import pdfplumber  # type: ignore

        text_parts = []
        pages = 0
        with pdfplumber.open(str(path)) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    text_parts.append(t)
        return ("\n\n".join(text_parts), pages)
    except Exception:
        pass

    # Fallback to pdftotext CLI
    if _pdftotext_available():
        try:
            out = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if out.returncode == 0:
                text = out.stdout.decode("utf-8", errors="ignore")
                # Page count unknown; best effort
                return (text, 0)
        except Exception:
            pass
    return ("", 0)


def run_ocrmypdf(in_pdf: Path, out_pdf: Path, langs: str) -> bool:
    """Run OCR on a PDF using ocrmypdf (Python API or CLI). Returns True on success."""
    # Try Python API
    try:
        import ocrmypdf  # type: ignore

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        ocrmypdf.ocr(
            str(in_pdf),
            str(out_pdf),
            language=langs.replace(",", "+"),
            force_ocr=True,
            optimize=1,
            progress_bar=False,
            deskew=True,
        )
        return out_pdf.exists()
    except Exception:
        pass

    # Fallback to CLI
    if shutil.which("ocrmypdf") is None:
        return False
    try:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",
                "--language",
                langs.replace(",", "+"),
                "--deskew",
                "--optimize",
                "1",
                str(in_pdf),
                str(out_pdf),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return proc.returncode == 0 and out_pdf.exists()
    except Exception:
        return False


def image_ocr(path: Path, langs: str = "eng") -> str:
    """OCR an image file using pytesseract.

    Returns extracted text or "" on failure or if deps missing.
    """
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore

        img = Image.open(str(path))
        txt = pytesseract.image_to_string(img, lang=langs.replace(",", "+"))
        return txt or ""
    except Exception:
        return ""


def extract_attachment_text(
    original_path: Path,
    mime: str,
    cache_dir: Path,
    cfg: OCRConfig,
) -> Tuple[Optional[Path], str]:
    """Extract text from an attachment.

    - For PDFs: try text extraction; OCR if content is too sparse. Cache to text.txt in the sha256 dir.
    - For images: OCR via tesseract if available.
    Returns (path_text, text). path_text may be None if nothing was extracted.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = cache_dir / "text.txt"

    # If cached, return
    if text_path.exists():
        try:
            txt = text_path.read_text(encoding="utf-8", errors="ignore")
            return text_path, txt
        except Exception:
            pass

    mime_lower = (mime or "").lower()
    ext = original_path.suffix.lower()

    if mime_lower.startswith("application/pdf") or ext == ".pdf":
        text, pages = pdf_extract_text(original_path)
        need_ocr = False
        if pages > 0:
            avg = (len(text) / pages) if pages else len(text)
            need_ocr = avg < cfg.min_chars_per_page
        else:
            need_ocr = len(text) < (cfg.min_chars_per_page * 2)

        if need_ocr:
            ocr_pdf = cache_dir / "ocr.pdf"
            if run_ocrmypdf(original_path, ocr_pdf, cfg.langs):
                text, _ = pdf_extract_text(ocr_pdf)

        if text:
            try:
                text_path.write_text(text, encoding="utf-8")
                return text_path, text
            except Exception:
                return None, text
        return None, ""

    if mime_lower.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        text = image_ocr(original_path, cfg.langs)
        if text:
            try:
                text_path.write_text(text, encoding="utf-8")
                return text_path, text
            except Exception:
                return None, text
        return None, ""

    # TODO: Office docs (docx, xlsx, pptx) using python-docx/docx2txt, etc.
    return None, ""

