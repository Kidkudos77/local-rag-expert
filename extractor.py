"""
extractor.py — Dynamic PDF Section Extractor

Dynamically detects top-level headings in any paper format.
Handles IEEE Roman numeral headings, ALL CAPS standalone headings,
multi-line wrapped headings, and pypdf split-word artifacts.
"""

import re
import os
import tempfile
from pypdf import PdfReader


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_uploaded(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


# ── Heading detection helpers ────────────────────────────────────────────────

def _repair_split_words(s: str) -> str:
    """Fix pypdf artifact: 'I NTRODUCTION' → 'INTRODUCTION'"""
    s = re.sub(r'\b([A-Z]) ([A-Z]{2,})', lambda m: m.group(1) + m.group(2), s)
    s = re.sub(r'\b([A-Z]) ([a-z]{2,})', lambda m: m.group(1) + m.group(2), s)
    return s


def _is_mostly_uppercase(text: str) -> bool:
    """Return True if >60% of alphabetic chars are uppercase — signals ALL CAPS heading."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.6


_STANDALONE_CAPS = {
    "ABSTRACT", "REFERENCES", "BIBLIOGRAPHY",
    "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "APPENDIX",
}


def _detect_heading(line: str) -> str | None:
    """
    Returns clean heading title if line is a top-level heading, else None.

    Rules:
      1. Roman numeral prefix + ALL CAPS body:
             "I. INTRODUCTION", "VII. CONCLUSION"
         Body must be mostly uppercase to reject subsections like "C. Outline"
      2. Standalone ALL CAPS known heading:
             "REFERENCES", "ACKNOWLEDGEMENTS"
      3. Em-dash abstract:
             "Abstract—..."
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 110:
        return None

    repaired = _repair_split_words(stripped)

    # Rule 1: Roman numeral + mostly-uppercase body
    roman = re.match(r'^([IVXivx]{1,6})\.\s+(.+)', repaired)
    if roman:
        body = roman.group(2).strip()
        if len(body) >= 3 and _is_mostly_uppercase(body):
            # Clean title: strip roman prefix, title-case
            clean = re.sub(r'^[IVXivx]+\.\s*', '', repaired).strip()
            return clean.title().rstrip(':').strip()

    # Rule 2: Standalone ALL CAPS known heading (1–3 words)
    if repaired.upper() == repaired:
        words = repaired.split()
        if 1 <= len(words) <= 3 and repaired in _STANDALONE_CAPS:
            return repaired.title().strip()

    # Rule 3: Em-dash abstract
    if re.match(r'^Abstract[—–-]', stripped, re.IGNORECASE):
        return "Abstract"

    return None


# ── Section map builder ──────────────────────────────────────────────────────

def build_section_map(full_text: str) -> list:
    """
    Scan document and return list of (title, content_start, content_end).

    Handles multi-line wrapped headings: if a detected heading line is
    immediately followed by another short ALL CAPS line (no roman prefix),
    that line is treated as a continuation of the same heading title.
    """
    lines = full_text.split("\n")
    raw = []  # (line_index, title)

    i = 0
    while i < len(lines):
        title = _detect_heading(lines[i])
        if title:
            # Check if next non-empty line is a wrapped continuation
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_stripped = _repair_split_words(lines[j].strip())
                # Continuation: short, all-caps, no roman prefix, no period structure
                if (next_stripped
                        and len(next_stripped) < 60
                        and _is_mostly_uppercase(next_stripped)
                        and not re.match(r'^[IVXivx]+\.', next_stripped)
                        and not re.match(r'^[A-Z]\.\s', next_stripped)):
                    title = (title + " " + next_stripped.title()).strip()
                    i = j  # skip the continuation line

            raw.append((i, title))
        i += 1

    if not raw:
        return []

    sections = []
    for idx, (line_no, title) in enumerate(raw):
        start = line_no + 1
        end   = raw[idx + 1][0] if idx + 1 < len(raw) else len(lines)
        sections.append((title, start, end))

    return sections


# ── Public API ───────────────────────────────────────────────────────────────

def get_available_sections(full_text: str) -> list:
    """Return section titles found in this document for the UI dropdown."""
    section_map = build_section_map(full_text)
    seen  = set()
    found = []
    for title, _, _ in section_map:
        if title not in seen:
            found.append(title)
            seen.add(title)
    found.append("Custom section")
    return found if len(found) > 1 else ["Custom section"]


def find_section(full_text: str, section_name: str) -> str:
    """Extract content for a named section, up to 3000 chars."""
    if section_name == "Custom section":
        return ""

    lines = full_text.split("\n")

    # Inline em-dash abstract
    if section_name.lower() == "abstract":
        for line in lines:
            em = re.match(r'^Abstract[—–-](.+)', line.strip(), re.IGNORECASE)
            if em:
                section_map = build_section_map(full_text)
                for title, start, end in section_map:
                    if title.lower() == "abstract":
                        body = [em.group(1).strip()] + lines[start:end]
                        return "\n".join(body).strip()[:3000]

    section_map = build_section_map(full_text)
    for title, start, end in section_map:
        if title.lower() == section_name.lower():
            return "\n".join(lines[start:end]).strip()[:3000]

    return ""


def get_raw_preview(full_text: str, chars: int = 3000) -> str:
    return full_text[:chars]


def list_ingested_pdfs(docs_folder: str = "documents") -> list:
    if not os.path.exists(docs_folder):
        return []
    return [f for f in os.listdir(docs_folder)
            if f.endswith(".pdf") or f.endswith(".txt")]
