"""
pdf_processor.py
================
Handles turning an uploaded file (PDF or ZIP) into clean, usable text
pieces.

TWO THINGS COME OUT OF THIS FILE:
1. The full text of the document(s), extracted.
2. That text split into individual SENTENCES -- these sentences are the
   unit both the retriever (retriever.py) and the MiniLLM trainer
   (llm_engine.py) work with.

We use pdfplumber because it handles real-world PDFs (multi-page
reports, articles, etc.) more reliably than simpler libraries.

ZIP SUPPORT:
A ZIP file can contain multiple .pdf and/or .txt files (e.g. a folder
of research papers, or several chapters of a book). We extract text
from every supported file inside the zip and combine it into one pool
of sentences, so you can ask questions across the whole collection at
once.
"""

import os
import re
import zipfile
import tempfile
import pdfplumber

from code_analyzer import is_code_file, analyze_code_file, build_code_sentences, build_explanation_text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and pulls out all the PROSE text (paragraphs, headers),
    page by page -- deliberately EXCLUDING any tables, which are
    handled separately by extract_tables_from_pdf() below.

    WHY SEPARATE TABLES FROM PROSE?
    A plain page.extract_text() call reads everything on the page in
    visual top-to-bottom order, which means a table's cells (numbers,
    short labels, no punctuation) get flattened into the same line as
    the paragraph before/after it -- e.g. "...quarter. Region Q1 Q2 Q3
    North 120 150 180... Overall, the North region..." all becomes ONE
    run-on sentence, since there's no period between table cells to
    split on. That garbled text is useless for search and confuses the
    tokenizer. Excluding each detected table's bounding box before
    extracting text keeps the prose clean; the table's actual data is
    extracted separately, structured, in extract_tables_from_pdf().
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()

            page_for_text = page
            for table in tables:
                page_for_text = page_for_text.outside_bbox(table.bbox)

            text = page_for_text.extract_text()
            if text:
                pages_text.append(text)
    return "\n".join(pages_text)


def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    Extracts every table in the PDF as clean, structured data, and
    turns each one into readable sentences (e.g. "In the table on page
    1, for North: Q1 is 120, Q2 is 150, Q3 is 180.") -- so table
    contents stay searchable and coherent, instead of being a jumble of
    numbers glued to nearby paragraph text.
    """
    table_sentences = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.find_tables():
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue

                header = [str(c).strip() if c else "" for c in rows[0]]
                data_rows = rows[1:]

                # Describe the table's shape once, so questions like
                # "how many rows/columns" or "what does this table show" work.
                table_sentences.append(
                    f"There is a table on page {page_num} with columns: "
                    f"{', '.join(h for h in header if h)}. It has {len(data_rows)} data rows."
                )

                # Then describe each row in plain language, using the
                # first column as the row's label when possible.
                for row in data_rows:
                    cells = [str(c).strip() if c else "" for c in row]
                    if not any(cells):
                        continue
                    row_label = cells[0] if cells[0] else "a row"
                    parts = [
                        f"{header[i]} is {cells[i]}"
                        for i in range(1, min(len(header), len(cells)))
                        if header[i] and cells[i]
                    ]
                    if parts:
                        table_sentences.append(
                            f"In the table on page {page_num}, for {row_label}: {', '.join(parts)}."
                        )

    return table_sentences


def split_into_sentences(text: str) -> list:
    """
    Splits raw extracted text into a list of individual sentences.

    PDFs often have odd line breaks in the middle of sentences (because
    of how the original document was laid out), so we first collapse
    all whitespace, then split on sentence-ending punctuation.
    """
    # Collapse all whitespace/newlines into single spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # Split on '.', '!', '?' followed by a space (keeps the punctuation
    # attached to the sentence it ends).
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        # Skip near-empty fragments (page numbers, headers, stray characters).
        if len(s.split()) >= 3:
            sentences.append(s)

    return sentences


NOISE_PATH_MARKERS = (
    "node_modules/", ".git/", "__pycache__/", "venv/", ".venv/",
    "dist/", "build/", "site-packages/", ".egg-info/",
)

# Cap on how many code files we'll individually parse inside a ZIP --
# a project with hundreds of files would make upload painfully slow
# otherwise. PDFs/TXT files aren't capped since they're usually few.
MAX_CODE_FILES_IN_ZIP = 60


def extract_text_from_zip(zip_path: str) -> dict:
    """
    Opens a ZIP archive and extracts content from every .pdf, .txt, AND
    code file inside it (ignores images, binaries, and common
    dependency/build folders like node_modules or venv). This means a
    zipped-up code PROJECT (not just documents) can be uploaded and
    asked about, the same way a single code file can.

    Returns the combined prose text, structured table sentences (from
    any PDFs), structured code sentences and explanations (from any
    code files), and the lists of files that were processed vs skipped.
    """
    combined_text = []
    processed_files = []
    skipped_files = []
    table_sentences = []
    code_sentences = []
    code_explanations = []
    code_files_processed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Skip directories and macOS metadata junk.
                if member.endswith("/") or "__MACOSX" in member or member.startswith("."):
                    continue

                # Skip dependency/build noise -- not the user's own code.
                if any(marker in member for marker in NOISE_PATH_MARKERS):
                    skipped_files.append(member)
                    continue

                lower = member.lower()
                is_pdf = lower.endswith(".pdf")
                is_txt = lower.endswith(".txt")
                is_code = is_code_file(member)

                if not (is_pdf or is_txt or is_code):
                    skipped_files.append(member)
                    continue

                if is_code and code_files_processed >= MAX_CODE_FILES_IN_ZIP:
                    skipped_files.append(member + " (code file limit reached)")
                    continue

                extracted_path = zf.extract(member, tmp_dir)

                try:
                    if is_pdf:
                        text = extract_text_from_pdf(extracted_path)
                        table_sentences.extend(extract_tables_from_pdf(extracted_path))
                        if text and text.strip():
                            combined_text.append(text)
                            processed_files.append(member)
                        else:
                            skipped_files.append(member)

                    elif is_txt:
                        with open(extracted_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        if text and text.strip():
                            combined_text.append(text)
                            processed_files.append(member)
                        else:
                            skipped_files.append(member)

                    else:  # code file
                        with open(extracted_path, "r", encoding="utf-8", errors="ignore") as f:
                            source = f.read()
                        if not source.strip():
                            skipped_files.append(member)
                            continue
                        analysis = analyze_code_file(source, member)
                        code_sentences.extend(build_code_sentences(analysis, member))
                        code_explanations.append(build_explanation_text(analysis, member))
                        processed_files.append(member)
                        code_files_processed += 1

                except Exception:
                    # A single unreadable file inside the zip shouldn't
                    # fail the whole upload -- just skip it.
                    skipped_files.append(member)

    return {
        "text": "\n".join(combined_text),
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "table_sentences": table_sentences,
        "code_sentences": code_sentences,
        "code_explanations": code_explanations,
    }


def process_uploaded_file(file_path: str, original_filename: str, max_sentences_for_training: int = 400) -> dict:
    """
    Full pipeline for a PDF, a ZIP (of PDFs/text files), or a single
    CODE file: file(s) -> extracted text/structure -> sentence list.

    Returns a dict with:
      - all_sentences: every sentence found (used for TF-IDF retrieval,
        which is cheap and scales fine to a whole document/collection)
      - training_sentences: a capped subset (used to train the tiny
        MiniLLM, which needs to stay fast -- see llm_engine.py)
      - source_files: which file(s) contributed text
      - doc_type: "pdf" | "zip" | "code"
      - code_explanation: pre-built structured explanation text
        (only set when doc_type == "code")
    """
    lower_name = original_filename.lower()
    code_explanation = None
    table_sentences = []
    zip_code_sentences = []

    if lower_name.endswith(".zip"):
        result = extract_text_from_zip(file_path)
        raw_text = result["text"]
        source_files = result["processed_files"]
        skipped_files = result["skipped_files"]
        table_sentences = result["table_sentences"]
        zip_code_sentences = result["code_sentences"]
        zip_code_explanations = result["code_explanations"]
        doc_type = "zip"

        if not source_files:
            raise ValueError(
                "No readable .pdf, .txt, or code files were found inside this ZIP "
                "(dependency folders like node_modules/venv are skipped automatically)."
            )

        if zip_code_explanations:
            # Combine every file's explanation into one project-level
            # explanation, so "explain this code" works for a whole
            # zipped codebase, not just a single uploaded file.
            code_explanation = (
                f"This ZIP contains {len(zip_code_explanations)} code file(s):\n\n"
                + "\n\n---\n\n".join(zip_code_explanations)
            )

    elif lower_name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
        table_sentences = extract_tables_from_pdf(file_path)
        source_files = [original_filename]
        skipped_files = []
        doc_type = "pdf"

    elif is_code_file(original_filename):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        analysis = analyze_code_file(source_code, original_filename)
        code_explanation = build_explanation_text(analysis, original_filename)
        code_sentences = build_code_sentences(analysis, original_filename)

        # Also feed the file's docstrings/comments in as extra sentences
        # so narrow questions ("what does load_config do") can be found
        # via retrieval too, not just the top-level structural summary.
        sentences = code_sentences
        source_files = [original_filename]
        skipped_files = []
        doc_type = "code"

        return {
            "all_sentences": sentences,
            "training_sentences": sentences[:max_sentences_for_training],
            "num_sentences": len(sentences),
            "source_files": source_files,
            "skipped_files": skipped_files,
            "doc_type": doc_type,
            "code_explanation": code_explanation,
        }

    else:
        raise ValueError("Only .pdf, .zip, and common code files (.py, .js, .java, ...) are supported.")

    sentences = split_into_sentences(raw_text) + table_sentences + zip_code_sentences

    if not sentences:
        raise ValueError(
            "No extractable text found. Scanned/image-only PDFs can't be read "
            "by this tool (there's no OCR step)."
        )

    return {
        "all_sentences": sentences,
        "training_sentences": sentences[:max_sentences_for_training],
        "num_sentences": len(sentences),
        "source_files": source_files,
        "skipped_files": skipped_files,
        "doc_type": doc_type,
        "code_explanation": code_explanation,
    }


# Kept for backward compatibility with any code calling the old name directly.
def process_pdf(pdf_path: str, max_sentences_for_training: int = 400) -> dict:
    return process_uploaded_file(pdf_path, "document.pdf", max_sentences_for_training)
