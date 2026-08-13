import re

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(RuntimeError):
    pass

def normalize_pdf_text(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        chunks = re.split(r"\s{2,}", line)

        normalized_chunks = []

        for chunk in chunks:
            tokens = chunk.split()

            if not tokens:
                continue

            single_char_tokens = sum(
                1 for token in tokens
                if len(token) == 1 and token.isalnum()
            )

            if (
                len(tokens) >= 2
                and single_char_tokens / len(tokens) >= 0.7
            ):
                chunk = "".join(tokens)

            normalized_chunks.append(chunk)

        cleaned_line = " ".join(normalized_chunks)

        cleaned_line = re.sub(r"[ \t]+", " ", cleaned_line)
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines).strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as exc:
        raise PDFExtractionError("Could not read the PDF file.") from exc

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise PDFExtractionError(
            "No readable text was found in the PDF."
        )

    full_text = normalize_pdf_text(full_text)

    return full_text