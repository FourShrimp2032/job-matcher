from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(RuntimeError):
    pass


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

    return full_text