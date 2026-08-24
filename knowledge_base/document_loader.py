"""Document loading helpers."""

import pymupdf


class InvalidPDFError(ValueError):
    """Raised when an uploaded PDF cannot be parsed."""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from all PDF pages."""
    try:
        with pymupdf.open(
            file_path
        ) as document:
            return "\n".join(
                page.get_text()
                for page in document
            )

    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
    ) as exc:
        raise InvalidPDFError(
            "PDF is invalid or corrupted."
        ) from exc
