import hashlib
from pathlib import Path


class UploadValidationError(ValueError):
    """Raised when an uploaded file should not be processed."""


def sanitize_filename(filename: str) -> str:
    """Return a safe display/storage name for an uploaded PDF."""
    base_name = Path(filename or "document.pdf").name
    cleaned = []

    for char in base_name:
        if char.isalnum() or char in {".", "-", "_", " "}:
            cleaned.append(char)
        else:
            cleaned.append("_")

    safe_name = "".join(cleaned).strip(" ._")
    if not safe_name:
        return "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    return safe_name


def sha256_bytes(file_bytes: bytes) -> str:
    """Return a stable content hash for duplicate detection and document IDs."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_pdf_upload(file_bytes: bytes, max_size_mb: int) -> None:
    """Validate basic PDF upload constraints before writing to disk."""
    if not file_bytes:
        raise UploadValidationError("The uploaded PDF is empty.")

    max_size_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        raise UploadValidationError(
            f"The uploaded PDF is larger than the {max_size_mb} MB limit."
        )

    if not file_bytes.startswith(b"%PDF-"):
        raise UploadValidationError("The uploaded file does not look like a PDF.")
