import pytest

from file_utils import (
    UploadValidationError,
    sanitize_filename,
    sha256_bytes,
    validate_pdf_upload,
)


def test_sanitize_filename_keeps_safe_pdf_name():
    assert sanitize_filename("../My Resume.pdf") == "My Resume.pdf"


def test_sanitize_filename_adds_pdf_extension():
    assert sanitize_filename("quarterly report") == "quarterly report.pdf"


def test_sha256_bytes_is_stable():
    assert sha256_bytes(b"same content") == sha256_bytes(b"same content")


def test_validate_pdf_upload_rejects_empty_file():
    with pytest.raises(UploadValidationError, match="empty"):
        validate_pdf_upload(b"", max_size_mb=1)


def test_validate_pdf_upload_rejects_non_pdf_file():
    with pytest.raises(UploadValidationError, match="PDF"):
        validate_pdf_upload(b"not a pdf", max_size_mb=1)


def test_validate_pdf_upload_accepts_pdf_header():
    validate_pdf_upload(b"%PDF-1.7\nbody", max_size_mb=1)
