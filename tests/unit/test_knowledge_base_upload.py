import asyncio
import os

import pytest
from fastapi import HTTPException

from app.routers import knowledge_base as kb_router
from knowledge_base.document_loader import (
    InvalidPDFError,
    extract_text_from_pdf,
)


class FakeUploadFile:
    def __init__(
        self,
        data,
        filename="policy.pdf",
        content_type="application/pdf",
    ):
        self.data = data
        self.filename = filename
        self.content_type = content_type
        self.offset = 0
        self.closed = False

    async def read(self, size=-1):
        if self.offset >= len(self.data):
            return b""

        if size < 0:
            chunk = self.data[self.offset :]
            self.offset = len(self.data)
            return chunk

        end = self.offset + size
        chunk = self.data[
            self.offset : end
        ]
        self.offset += len(chunk)
        return chunk

    async def close(self):
        self.closed = True


class FakeScalarResult:
    def __init__(self, value=None):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDB:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = None
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement):
        return FakeScalarResult(
            self.existing
        )

    def add(self, document):
        self.added = document

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def run_upload(file, db=None):
    return asyncio.run(
        kb_router.upload_document(
            project_id="project-a",
            file=file,
            db=db or FakeDB(),
        )
    )


def test_upload_rejects_non_pdf_extension():
    file = FakeUploadFile(
        b"%PDF-1.7\n",
        filename="notes.txt",
        content_type="application/pdf",
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 400
    assert file.closed is True


def test_upload_rejects_wrong_content_type():
    file = FakeUploadFile(
        b"%PDF-1.7\n",
        content_type="text/plain",
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 415
    assert file.closed is True


def test_upload_rejects_empty_file():
    file = FakeUploadFile(b"")

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 400
    assert file.closed is True


def test_upload_rejects_file_over_size_limit(
    monkeypatch,
):
    monkeypatch.setattr(
        kb_router,
        "MAX_PDF_UPLOAD_BYTES",
        8,
    )
    monkeypatch.setattr(
        kb_router,
        "MAX_PDF_UPLOAD_MB",
        20,
    )
    monkeypatch.setattr(
        kb_router,
        "UPLOAD_READ_CHUNK_BYTES",
        4,
    )

    file = FakeUploadFile(
        b"%PDF-123456789"
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 413
    assert (
        "20 MiB"
        in exc_info.value.detail
    )
    assert file.closed is True


def test_upload_rejects_missing_pdf_header(
    monkeypatch,
):
    removed_paths = []
    real_remove = os.remove

    def tracked_remove(path):
        removed_paths.append(path)
        real_remove(path)

    monkeypatch.setattr(
        kb_router.os,
        "remove",
        tracked_remove,
    )

    file = FakeUploadFile(
        b"this is not a pdf"
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 400
    assert file.closed is True
    assert len(removed_paths) == 1
    assert not os.path.exists(
        removed_paths[0]
    )


def test_upload_reports_corrupted_pdf(
    monkeypatch,
):
    def fail_index(**kwargs):
        raise InvalidPDFError(
            "broken pdf"
        )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        fail_index,
    )

    file = FakeUploadFile(
        b"%PDF-1.7\nbroken"
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 400
    assert file.closed is True


def test_upload_converts_no_text_to_422(
    monkeypatch,
):
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: {
            "success": False,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 0,
            "message": (
                "No text could be extracted "
                "from this PDF."
            ),
        },
    )

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        lambda project_id, document_id: (
            vector_calls.append(
                (project_id, document_id)
            )
        ),
    )

    file = FakeUploadFile(
        b"%PDF-1.7\nno text"
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        run_upload(file)

    assert exc_info.value.status_code == 422
    assert len(vector_calls) == 1
    assert file.closed is True


def test_upload_cleans_vectors_when_indexing_raises(
    monkeypatch,
):
    vector_calls = []

    def fail_index(**kwargs):
        raise RuntimeError(
            "embedding failure"
        )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        fail_index,
    )

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        lambda project_id, document_id: (
            vector_calls.append(
                (project_id, document_id)
            )
        ),
    )

    file = FakeUploadFile(
        b"%PDF-1.7\ncontent"
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failure",
    ):
        run_upload(file)

    assert len(vector_calls) == 1
    assert file.closed is True


def test_upload_sanitizes_filename_and_indexes(
    monkeypatch,
):
    captured = {}

    def fake_index_document(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 3,
            "message": "indexed",
        }

    monkeypatch.setattr(
        kb_router,
        "index_document",
        fake_index_document,
    )

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "a" * 64,
    )

    db = FakeDB()
    file = FakeUploadFile(
        b"%PDF-1.7\ncontent",
        filename=(
            "..\\unsafe\\policy.pdf"
        ),
    )

    response = run_upload(
        file,
        db=db,
    )

    assert response["success"] is True
    assert response["duplicate"] is False
    assert captured["source_name"] == "policy.pdf"
    assert db.added.source_name == "policy.pdf"
    assert db.added.status == "INDEXED"
    assert db.commits == 1
    assert file.closed is True


def test_document_loader_wraps_corrupted_pdf(
    tmp_path,
):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(
        b"not a real pdf"
    )

    with pytest.raises(
        InvalidPDFError
    ):
        extract_text_from_pdf(
            str(path)
        )


def test_upload_closes_file_for_empty_project_id():
    file = FakeUploadFile(
        b"%PDF-1.7\ncontent"
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.upload_document(
                project_id="   ",
                file=file,
                db=FakeDB(),
            )
        )

    assert exc_info.value.status_code == 400
    assert file.closed is True


def test_upload_limit_reads_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "KB_MAX_UPLOAD_MB",
        "50",
    )

    assert (
        kb_router._load_max_pdf_upload_mb()
        == 50
    )


def test_upload_limit_rejects_invalid_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "KB_MAX_UPLOAD_MB",
        "invalid",
    )

    with pytest.raises(
        RuntimeError,
        match="positive integer",
    ):
        kb_router._load_max_pdf_upload_mb()
