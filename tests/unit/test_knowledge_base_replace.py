import asyncio
from types import SimpleNamespace
import uuid

import pytest
from fastapi import HTTPException

from app.routers import knowledge_base as kb_router


class FakeUploadFile:
    def __init__(
        self,
        data=b"%PDF-1.7\nreplacement",
        filename="replacement.pdf",
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


class FakeReplaceSession:
    def __init__(
        self,
        document,
        existing=None,
        fail_commit_at=None,
    ):
        self.document = document
        self.existing = existing
        self.fail_commit_at = (
            fail_commit_at
        )
        self.added = None
        self.deleted = []
        self.commits = 0
        self.rollbacks = 0

    def get(
        self,
        model,
        document_id,
    ):
        if (
            self.document is not None
            and self.document.id
            == document_id
        ):
            return self.document

        return None

    def execute(self, statement):
        return FakeScalarResult(
            self.existing
        )

    def add(self, document):
        self.added = document

    def delete(self, document):
        self.deleted.append(document)

    def commit(self):
        self.commits += 1

        if (
            self.fail_commit_at
            == self.commits
        ):
            raise RuntimeError(
                "database failure"
            )

    def rollback(self):
        self.rollbacks += 1


def make_document(
    content_sha256="a" * 64,
    status="INDEXED",
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id="project-a",
        source_name="policy.pdf",
        content_sha256=content_sha256,
        chunks_indexed=4,
        status=status,
    )


def run_replace(
    document,
    file=None,
    existing=None,
):
    db = FakeReplaceSession(
        document=document,
        existing=existing,
    )

    upload = file or FakeUploadFile()

    result = asyncio.run(
        kb_router.replace_knowledge_base_document(
            document_id=document.id,
            file=upload,
            db=db,
        )
    )

    return result, db, upload


def test_replace_document_success(
    monkeypatch,
):
    document = make_document()
    db = FakeReplaceSession(document)
    file = FakeUploadFile(
        filename=(
            "..\\unsafe\\replacement.pdf"
        )
    )

    captured = {}
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

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
        "delete_vector_document",
        lambda project_id, document_id: (
            vector_calls.append(
                (project_id, document_id)
            )
        ),
    )

    response = asyncio.run(
        kb_router.replace_knowledge_base_document(
            document_id=document.id,
            file=file,
            db=db,
        )
    )

    assert response["success"] is True
    assert response["replaced"] is True
    assert (
        response["old_document_id"]
        == str(document.id)
    )
    assert (
        response["document_id"]
        != str(document.id)
    )
    assert (
        captured["source_name"]
        == "replacement.pdf"
    )
    assert (
        captured["document_id"]
        == response["document_id"]
    )
    assert vector_calls == [
        (
            "project-a",
            str(document.id),
        )
    ]
    assert db.added.status == "INDEXED"
    assert document in db.deleted
    assert db.commits == 2
    assert file.closed is True


def test_replace_identical_document_is_noop(
    monkeypatch,
):
    document = make_document()

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: document.content_sha256,
    )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: (
            pytest.fail(
                "Identical document "
                "must not be re-indexed."
            )
        ),
    )

    response, db, file = run_replace(
        document=document
    )

    assert response["success"] is True
    assert response["replaced"] is False
    assert (
        response["document_id"]
        == str(document.id)
    )
    assert db.commits == 0
    assert file.closed is True


def test_replace_rejects_existing_duplicate(
    monkeypatch,
):
    document = make_document()

    existing = make_document(
        content_sha256="b" * 64
    )

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

    db = FakeReplaceSession(
        document=document,
        existing=existing,
    )

    file = FakeUploadFile()

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert db.commits == 0
    assert file.closed is True


def test_replace_index_failure_keeps_old_document(
    monkeypatch,
):
    document = make_document()
    db = FakeReplaceSession(document)
    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

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

    with pytest.raises(
        RuntimeError,
        match="embedding failure",
    ):
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert document.status == "INDEXED"
    assert db.commits == 0
    assert len(vector_calls) == 1
    assert (
        vector_calls[0][1]
        != str(document.id)
    )
    assert file.closed is True


def test_replace_restores_old_on_vector_delete_failure(
    monkeypatch,
):
    document = make_document()
    db = FakeReplaceSession(document)
    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: {
            "success": True,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 2,
            "message": "indexed",
        },
    )

    def fake_delete(
        project_id,
        document_id,
    ):
        vector_calls.append(
            (project_id, document_id)
        )

        if document_id == str(
            document.id
        ):
            raise RuntimeError(
                "old vector delete failed"
            )

        return True

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        fake_delete,
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 503
    assert document.status == "INDEXED"
    assert db.added in db.deleted
    assert db.commits == 2
    assert len(vector_calls) == 2
    assert vector_calls[0] == (
        "project-a",
        str(document.id),
    )
    assert (
        vector_calls[1][1]
        == str(db.added.id)
    )
    assert file.closed is True


def test_replace_rejects_non_indexed_document():
    document = make_document(
        status="REPLACING"
    )
    db = FakeReplaceSession(document)
    file = FakeUploadFile()

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert file.closed is True


def test_replace_returns_404_and_closes_file():
    document_id = uuid.uuid4()

    db = FakeReplaceSession(
        document=None
    )
    file = FakeUploadFile()

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document_id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 404
    assert file.closed is True


def test_replace_no_text_returns_422_and_cleans_vectors(
    monkeypatch,
):
    document = make_document()
    db = FakeReplaceSession(document)
    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

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

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 422
    assert document.status == "INDEXED"
    assert db.commits == 0
    assert len(vector_calls) == 1
    assert file.closed is True


def test_replace_staging_db_failure_cleans_new_vectors(
    monkeypatch,
):
    document = make_document()

    db = FakeReplaceSession(
        document,
        fail_commit_at=1,
    )

    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: {
            "success": True,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 2,
            "message": "indexed",
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

    with pytest.raises(
        RuntimeError,
        match="database failure",
    ):
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert db.commits == 1
    assert db.rollbacks == 1
    assert document.status == "INDEXED"
    assert len(vector_calls) == 1
    assert (
        vector_calls[0][1]
        == str(db.added.id)
    )
    assert file.closed is True


def test_replace_final_db_failure_returns_503(
    monkeypatch,
):
    document = make_document()

    db = FakeReplaceSession(
        document,
        fail_commit_at=2,
    )

    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: {
            "success": True,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 2,
            "message": "indexed",
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

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 503
    assert db.commits == 2
    assert db.rollbacks == 1

    # Only the old vectors were removed.
    # The successfully staged replacement
    # stays available for recovery.
    assert vector_calls == [
        (
            "project-a",
            str(document.id),
        )
    ]

    assert file.closed is True


def test_replace_reports_database_recovery_failure(
    monkeypatch,
):
    document = make_document()

    db = FakeReplaceSession(
        document,
        fail_commit_at=2,
    )

    file = FakeUploadFile()
    vector_calls = []

    monkeypatch.setattr(
        kb_router,
        "calculate_file_sha256",
        lambda path: "b" * 64,
    )

    monkeypatch.setattr(
        kb_router,
        "index_document",
        lambda **kwargs: {
            "success": True,
            "document_id": kwargs[
                "document_id"
            ],
            "chunks_indexed": 2,
            "message": "indexed",
        },
    )

    def fake_delete(
        project_id,
        document_id,
    ):
        vector_calls.append(
            (project_id, document_id)
        )

        if document_id == str(
            document.id
        ):
            raise RuntimeError(
                "old vector delete failed"
            )

        return True

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        fake_delete,
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        asyncio.run(
            kb_router.replace_knowledge_base_document(
                document_id=document.id,
                file=file,
                db=db,
            )
        )

    assert exc_info.value.status_code == 503
    assert (
        "database recovery"
        in exc_info.value.detail
    )
    assert (
        "Manual recovery is required"
        in exc_info.value.detail
    )
    assert db.commits == 2
    assert db.rollbacks == 1

    assert vector_calls == [
        (
            "project-a",
            str(document.id),
        ),
        (
            "project-a",
            str(db.added.id),
        ),
    ]

    assert file.closed is True
