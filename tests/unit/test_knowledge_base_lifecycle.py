from datetime import datetime
from types import SimpleNamespace
import uuid

import pytest
from chromadb.errors import NotFoundError
from fastapi import HTTPException

from app.routers import knowledge_base as kb_router
from knowledge_base import vector_store


class FakeScalarResult:
    def __init__(self, documents):
        self.documents = documents

    def scalars(self):
        return self

    def all(self):
        return self.documents


class FakeListSession:
    def __init__(self, documents):
        self.documents = documents

    def execute(self, statement):
        return FakeScalarResult(self.documents)


class FakeDeleteSession:
    def __init__(self, document):
        self.document = document
        self.deleted = None
        self.commits = 0
        self.rollbacks = 0

    def get(self, model, document_id):
        if (
            self.document
            and self.document.id == document_id
        ):
            return self.document

        return None

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def delete(self, document):
        self.deleted = document


class MissingCollectionClient:
    def __init__(self):
        self.get_calls = 0

    def get_collection(self, name):
        self.get_calls += 1
        raise NotFoundError(
            f"Collection {name} does not exist."
        )

    def get_or_create_collection(self, name):
        raise AssertionError(
            "Read path must not create collections."
        )


def make_document():
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id="project-a",
        source_name="policy.pdf",
        content_sha256="a" * 64,
        chunks_indexed=4,
        status="INDEXED",
        created_at=datetime(
            2026,
            8,
            23,
            10,
            0,
            0,
        ),
    )


def test_list_documents_returns_project_documents():
    document = make_document()
    db = FakeListSession([document])

    response = kb_router.list_documents(
        project_id=" project-a ",
        db=db,
    )

    assert response["project_id"] == "project-a"
    assert len(response["documents"]) == 1
    assert (
        response["documents"][0]["document_id"]
        == str(document.id)
    )
    assert (
        response["documents"][0]["status"]
        == "INDEXED"
    )


def test_list_documents_rejects_empty_project_id():
    db = FakeListSession([])

    with pytest.raises(HTTPException) as exc_info:
        kb_router.list_documents(
            project_id="   ",
            db=db,
        )

    assert exc_info.value.status_code == 400


def test_delete_document_removes_vectors_then_db(
    monkeypatch,
):
    document = make_document()
    db = FakeDeleteSession(document)
    vector_calls = []

    def fake_delete_vector_document(
        project_id,
        document_id,
    ):
        vector_calls.append(
            (project_id, document_id)
        )
        return True

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        fake_delete_vector_document,
    )

    response = (
        kb_router.delete_knowledge_base_document(
            document_id=document.id,
            db=db,
        )
    )

    assert vector_calls == [
        ("project-a", str(document.id))
    ]
    assert db.deleted is document
    assert db.commits == 2
    assert response["success"] is True


def test_delete_document_restores_status_on_vector_failure(
    monkeypatch,
):
    document = make_document()
    db = FakeDeleteSession(document)

    def fail_vector_delete(
        project_id,
        document_id,
    ):
        raise RuntimeError("vector failure")

    monkeypatch.setattr(
        kb_router,
        "delete_vector_document",
        fail_vector_delete,
    )

    with pytest.raises(HTTPException) as exc_info:
        kb_router.delete_knowledge_base_document(
            document_id=document.id,
            db=db,
        )

    assert exc_info.value.status_code == 503
    assert document.status == "INDEXED"
    assert db.deleted is None
    assert db.commits == 2


def test_delete_document_returns_404_for_unknown_document():
    db = FakeDeleteSession(None)

    with pytest.raises(HTTPException) as exc_info:
        kb_router.delete_knowledge_base_document(
            document_id=uuid.uuid4(),
            db=db,
        )

    assert exc_info.value.status_code == 404


def test_read_paths_do_not_create_missing_collection(
    monkeypatch,
):
    client = MissingCollectionClient()

    monkeypatch.setattr(
        vector_store,
        "get_client",
        lambda: client,
    )

    assert (
        vector_store.collection_record_count(
            "project-a"
        )
        == 0
    )

    assert (
        vector_store.delete_document(
            "project-a",
            "document-a",
        )
        is False
    )

    assert (
        vector_store.query_similar_chunks(
            "project-a",
            "refund policy",
        )
        == []
    )

    assert client.get_calls == 3

def test_read_path_handles_value_error_for_missing_collection(
    monkeypatch,
):
    class LegacyMissingCollectionClient:
        def get_collection(self, name):
            raise ValueError(
                f"Collection {name} does not exist."
            )

        def get_or_create_collection(self, name):
            raise AssertionError(
                "Read path must not create collections."
            )

    monkeypatch.setattr(
        vector_store,
        "get_client",
        lambda: LegacyMissingCollectionClient(),
    )

    assert (
        vector_store.collection_record_count(
            "project-a"
        )
        == 0
    )


def test_read_path_preserves_invalid_project_id_error():
    with pytest.raises(ValueError):
        vector_store.collection_record_count(
            "   "
        )
