"""Persistent Chroma vector storage for company knowledge bases."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from chromadb.errors import NotFoundError

from evaluation.generation.embedding_service import get_embedding_model


load_dotenv()


CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_VERSION = "v2"


def collection_name_for_project(project_id: str) -> str:
    """Return a stable Chroma-safe collection name for a project."""
    clean_project_id = project_id.strip()
    if not clean_project_id:
        raise ValueError("project_id cannot be empty.")

    digest = hashlib.sha256(
        clean_project_id.encode("utf-8")
    ).hexdigest()[:24]

    return f"kb_{COLLECTION_VERSION}_{digest}"


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(project_id: str):
    return get_client().get_or_create_collection(
        name=collection_name_for_project(project_id)
    )


def get_collection_if_exists(project_id: str):
    """Return an existing project collection without creating one."""
    collection_name = collection_name_for_project(
        project_id
    )

    try:
        return get_client().get_collection(
            name=collection_name
        )
    except (NotFoundError, ValueError):
        return None


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return [
        vector.tolist()
        for vector in embeddings
    ]


def collection_record_count(project_id: str) -> int:
    collection = get_collection_if_exists(project_id)

    if collection is None:
        return 0

    return int(collection.count())


def add_chunks(
    project_id: str,
    chunks: list[str],
    source: str,
    document_id: str,
) -> None:
    """Embed and upsert document chunks with collision-safe IDs."""
    clean_chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]

    if not clean_chunks:
        return

    collection = get_or_create_collection(project_id)
    embeddings = _embed_texts(clean_chunks)

    ids = [
        f"{document_id}_chunk_{index}"
        for index in range(len(clean_chunks))
    ]

    metadatas = [
        {
            "project_id": project_id,
            "source": source,
            "document_id": document_id,
            "chunk_index": index,
        }
        for index in range(len(clean_chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=clean_chunks,
        metadatas=metadatas,
    )


def delete_document(
    project_id: str,
    document_id: str,
) -> bool:
    """Delete all Chroma records belonging to one document."""
    collection = get_collection_if_exists(project_id)

    if collection is None:
        return False

    collection.delete(
        where={
            "document_id": document_id,
        }
    )
    return True


def query_similar_chunks(
    project_id: str,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Return nearest company-document chunks for a question."""
    clean_query = query.strip()

    if not clean_query:
        raise ValueError("query cannot be empty.")

    collection = get_collection_if_exists(project_id)

    if collection is None:
        return []

    record_count = int(collection.count())

    if record_count == 0:
        return []

    n_results = min(
        max(1, int(top_k)),
        record_count,
    )

    query_embedding = _embed_texts(
        [clean_query]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    distances = (
        results.get("distances")
        or [[]]
    )[0]

    matches: list[dict] = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            and metadatas[index]
            else {}
        )

        distance = (
            float(distances[index])
            if index < len(distances)
            and distances[index] is not None
            else None
        )

        matches.append(
            {
                "text": document or "",
                "source": metadata.get(
                    "source",
                    "",
                ),
                "document_id": metadata.get(
                    "document_id",
                    "",
                ),
                "distance": distance,
            }
        )

    return matches
