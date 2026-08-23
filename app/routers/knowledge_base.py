from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    KnowledgeBaseDocument,
)
from app.schemas.evaluation_result import (
    KnowledgeBaseVerificationResponse,
)
from knowledge_base.indexing_service import (
    calculate_file_sha256,
    index_document,
)
from knowledge_base.vector_store import (
    delete_document as delete_vector_document,
)
from knowledge_base.verification_agent import (
    verify_answer,
)


router = APIRouter(
    prefix="/api/v1/knowledge-base",
    tags=["Knowledge Base"],
)


@router.post("/upload")
async def upload_document(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    clean_project_id = project_id.strip()

    if not clean_project_id:
        raise HTTPException(
            status_code=400,
            detail="project_id cannot be empty.",
        )

    source_name = (
        file.filename
        or "uploaded.pdf"
    )

    if not source_name.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are currently "
                "supported."
            ),
        )

    temp_path = None
    document_id = uuid.uuid4()

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            shutil.copyfileobj(
                file.file,
                temp_file,
            )
            temp_path = temp_file.name

        if os.path.getsize(temp_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        content_sha256 = (
            calculate_file_sha256(
                temp_path
            )
        )

        existing = db.execute(
            select(
                KnowledgeBaseDocument
            ).where(
                KnowledgeBaseDocument.project_id
                == clean_project_id,
                KnowledgeBaseDocument.content_sha256
                == content_sha256,
            )
        ).scalar_one_or_none()

        if existing is not None:
            return {
                "success": True,
                "duplicate": True,
                "document_id": str(
                    existing.id
                ),
                "chunks_indexed": (
                    existing.chunks_indexed
                ),
                "message": (
                    "This exact document is "
                    "already indexed for the "
                    "project."
                ),
            }

        result = index_document(
            file_path=temp_path,
            project_id=clean_project_id,
            source_name=source_name,
            document_id=str(
                document_id
            ),
        )

        if not result.get(
            "success"
        ):
            return result

        db_document = (
            KnowledgeBaseDocument(
                id=document_id,
                project_id=(
                    clean_project_id
                ),
                source_name=source_name,
                content_sha256=(
                    content_sha256
                ),
                chunks_indexed=(
                    int(
                        result[
                            "chunks_indexed"
                        ]
                    )
                ),
                status="INDEXED",
            )
        )

        try:
            db.add(
                db_document
            )
            db.commit()

        except Exception:
            db.rollback()

            delete_vector_document(
                project_id=(
                    clean_project_id
                ),
                document_id=str(
                    document_id
                ),
            )
            raise

        return {
            **result,
            "duplicate": False,
        }

    finally:
        await file.close()

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )


@router.get("/documents")
def list_documents(
    project_id: str,
    db: Session = Depends(get_db),
) -> dict:
    clean_project_id = project_id.strip()

    if not clean_project_id:
        raise HTTPException(
            status_code=400,
            detail="project_id cannot be empty.",
        )

    documents = db.execute(
        select(
            KnowledgeBaseDocument
        )
        .where(
            KnowledgeBaseDocument.project_id
            == clean_project_id
        )
        .order_by(
            KnowledgeBaseDocument.created_at.desc()
        )
    ).scalars().all()

    return {
        "project_id": clean_project_id,
        "documents": [
            {
                "document_id": str(document.id),
                "source_name": document.source_name,
                "content_sha256": (
                    document.content_sha256
                ),
                "chunks_indexed": (
                    document.chunks_indexed
                ),
                "status": document.status,
                "created_at": document.created_at,
            }
            for document in documents
        ],
    }


@router.delete(
    "/documents/{document_id}"
)
def delete_knowledge_base_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(
        KnowledgeBaseDocument,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base document not found.",
        )

    project_id = document.project_id
    document_id_text = str(document.id)
    original_status = document.status

    # Persist an intermediate state before touching
    # the external vector store. If final DB cleanup
    # fails, the incomplete deletion remains visible.
    document.status = "DELETING"

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    try:
        delete_vector_document(
            project_id=project_id,
            document_id=document_id_text,
        )
    except Exception as exc:
        document.status = original_status

        try:
            db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=503,
            detail=(
                "Failed to delete document "
                "from the vector store."
            ),
        ) from exc

    try:
        db.delete(document)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "success": True,
        "document_id": document_id_text,
        "project_id": project_id,
        "message": (
            "Knowledge base document deleted."
        ),
    }


@router.post(
    "/verify",
    response_model=(
        KnowledgeBaseVerificationResponse
    ),
)
async def verify_against_knowledge_base(
    project_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    rag_context: str | None = Form(
        default=None
    ),
) -> KnowledgeBaseVerificationResponse:
    contexts = (
        [rag_context]
        if (
            rag_context
            and rag_context.strip()
        )
        else []
    )

    result = verify_answer(
        project_id=project_id,
        question=question,
        answer=answer,
        rag_contexts=contexts,
    )

    return KnowledgeBaseVerificationResponse(
        status=result.status,
        evidence_found=(
            result.evidence_found
        ),
        is_supported=(
            result.is_supported
        ),
        best_match_text=(
            result.best_match_text
        ),
        best_match_source=(
            result.best_match_source
        ),
        similarity_distance=(
            result.similarity_distance
        ),
        question_relevance_score=(
            result.question_relevance_score
        ),
        answer_support_score=(
            result.answer_support_score
        ),
        context_alignment_score=(
            result.context_alignment_score
        ),
        numeric_contradiction=(
            result.numeric_contradiction
        ),
        explanation=(
            result.explanation
        ),
    )
