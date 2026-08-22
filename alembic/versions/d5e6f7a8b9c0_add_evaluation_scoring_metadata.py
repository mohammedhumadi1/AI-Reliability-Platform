"""Add evaluation and scoring metadata.

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d5e6f7a8b9c0"
down_revision: Union[
    str,
    Sequence[str],
    None,
] = "c4d5e6f7a8b9"
branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None
depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


def upgrade() -> None:
    op.add_column(
        "evaluation_metrics",
        sa.Column(
            "evaluation_version",
            sa.String(length=50),
            nullable=True,
        ),
    )
    op.add_column(
        "evaluation_metrics",
        sa.Column(
            "embedding_model_name",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.add_column(
        "evaluation_metrics",
        sa.Column(
            "scoring_version",
            sa.String(length=50),
            nullable=True,
        ),
    )
    op.add_column(
        "evaluation_metrics",
        sa.Column(
            "score_profile",
            sa.String(length=50),
            nullable=True,
        ),
    )
    op.add_column(
        "evaluation_metrics",
        sa.Column(
            "weights_used",
            sa.JSON(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "evaluation_metrics",
        "weights_used",
    )
    op.drop_column(
        "evaluation_metrics",
        "score_profile",
    )
    op.drop_column(
        "evaluation_metrics",
        "scoring_version",
    )
    op.drop_column(
        "evaluation_metrics",
        "embedding_model_name",
    )
    op.drop_column(
        "evaluation_metrics",
        "evaluation_version",
    )
