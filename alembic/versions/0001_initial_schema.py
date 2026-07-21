"""initial schema: quota_buckets and stats

Revision ID: 0001
Revises:
Create Date: 2026-07-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "quota_buckets",
        sa.Column("user_email", sa.Text(), primary_key=True),
        sa.Column("seconds_remaining", sa.Float()),
        sa.Column("last_update", sa.Float()),
        sa.Column("max_seconds", sa.Float()),
    )
    op.create_table(
        "stats",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Float()),
    )


def downgrade() -> None:
    op.drop_table("stats")
    op.drop_table("quota_buckets")
