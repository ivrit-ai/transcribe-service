"""web push subscriptions

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # endpoint is the primary key so a browser re-registering after a permission
    # reset replaces its row instead of accumulating duplicates.
    # lang is stored because the notification text is built when the job finishes,
    # at which point there is no browser to ask which language to use.
    op.create_table(
        "push_subscriptions",
        sa.Column("endpoint", sa.Text(), primary_key=True),
        sa.Column("user_email", sa.Text(), nullable=False),
        sa.Column("p256dh", sa.Text(), nullable=False),
        sa.Column("auth", sa.Text(), nullable=False),
        sa.Column("lang", sa.Text(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
    )
    op.create_index("push_subscriptions_user_idx", "push_subscriptions", ["user_email"])


def downgrade() -> None:
    op.drop_index("push_subscriptions_user_idx", table_name="push_subscriptions")
    op.drop_table("push_subscriptions")
