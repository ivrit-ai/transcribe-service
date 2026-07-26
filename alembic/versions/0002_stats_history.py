"""stats history: job_events and queue_samples

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ts is whole epoch seconds so that "ts / bucket * bucket" is integer division
    # on both SQLite and Postgres; a float column would round differently on each.
    op.create_table(
        "job_events",
        sa.Column("ts", sa.BigInteger(), nullable=False),
        sa.Column("job_type", sa.Text()),
        sa.Column("language", sa.Text()),
        sa.Column("audio_seconds", sa.Float()),
        sa.Column("transcribe_seconds", sa.Float()),
        sa.Column("status", sa.Text()),
    )
    op.create_index("job_events_ts_idx", "job_events", ["ts"])
    op.create_table(
        "queue_samples",
        sa.Column("bucket_ts", sa.BigInteger(), primary_key=True),
        sa.Column("queued_short", sa.Integer()),
        sa.Column("queued_long", sa.Integer()),
        sa.Column("queued_private", sa.Integer()),
        sa.Column("running_short", sa.Integer()),
        sa.Column("running_long", sa.Integer()),
        sa.Column("running_private", sa.Integer()),
        sa.Column("transcoding_queued", sa.Integer()),
        sa.Column("transcoding_running", sa.Integer()),
    )


def downgrade() -> None:
    op.drop_table("queue_samples")
    op.drop_index("job_events_ts_idx", table_name="job_events")
    op.drop_table("job_events")
