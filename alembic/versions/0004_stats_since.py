"""stats_since: when the durable stats counters started counting

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-26

"""
import time
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The counters in `stats` carry no timestamps. A database that already holds counters
# predates this migration, and none can be older than revision 0001 (2026-07-09), which
# is also when the production database was created.
INITIAL_SCHEMA_TS = 1783555200  # 2026-07-09T00:00:00Z


def upgrade() -> None:
    conn = op.get_bind()
    has_counters = conn.execute(sa.text("SELECT 1 FROM stats LIMIT 1")).first() is not None
    since = INITIAL_SCHEMA_TS if has_counters else time.time()
    conn.execute(sa.text("INSERT INTO stats (key, value) VALUES ('stats_since', :since)"), {"since": since})


def downgrade() -> None:
    op.execute("DELETE FROM stats WHERE key = 'stats_since'")
