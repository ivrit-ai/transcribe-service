"""Unified async SQL state layer.

Source of truth for durable data (quota buckets, stats counters).
Sessions stay in memory — they're keyed on an ephemeral cookie and re-login
after a restart is cheap, so there's nothing worth persisting.
Uses SQLite (aiosqlite) when running locally and Postgres (asyncpg) on xhost.

Write ``?``-style SQL everywhere; the Postgres adapter rewrites ``?`` -> ``$n``.
Both backends support ``INSERT ... ON CONFLICT(...) DO UPDATE`` with ``excluded``.

The schema is owned by Alembic (see ``alembic/``); ``run_migrations`` applies it
at startup. Runtime queries use the async drivers below, not SQLAlchemy.
"""

import re
import os
import asyncio
import logging

logger = logging.getLogger("transcribe_service.db")

# Depth columns of queue_samples, in the order they are bound to SQL parameters.
QUEUE_DEPTH_FIELDS = (
    "queued_short",
    "queued_long",
    "queued_private",
    "running_short",
    "running_long",
    "running_private",
    "transcoding_queued",
    "transcoding_running",
)


def run_migrations(url: str):
    """Apply Alembic migrations up to head against ``url`` (synchronous).

    ``url`` is a SQLAlchemy URL: ``sqlite:///abs/path/state.db`` locally or the
    ``postgresql://`` DSN on xhost. Uses sync drivers (stdlib sqlite3 / psycopg2)
    independently of the async runtime pool.
    """
    from alembic.config import Config
    from alembic import command

    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Config(os.path.join(here, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(here, "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    logger.info("Database migrations applied (head)")


class Database:
    def __init__(self, *, backend: str, dsn: str = None, path: str = None):
        assert backend in ("sqlite", "postgres")
        self.backend = backend
        self._dsn = dsn
        self._path = path
        self._pool = None          # postgres
        self._conn = None          # sqlite connection
        self._lock = None          # serialize sqlite writes

    # ---- connection lifecycle ----

    async def connect(self):
        if self.backend == "postgres":
            import asyncpg
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        else:
            import aiosqlite
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            self._conn = await aiosqlite.connect(self._path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            self._lock = asyncio.Lock()
        logger.info("Database connected (backend=%s)", self.backend)

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
        if self._conn is not None:
            await self._conn.close()

    def _q(self, sql: str) -> str:
        if self.backend != "postgres":
            return sql
        counter = {"i": 0}

        def repl(_):
            counter["i"] += 1
            return f"${counter['i']}"

        return re.sub(r"\?", repl, sql)

    async def execute(self, sql: str, *params):
        sql = self._q(sql)
        if self.backend == "postgres":
            await self._pool.execute(sql, *params)
        else:
            async with self._lock:
                await self._conn.execute(sql, params)
                await self._conn.commit()

    async def fetchrow(self, sql: str, *params):
        sql = self._q(sql)
        if self.backend == "postgres":
            row = await self._pool.fetchrow(sql, *params)
            return dict(row) if row else None
        else:
            async with self._lock:
                cur = await self._conn.execute(sql, params)
                row = await cur.fetchone()
                await cur.close()
            return dict(row) if row else None

    async def fetch(self, sql: str, *params):
        sql = self._q(sql)
        if self.backend == "postgres":
            rows = await self._pool.fetch(sql, *params)
            return [dict(r) for r in rows]
        else:
            async with self._lock:
                cur = await self._conn.execute(sql, params)
                rows = await cur.fetchall()
                await cur.close()
            return [dict(r) for r in rows]

    # ---- quota buckets ----

    async def get_quota(self, user_email: str):
        return await self.fetchrow(
            "SELECT seconds_remaining, last_update, max_seconds FROM quota_buckets WHERE user_email = ?",
            user_email,
        )

    async def save_quota(self, user_email: str, seconds_remaining: float, last_update: float, max_seconds: float):
        await self.execute(
            "INSERT INTO quota_buckets (user_email, seconds_remaining, last_update, max_seconds) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_email) DO UPDATE SET "
            "seconds_remaining = excluded.seconds_remaining, last_update = excluded.last_update, max_seconds = excluded.max_seconds",
            user_email,
            seconds_remaining,
            last_update,
            max_seconds,
        )

    # ---- web push subscriptions ----

    async def save_push_subscription(
        self, endpoint: str, user_email: str, p256dh: str, auth: str, lang: str, created_at: int
    ):
        await self.execute(
            "INSERT INTO push_subscriptions (endpoint, user_email, p256dh, auth, lang, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (endpoint) DO UPDATE SET "
            "user_email = excluded.user_email, p256dh = excluded.p256dh, "
            "auth = excluded.auth, lang = excluded.lang",
            endpoint,
            user_email,
            p256dh,
            auth,
            lang,
            created_at,
        )

    async def get_push_subscriptions(self, user_email: str) -> list:
        return await self.fetch(
            "SELECT endpoint, p256dh, auth, lang FROM push_subscriptions WHERE user_email = ?",
            user_email,
        )

    async def delete_push_subscription(self, endpoint: str, user_email: str):
        await self.execute("DELETE FROM push_subscriptions WHERE endpoint = ? AND user_email = ?", endpoint, user_email)

    # ---- stats counters ----

    async def incr_stat(self, key: str, amount: float):
        await self.execute(
            "INSERT INTO stats (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = stats.value + excluded.value",
            key,
            amount,
        )

    async def get_stats(self) -> dict:
        rows = await self.fetch("SELECT key, value FROM stats")
        return {r["key"]: r["value"] for r in rows}

    # ---- history: finished jobs and queue depth over time ----

    async def record_job_event(
        self, ts: int, job_type: str, language: str, audio_seconds: float, transcribe_seconds: float, status: str
    ):
        await self.execute(
            "INSERT INTO job_events (ts, job_type, language, audio_seconds, transcribe_seconds, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ts,
            job_type,
            language,
            audio_seconds,
            transcribe_seconds,
            status,
        )

    async def get_job_buckets(self, since_ts: int, bucket_seconds: int) -> list:
        """Jobs and audio grouped into fixed-width time buckets, split by status.

        Groups by the output ordinal: repeating the bucket expression in GROUP BY would
        bind a second pair of parameters, which Postgres refuses to match to the first.
        """
        return await self.fetch(
            "SELECT (ts / ?) * ? AS bucket, status, "
            "COUNT(*) AS jobs, SUM(audio_seconds) AS audio_seconds, SUM(transcribe_seconds) AS transcribe_seconds "
            "FROM job_events WHERE ts >= ? "
            "GROUP BY 1, status",
            bucket_seconds,
            bucket_seconds,
            since_ts,
        )

    async def get_job_languages(self, since_ts: int) -> list:
        return await self.fetch(
            "SELECT language, COUNT(*) AS jobs, SUM(audio_seconds) AS audio_seconds "
            "FROM job_events WHERE ts >= ? AND status = 'completed' "
            "GROUP BY language ORDER BY COUNT(*) DESC",
            since_ts,
        )

    async def get_queue_sample(self, bucket_ts: int):
        return await self.fetchrow("SELECT * FROM queue_samples WHERE bucket_ts = ?", bucket_ts)

    async def save_queue_sample(self, bucket_ts: int, depths: dict):
        cols = ", ".join(QUEUE_DEPTH_FIELDS)
        placeholders = ", ".join("?" for _ in QUEUE_DEPTH_FIELDS)
        updates = ", ".join(f"{f} = excluded.{f}" for f in QUEUE_DEPTH_FIELDS)
        await self.execute(
            f"INSERT INTO queue_samples (bucket_ts, {cols}) VALUES (?, {placeholders}) "
            f"ON CONFLICT (bucket_ts) DO UPDATE SET {updates}",
            bucket_ts,
            *(depths[f] for f in QUEUE_DEPTH_FIELDS),
        )

    async def get_queue_samples(self, since_ts: int) -> list:
        return await self.fetch(
            "SELECT * FROM queue_samples WHERE bucket_ts >= ? ORDER BY bucket_ts", since_ts
        )

    async def prune_history(self, job_events_before: int, queue_samples_before: int):
        await self.execute("DELETE FROM job_events WHERE ts < ?", job_events_before)
        await self.execute("DELETE FROM queue_samples WHERE bucket_ts < ?", queue_samples_before)
