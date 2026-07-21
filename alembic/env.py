"""Alembic migration environment.

The URL is resolved dynamically so the same migrations apply to SQLite locally
and Postgres on xhost. It is taken from (in order): the ``sqlalchemy.url`` option
set by the caller (see ``db.run_migrations``), then ``DATABASE_URL`` in the
environment, then a local SQLite default.

We deliberately do NOT call ``logging.config.fileConfig`` here: migrations run
inside the app process at startup, and reconfiguring logging would clobber the
app's own handlers.
"""

import os

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

# No SQLAlchemy models in this project; migrations are hand-written, so
# autogenerate diffing is not used.
target_metadata = None


def get_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return "sqlite:///" + os.path.abspath(os.path.join("local_data", "state.db"))


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
