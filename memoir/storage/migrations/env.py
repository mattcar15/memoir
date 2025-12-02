"""
Alembic environment configuration for Memoir.

This module configures how Alembic runs migrations.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the project root to the path so we can import memoir
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from memoir.storage.models import Base

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """Get the database URL from environment or default."""
    # Check for environment variable first
    url = os.environ.get("MEMOIR_DATABASE_URL")
    if url:
        return url
    
    # Default to logs/memoir.db relative to project root
    default_db = project_root / "logs" / "memoir.db"
    
    # Ensure parent directory exists
    default_db.parent.mkdir(parents=True, exist_ok=True)
    
    return f"sqlite:///{default_db}"


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render as batch operations for SQLite compatibility
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    # Override URL in config
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Render as batch operations for SQLite compatibility
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

