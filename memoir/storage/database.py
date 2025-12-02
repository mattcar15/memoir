"""
SQLAlchemy database layer for the three-layer memoir architecture.

Provides engine/session management and database initialization.
SQLite is the source of truth; Chroma is the search index.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Episode, Snapshot, Memory

# Default database filename
DEFAULT_DB_NAME = "memoir.db"

# Global engine reference (set during init)
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


# =============================================================================
# SQLite Configuration
# =============================================================================


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign keys and WAL mode for SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


# =============================================================================
# Path Utilities
# =============================================================================


def get_db_path(logs_dir: Path | str) -> Path:
    """Get the database path for a given logs directory."""
    return Path(logs_dir) / DEFAULT_DB_NAME


def get_db_url(logs_dir: Path | str) -> str:
    """Get the SQLAlchemy database URL for a given logs directory."""
    db_path = get_db_path(logs_dir)
    return f"sqlite:///{db_path}"


# =============================================================================
# Engine & Session Management
# =============================================================================


def create_db_engine(logs_dir: Path | str, echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine for the database.

    Args:
        logs_dir: Directory containing the database
        echo: Whether to echo SQL statements (for debugging)

    Returns:
        SQLAlchemy Engine
    """
    db_url = get_db_url(logs_dir)
    engine = create_engine(
        db_url,
        echo=echo,
        # SQLite specific settings
        connect_args={"check_same_thread": False},
    )
    return engine


def get_engine() -> Engine:
    """Get the global engine (must call init_database first)."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the global session factory (must call init_database first)."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            memories = session.query(Memory).all()
            ...
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_session(engine: Engine) -> Session:
    """
    Create a new session from an engine.

    Useful for one-off operations or testing.
    """
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# =============================================================================
# Database Initialization
# =============================================================================


def init_database(
    logs_dir: Path | str,
    echo: bool = False,
    create_fts: bool = True,
) -> Path:
    """
    Initialize the database in the given logs directory.

    Creates the directory if needed, initializes the engine,
    creates tables, and sets up the global session factory.

    Args:
        logs_dir: Directory to store the database
        echo: Whether to echo SQL statements
        create_fts: Whether to create FTS5 virtual table

    Returns:
        Path to the database file
    """
    global _engine, _SessionLocal

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    db_path = get_db_path(logs_dir)

    # Create engine
    _engine = create_db_engine(logs_dir, echo=echo)

    # Create session factory
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    # Create all tables
    Base.metadata.create_all(_engine)

    # Create FTS if requested
    if create_fts:
        _create_fts_tables(_engine)

    return db_path


def _create_fts_tables(engine: Engine) -> None:
    """Create FTS5 virtual table for full-text search."""
    with engine.connect() as conn:
        # Check if FTS table already exists
        result = conn.execute(
            text(
                """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='memories_fts'
        """
            )
        )
        if result.fetchone() is not None:
            return  # Already exists

        try:
            # Create FTS5 virtual table
            conn.execute(
                text(
                    """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    search_text,
                    content='memories',
                    content_rowid='rowid'
                )
            """
                )
            )

            # Create triggers to keep FTS in sync
            conn.execute(
                text(
                    """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, search_text) VALUES (NEW.rowid, NEW.search_text);
                END
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text) VALUES('delete', OLD.rowid, OLD.search_text);
                END
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text) VALUES('delete', OLD.rowid, OLD.search_text);
                    INSERT INTO memories_fts(rowid, search_text) VALUES (NEW.rowid, NEW.search_text);
                END
            """
                )
            )

            conn.commit()
        except Exception as e:
            print(f"Warning: Could not create FTS5 table: {e}")


# =============================================================================
# Database Info & Utilities
# =============================================================================


def get_table_counts(session: Session) -> dict[str, int]:
    """Get row counts for all tables."""
    return {
        "episodes": session.query(Episode).count(),
        "snapshots": session.query(Snapshot).count(),
        "memories": session.query(Memory).count(),
    }


def get_database_info(session: Session) -> dict:
    """Get database statistics and info."""
    from sqlalchemy import func

    counts = get_table_counts(session)

    # Get date range for snapshots
    result = session.query(
        func.min(Snapshot.captured_at), func.max(Snapshot.captured_at)
    ).first()
    min_captured, max_captured = result if result else (None, None)

    # Check if FTS is available
    engine = session.get_bind()
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='memories_fts'
        """
            )
        )
        has_fts = result.fetchone() is not None

    return {
        "counts": counts,
        "snapshot_range": {
            "min_captured_at": min_captured,
            "max_captured_at": max_captured,
        },
        "has_fts": has_fts,
    }


def rebuild_fts(session: Session) -> None:
    """Rebuild the FTS index from the memories table."""
    engine = session.get_bind()
    with engine.connect() as conn:
        try:
            conn.execute(text("DELETE FROM memories_fts"))
            conn.execute(
                text(
                    """
                INSERT INTO memories_fts(rowid, search_text)
                SELECT rowid, search_text FROM memories
            """
                )
            )
            conn.commit()
        except Exception as e:
            print(f"Warning: Could not rebuild FTS: {e}")


# =============================================================================
# Legacy Compatibility
# =============================================================================


@contextmanager
def get_connection(db_path: Path | str) -> Generator[Session, None, None]:
    """
    Legacy compatibility: Get a session using a db_path.

    This is for backward compatibility with code that expects
    a connection-style interface.
    """
    # Extract logs_dir from db_path
    db_path = Path(db_path)
    if db_path.name == DEFAULT_DB_NAME:
        logs_dir = db_path.parent
    else:
        logs_dir = db_path

    # Create engine if needed
    engine = create_db_engine(logs_dir)
    Base.metadata.create_all(engine)

    session = create_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_connection(db_path: Path | str) -> Session:
    """
    Legacy compatibility: Create a session using a db_path.

    Note: Caller is responsible for closing the session.
    """
    db_path = Path(db_path)
    if db_path.name == DEFAULT_DB_NAME:
        logs_dir = db_path.parent
    else:
        logs_dir = db_path

    engine = create_db_engine(logs_dir)
    Base.metadata.create_all(engine)

    return create_session(engine)
