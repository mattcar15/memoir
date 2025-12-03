"""Add search_index table for unified hybrid search

Revision ID: 002
Revises: 001
Create Date: 2024-12-02 00:00:01.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === SearchIndex table ===
    op.create_table(
        "search_index",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("entity_type", sa.String(16), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("title", sa.String(256), nullable=True),
        sa.Column("captured_at", sa.BigInteger(), nullable=True),
        sa.Column("app", sa.String(128), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.CheckConstraint(
            "entity_type IN ('snapshot', 'episode', 'memory')",
            name="ck_search_index_type"
        ),
    )

    # Indexes
    op.create_index("idx_search_index_type", "search_index", ["entity_type"])
    op.create_index(
        "idx_search_index_entity",
        "search_index",
        ["entity_type", "entity_id"],
        unique=True
    )
    op.create_index("idx_search_index_captured", "search_index", ["captured_at"])
    op.create_index("idx_search_index_app", "search_index", ["app"])

    # === FTS5 Virtual Table for BM25 ===
    # Note: Using raw SQL for FTS5 since Alembic doesn't support virtual tables directly
    op.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index_fts USING fts5(
            search_text,
            title,
            content='search_index',
            content_rowid='rowid'
        )
    """)

    # Triggers to keep FTS in sync
    op.execute("""
        CREATE TRIGGER IF NOT EXISTS search_index_ai AFTER INSERT ON search_index BEGIN
            INSERT INTO search_index_fts(rowid, search_text, title)
            VALUES (NEW.rowid, NEW.search_text, NEW.title);
        END
    """)

    op.execute("""
        CREATE TRIGGER IF NOT EXISTS search_index_ad AFTER DELETE ON search_index BEGIN
            INSERT INTO search_index_fts(search_index_fts, rowid, search_text, title)
            VALUES('delete', OLD.rowid, OLD.search_text, OLD.title);
        END
    """)

    op.execute("""
        CREATE TRIGGER IF NOT EXISTS search_index_au AFTER UPDATE ON search_index BEGIN
            INSERT INTO search_index_fts(search_index_fts, rowid, search_text, title)
            VALUES('delete', OLD.rowid, OLD.search_text, OLD.title);
            INSERT INTO search_index_fts(rowid, search_text, title)
            VALUES (NEW.rowid, NEW.search_text, NEW.title);
        END
    """)


def downgrade() -> None:
    # Drop FTS triggers and table
    op.execute("DROP TRIGGER IF EXISTS search_index_au")
    op.execute("DROP TRIGGER IF EXISTS search_index_ad")
    op.execute("DROP TRIGGER IF EXISTS search_index_ai")
    op.execute("DROP TABLE IF EXISTS search_index_fts")

    # Drop indexes
    op.drop_index("idx_search_index_app", table_name="search_index")
    op.drop_index("idx_search_index_captured", table_name="search_index")
    op.drop_index("idx_search_index_entity", table_name="search_index")
    op.drop_index("idx_search_index_type", table_name="search_index")

    # Drop table
    op.drop_table("search_index")

