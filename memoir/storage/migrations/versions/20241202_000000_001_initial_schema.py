"""Initial three-layer schema: episodes, snapshots, memories

Revision ID: 001
Revises: 
Create Date: 2024-12-02 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === Episodes table ===
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("started_at", sa.BigInteger(), nullable=False),
        sa.Column("ended_at", sa.BigInteger(), nullable=True),
        sa.Column("title", sa.String(256), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("tags_json", sa.Text(), server_default="[]"),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
    )

    # === Snapshots table ===
    op.create_table(
        "snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "episode_id",
            sa.String(64),
            sa.ForeignKey("episodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("captured_at", sa.BigInteger(), nullable=False),
        sa.Column("app", sa.String(128), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("window_title", sa.String(512), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("extra_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
    )
    
    # Snapshot indexes
    op.create_index("idx_snapshots_episode", "snapshots", ["episode_id"])
    op.create_index("idx_snapshots_captured_at", "snapshots", ["captured_at"])
    op.create_index("idx_snapshots_app", "snapshots", ["app"])

    # === Memories table ===
    op.create_table(
        "memories",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column(
            "episode_id",
            sa.String(64),
            sa.ForeignKey("episodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "snapshot_id",
            sa.String(64),
            sa.ForeignKey("snapshots.id", ondelete="CASCADE"),
            nullable=True,
            unique=True,
        ),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("bullets_json", sa.Text(), server_default="[]"),
        sa.Column("tags_json", sa.Text(), server_default="[]"),
        sa.Column("entities_json", sa.Text(), server_default="[]"),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("kind IN ('snapshot', 'episode')", name="ck_memories_kind"),
    )
    
    # Memory indexes
    op.create_index("idx_memories_episode", "memories", ["episode_id"])
    op.create_index("idx_memories_kind", "memories", ["kind"])
    op.create_index("idx_memories_snapshot", "memories", ["snapshot_id"])


def downgrade() -> None:
    # Drop indexes first
    op.drop_index("idx_memories_snapshot", table_name="memories")
    op.drop_index("idx_memories_kind", table_name="memories")
    op.drop_index("idx_memories_episode", table_name="memories")
    
    op.drop_index("idx_snapshots_app", table_name="snapshots")
    op.drop_index("idx_snapshots_captured_at", table_name="snapshots")
    op.drop_index("idx_snapshots_episode", table_name="snapshots")
    
    # Drop tables
    op.drop_table("memories")
    op.drop_table("snapshots")
    op.drop_table("episodes")

