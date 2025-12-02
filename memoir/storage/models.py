"""
SQLAlchemy models for the three-layer memoir architecture.

- Episode: A cluster/session of snapshots & memories (a task, a session, a "story")
- Snapshot: Raw capture (screenshot + OCR + UI metadata)
- Memory: LLM note about something (tied to a snapshot or episode)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, List, Literal, Optional, TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    BigInteger,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

if TYPE_CHECKING:
    pass


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


def now_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


# =============================================================================
# Episode Model
# =============================================================================


class Episode(Base):
    """
    A cluster/session of snapshots that belong to the same activity.

    Episodes group related snapshots over time - like a work session,
    a browsing session, or a specific task.
    """

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[int] = mapped_column(BigInteger, nullable=False)  # unix ms
    ended_at: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )  # nullable until closed
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    created_at: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    updated_at: Mapped[int] = mapped_column(BigInteger, default=now_ms, onupdate=now_ms)

    # Relationships
    snapshots: Mapped[List["Snapshot"]] = relationship(
        "Snapshot", back_populates="episode", cascade="all, delete-orphan"
    )
    memories: Mapped[List["Memory"]] = relationship(
        "Memory", back_populates="episode", cascade="all, delete-orphan"
    )

    @property
    def tags(self) -> List[str]:
        """Get tags as a Python list."""
        return json.loads(self.tags_json) if self.tags_json else []

    @tags.setter
    def tags(self, value: List[str]) -> None:
        """Set tags from a Python list."""
        self.tags_json = json.dumps(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __repr__(self) -> str:
        return f"<Episode(id={self.id!r}, title={self.title!r})>"


# =============================================================================
# Snapshot Model
# =============================================================================


class Snapshot(Base):
    """
    A raw capture - what actually happened on-screen at a point in time.

    Contains the screenshot reference, OCR text, app/window metadata,
    and any extra UI element information.
    """

    __tablename__ = "snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    episode_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True
    )
    captured_at: Mapped[int] = mapped_column(
        BigInteger, nullable=False, index=True
    )  # unix ms
    app: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    window_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON for UI elements
    created_at: Mapped[int] = mapped_column(BigInteger, default=now_ms)

    # Relationships
    episode: Mapped[Optional["Episode"]] = relationship(
        "Episode", back_populates="snapshots"
    )
    memory: Mapped[Optional["Memory"]] = relationship(
        "Memory", back_populates="snapshot", uselist=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_snapshots_episode", "episode_id"),
        Index("idx_snapshots_captured_at", "captured_at"),
        Index("idx_snapshots_app", "app"),
    )

    @property
    def extra(self) -> dict[str, Any]:
        """Get extra data as a Python dict."""
        return json.loads(self.extra_json) if self.extra_json else {}

    @extra.setter
    def extra(self, value: dict[str, Any]) -> None:
        """Set extra data from a Python dict."""
        self.extra_json = json.dumps(value) if value else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "captured_at": self.captured_at,
            "app": self.app,
            "url": self.url,
            "window_title": self.window_title,
            "image_path": self.image_path,
            "ocr_text": self.ocr_text,
            "extra": self.extra,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return f"<Snapshot(id={self.id!r}, app={self.app!r})>"


# =============================================================================
# Memory Model
# =============================================================================


class Memory(Base):
    """
    The user-facing notes you show/search - LLM-generated content.

    Can be at snapshot level (memory about a single capture) or
    episode level (summary of a whole episode/session).

    Invariants:
    - kind='snapshot' → snapshot_id must be non-null
    - kind='episode' → snapshot_id is null, episode_id links it
    """

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # 'snapshot' or 'episode'
    episode_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True
    )
    snapshot_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("snapshots.id", ondelete="CASCADE"),
        nullable=True,
        unique=True,
    )

    # LLM fields
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    bullets_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    tags_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    entities_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array

    # Derived text for search
    search_text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[int] = mapped_column(BigInteger, default=now_ms)
    updated_at: Mapped[int] = mapped_column(BigInteger, default=now_ms, onupdate=now_ms)

    # Relationships
    episode: Mapped[Optional["Episode"]] = relationship(
        "Episode", back_populates="memories"
    )
    snapshot: Mapped[Optional["Snapshot"]] = relationship(
        "Snapshot", back_populates="memory"
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_memories_episode", "episode_id"),
        Index("idx_memories_kind", "kind"),
        Index("idx_memories_snapshot", "snapshot_id"),
        CheckConstraint("kind IN ('snapshot', 'episode')", name="ck_memories_kind"),
    )

    @property
    def bullets(self) -> List[str]:
        """Get bullets as a Python list."""
        return json.loads(self.bullets_json) if self.bullets_json else []

    @bullets.setter
    def bullets(self, value: List[str]) -> None:
        """Set bullets from a Python list."""
        self.bullets_json = json.dumps(value)

    @property
    def tags(self) -> List[str]:
        """Get tags as a Python list."""
        return json.loads(self.tags_json) if self.tags_json else []

    @tags.setter
    def tags(self, value: List[str]) -> None:
        """Set tags from a Python list."""
        self.tags_json = json.dumps(value)

    @property
    def entities(self) -> List[str]:
        """Get entities as a Python list."""
        return json.loads(self.entities_json) if self.entities_json else []

    @entities.setter
    def entities(self, value: List[str]) -> None:
        """Set entities from a Python list."""
        self.entities_json = json.dumps(value)

    @staticmethod
    def build_search_text(
        title: str,
        summary: str,
        bullets: List[str],
        tags: List[str],
        entities: List[str],
    ) -> str:
        """Build the search_text field from components."""
        parts = [title, summary]
        parts.extend(bullets)
        parts.extend(tags)
        parts.extend(entities)
        return " ".join(filter(None, parts))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "kind": self.kind,
            "episode_id": self.episode_id,
            "snapshot_id": self.snapshot_id,
            "title": self.title,
            "summary": self.summary,
            "bullets": self.bullets,
            "tags": self.tags,
            "entities": self.entities,
            "search_text": self.search_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_chroma_metadata(self, snapshot: Optional[Snapshot] = None) -> dict[str, Any]:
        """
        Create metadata dict for Chroma indexing.

        Includes minimal but search-relevant fields.
        """
        metadata: dict[str, Any] = {
            "kind": self.kind,
            "episode_id": self.episode_id or "",
            "snapshot_id": self.snapshot_id or "",
            "title": self.title,
        }

        # Add snapshot-derived fields if available
        if snapshot:
            metadata["app"] = snapshot.app or ""
            metadata["captured_at"] = snapshot.captured_at
            metadata["window_title"] = snapshot.window_title or ""
        elif self.kind == "snapshot" and self.snapshot_id:
            metadata["app"] = ""
            metadata["captured_at"] = 0
            metadata["window_title"] = ""

        return metadata

    def __repr__(self) -> str:
        return f"<Memory(id={self.id!r}, kind={self.kind!r}, title={self.title!r})>"


# =============================================================================
# Helper Functions for creating models from dicts (for backwards compat)
# =============================================================================


def episode_from_dict(data: dict[str, Any]) -> Episode:
    """Create an Episode from a dictionary."""
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = json.loads(tags)

    episode = Episode(
        id=data["id"],
        started_at=data["started_at"],
        ended_at=data.get("ended_at"),
        title=data.get("title"),
        summary=data.get("summary"),
        created_at=data.get("created_at", now_ms()),
        updated_at=data.get("updated_at", now_ms()),
    )
    episode.tags = tags
    return episode


def snapshot_from_dict(data: dict[str, Any]) -> Snapshot:
    """Create a Snapshot from a dictionary."""
    extra = data.get("extra", {})
    if isinstance(extra, str):
        extra = json.loads(extra)

    snapshot = Snapshot(
        id=data["id"],
        episode_id=data.get("episode_id"),
        captured_at=data["captured_at"],
        app=data.get("app"),
        url=data.get("url"),
        window_title=data.get("window_title"),
        image_path=data.get("image_path"),
        ocr_text=data.get("ocr_text"),
        created_at=data.get("created_at", now_ms()),
    )
    snapshot.extra = extra
    return snapshot


def memory_from_dict(data: dict[str, Any]) -> Memory:
    """Create a Memory from a dictionary."""
    bullets = data.get("bullets", [])
    if isinstance(bullets, str):
        bullets = json.loads(bullets)

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = json.loads(tags)

    entities = data.get("entities", [])
    if isinstance(entities, str):
        entities = json.loads(entities)

    search_text = data.get("search_text", "")
    if not search_text:
        search_text = Memory.build_search_text(
            data.get("title", ""),
            data.get("summary", ""),
            bullets,
            tags,
            entities,
        )

    memory = Memory(
        id=data["id"],
        kind=data["kind"],
        episode_id=data.get("episode_id"),
        snapshot_id=data.get("snapshot_id"),
        title=data["title"],
        summary=data["summary"],
        search_text=search_text,
        created_at=data.get("created_at", now_ms()),
        updated_at=data.get("updated_at", now_ms()),
    )
    memory.bullets = bullets
    memory.tags = tags
    memory.entities = entities
    return memory
