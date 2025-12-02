"""
Vector database functionality using ChromaDB for semantic search.

This is a SEARCH INDEX ONLY - SQLite is the source of truth.
The collection stores memories (search_text + minimal metadata) for fast retrieval.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Any
import json


# Collection name for the memories index
COLLECTION_NAME = "memories_index"


class VectorStore:
    """ChromaDB-based vector store for semantic search of memories."""

    def __init__(self, persist_directory: str = "vector_db"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection for memories
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Memory search index for semantic retrieval"},
        )

        print(f"📦 Vector store initialized: {self.collection.count()} memories")

    def add_memory(
        self,
        memory_id: str,
        search_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a memory to the vector store.

        Args:
            memory_id: Unique identifier for the memory (should match SQLite memories.id)
            search_text: Concatenated searchable text (title+summary+bullets+tags+entities)
            embedding: Embedding vector for the search_text
            metadata: Metadata dict with: kind, episode_id, snapshot_id, app, captured_at, title

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare metadata - convert all values to Chroma-compatible types
            meta = self._prepare_metadata(metadata)

            # Add to collection
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[meta],
            )

            return True

        except Exception as e:
            print(f"❌ Error adding memory to vector store: {e}")
            return False

    def upsert_memory(
        self,
        memory_id: str,
        search_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upsert a memory to the vector store (insert or update).

        Args:
            memory_id: Unique identifier for the memory
            search_text: Concatenated searchable text
            embedding: Embedding vector for the search_text
            metadata: Metadata dict

        Returns:
            True if successful, False otherwise
        """
        try:
            meta = self._prepare_metadata(metadata)

            self.collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[meta],
            )

            return True

        except Exception as e:
            print(f"❌ Error upserting memory to vector store: {e}")
            return False

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare metadata for Chroma storage.

        Chroma requires: str, int, float, or bool values (no None, no nested objects).
        """
        meta = metadata or {}
        meta_prepared = {}

        for key, value in meta.items():
            if isinstance(value, (dict, list)):
                # Serialize complex types as JSON
                meta_prepared[key] = json.dumps(value)
            elif value is None:
                # Convert None to empty string
                meta_prepared[key] = ""
            else:
                # Keep str, int, float, bool as-is
                meta_prepared[key] = value

        return meta_prepared

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using an embedding vector.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters (e.g., {"kind": "snapshot"}, {"app": "Chrome"})
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of memory dictionaries with id, document, distance, and metadata
        """
        try:
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include,
            )

            # Format results
            memories = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    memory = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                    }
                    if include_embeddings and results.get("embeddings"):
                        memory["embedding"] = results["embeddings"][0][i]
                    memories.append(memory)

            return memories

        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []

    def search_by_text(
        self, query_text: str, query_embedding: List[float], n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using a text query.

        Args:
            query_text: Query text (for display purposes)
            query_embedding: Pre-computed embedding for the query
            n_results: Number of results to return

        Returns:
            List of memory dictionaries
        """
        print(f'🔍 Searching for: "{query_text}"')
        results = self.search(query_embedding, n_results)
        print(f"📊 Found {len(results)} similar memories")
        return results

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory dictionary or None if not found
        """
        try:
            results = self.collection.get(
                ids=[memory_id], include=["documents", "metadatas", "embeddings"]
            )

            if results["ids"] and len(results["ids"]) > 0:
                embedding = None
                if results["embeddings"] is not None and len(results["embeddings"]) > 0:
                    embedding = results["embeddings"][0]

                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "embedding": embedding,
                }

            return None

        except Exception as e:
            print(f"❌ Error retrieving memory: {e}")
            return None

    def get_memories(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple memories by their IDs.

        Args:
            memory_ids: List of memory identifiers

        Returns:
            List of memory dictionaries (in order of IDs)
        """
        if not memory_ids:
            return []

        try:
            results = self.collection.get(
                ids=memory_ids, include=["documents", "metadatas"]
            )

            memories = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    memories.append(
                        {
                            "id": results["ids"][i],
                            "document": results["documents"][i],
                            "metadata": (
                                results["metadatas"][i] if results["metadatas"] else {}
                            ),
                        }
                    )

            return memories

        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from the vector store.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"❌ Error deleting memory: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]) -> bool:
        """
        Delete multiple memories from the vector store.

        Args:
            memory_ids: List of memory identifiers

        Returns:
            True if successful, False otherwise
        """
        if not memory_ids:
            return True

        try:
            self.collection.delete(ids=memory_ids)
            return True
        except Exception as e:
            print(f"❌ Error deleting memories: {e}")
            return False

    def count(self) -> int:
        """Get the total number of memories in the store."""
        return self.collection.count()

    def list_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        List the most recent memories.

        Note: ChromaDB doesn't have built-in sorting by timestamp,
        so this returns arbitrary memories up to n.
        For proper time-based queries, use SQLite.

        Args:
            n: Number of memories to return

        Returns:
            List of memory dictionaries
        """
        try:
            results = self.collection.get(limit=n, include=["documents", "metadatas"])

            memories = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    memory = {
                        "id": results["ids"][i],
                        "document": results["documents"][i],
                        "metadata": (
                            results["metadatas"][i] if results["metadatas"] else {}
                        ),
                    }
                    memories.append(memory)

            return memories

        except Exception as e:
            print(f"❌ Error listing memories: {e}")
            return []

    def reset(self) -> bool:
        """
        Clear all memories from the vector store.
        WARNING: This operation cannot be undone!

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Memory search index for semantic retrieval"},
            )
            print("✅ Vector store reset successfully")
            return True
        except Exception as e:
            print(f"❌ Error resetting vector store: {e}")
            return False

    # =========================================================================
    # Legacy compatibility methods (for existing code)
    # =========================================================================

    def add_memory_legacy(
        self,
        memory_id: str,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Legacy method for backward compatibility.

        Wraps add_memory with the old parameter name 'summary' -> 'search_text'.
        """
        return self.add_memory(
            memory_id=memory_id,
            search_text=summary,
            embedding=embedding,
            metadata=metadata,
        )
