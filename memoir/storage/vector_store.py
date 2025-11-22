"""
Vector database functionality using ChromaDB for semantic search.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


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
            name="memories", metadata={"description": "Screenshot memory embeddings"}
        )

        print(f"📦 Vector store initialized: {self.collection.count()} memories")

    def add_memory(
        self,
        memory_id: str,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a memory to the vector store.

        Args:
            memory_id: Unique identifier for the memory
            summary: Text summary of the memory
            embedding: Embedding vector for the summary
            metadata: Additional metadata (timestamp, stats, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare metadata
            meta = metadata or {}

            # Convert all metadata values to strings (ChromaDB requirement)
            meta_str = {}
            for key, value in meta.items():
                if isinstance(value, (dict, list)):
                    meta_str[key] = json.dumps(value)
                elif value is not None:
                    meta_str[key] = str(value)

            # Add to collection
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[meta_str],
            )

            return True

        except Exception as e:
            print(f"❌ Error adding memory to vector store: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using an embedding vector.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters

        Returns:
            List of memory dictionaries with id, summary, distance, and metadata
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results, where=where
            )

            # Format results
            memories = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    memory = {
                        "id": results["ids"][0][i],
                        "summary": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                    }
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

            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "summary": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "embedding": (
                        results["embeddings"][0] if results["embeddings"] else None
                    ),
                }

            return None

        except Exception as e:
            print(f"❌ Error retrieving memory: {e}")
            return None

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

    def count(self) -> int:
        """Get the total number of memories in the store."""
        return self.collection.count()

    def list_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        List the most recent memories.

        Args:
            n: Number of memories to return

        Returns:
            List of memory dictionaries
        """
        try:
            # Get all memories (ChromaDB doesn't have built-in sorting by timestamp)
            results = self.collection.get(limit=n, include=["documents", "metadatas"])

            memories = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    memory = {
                        "id": results["ids"][i],
                        "summary": results["documents"][i],
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
            self.client.delete_collection("memories")
            self.collection = self.client.get_or_create_collection(
                name="memories",
                metadata={"description": "Screenshot memory embeddings"},
            )
            print("✅ Vector store reset successfully")
            return True
        except Exception as e:
            print(f"❌ Error resetting vector store: {e}")
            return False
