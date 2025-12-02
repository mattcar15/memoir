"""
Image processing pipeline for screenshot analysis and memory consolidation.

This module implements the complete pipeline for processing screenshots:
1. pHash-based deduplication (checks last 5 minutes)
2. Image downscaling for efficient processing
3. VLM (MLX) processing for image understanding and text extraction
4. Memory consolidation using cosine similarity
"""

import time
import io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import imagehash
import numpy as np

from ...config import RESOLUTION_TIERS, MLX_MODEL_NAME, MLX_MAX_NEW_TOKENS, SCREENSHOT_PROMPT_PATH
from ..mlx_processor import get_global_processor, load_prompt
from ...storage.embeddings import create_embedding


class Memory:
    """Represents a consolidated memory with multiple images and summaries."""

    def __init__(
        self,
        memory_id: str,
        timestamp: datetime,
        summary: str,
        embedding: List[float],
        image_path: Optional[Path] = None,
        processing_method: Optional[str] = None,
        extracted_text: Optional[str] = None,
        ocr_stats: Optional[Dict[str, Any]] = None,
        llm_stats: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        bullets: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ):
        self.memory_id = memory_id
        self.created_at = timestamp
        self.last_updated = timestamp
        self.embedding = embedding
        self.entries: List[Dict[str, Any]] = []
        self.summaries: List[str] = []
        self.image_paths: List[Path] = []
        self.image_count = 0

        self.add_image(
            summary,
            embedding,
            timestamp,
            image_path=image_path,
            processing_method=processing_method,
            extracted_text=extracted_text,
            ocr_stats=ocr_stats,
            llm_stats=llm_stats,
            title=title,
            bullets=bullets,
            tags=tags,
            entities=entities,
        )

    def _compute_average_embedding(self) -> List[float]:
        """Compute the average embedding across all entries."""
        valid_embeddings = [
            entry["embedding"]
            for entry in self.entries
            if entry.get("embedding") is not None
        ]

        if not valid_embeddings:
            return self.embedding

        avg_embedding = [0.0] * len(valid_embeddings[0])
        for emb in valid_embeddings:
            avg_embedding = [a + b for a, b in zip(avg_embedding, emb)]

        count = len(valid_embeddings)
        return [value / count for value in avg_embedding]

    def add_image(
        self,
        summary: str,
        embedding: List[float],
        timestamp: datetime,
        image_path: Optional[Path] = None,
        processing_method: Optional[str] = None,
        extracted_text: Optional[str] = None,
        ocr_stats: Optional[Dict[str, Any]] = None,
        llm_stats: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        bullets: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ):
        """Add a new image/summary to this memory."""
        self.summaries.append(summary)
        self.last_updated = timestamp

        entry_image_path = None
        if image_path:
            if not isinstance(image_path, Path):
                image_path = Path(image_path)
            self.image_paths.append(image_path)
            entry_image_path = str(image_path)

        entry = {
            "summary": summary,
            "embedding": embedding,
            "timestamp": timestamp,
            "image_path": entry_image_path,
            "processing_method": processing_method,
            "extracted_text": extracted_text,
            "ocr_stats": ocr_stats,
            "llm_stats": llm_stats,
            "title": title,
            "bullets": bullets or [],
            "tags": tags or [],
            "entities": entities or [],
        }
        self.entries.append(entry)

        # Update embedding by averaging all entry embeddings
        self.embedding = self._compute_average_embedding()
        self.image_count = len(self.entries)

    def is_active(self, current_time: datetime, window_minutes: int = 5) -> bool:
        """Check if memory is still in the active window."""
        return (current_time - self.last_updated).total_seconds() < (
            window_minutes * 60
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary format."""
        return {
            "memory_id": self.memory_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "summaries": self.summaries,
            "image_paths": [str(p) for p in self.image_paths if p],
            "image_count": self.image_count,
            "entries": [
                {
                    "summary": entry["summary"],
                    "title": entry.get("title"),
                    "bullets": entry.get("bullets", []),
                    "tags": entry.get("tags", []),
                    "entities": entry.get("entities", []),
                    "timestamp": (
                        entry["timestamp"].isoformat()
                        if entry.get("timestamp")
                        else None
                    ),
                    "image_path": entry.get("image_path"),
                    "processing_method": entry.get("processing_method"),
                }
                for entry in self.entries
            ],
        }


class ImagePipeline:
    """Main pipeline for processing screenshots and managing memories."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        memory_window_minutes: int = 5,
        phash_threshold: int = 10,
        embedding_model: str = "embeddinggemma",
        mlx_model: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the image pipeline.

        Args:
            similarity_threshold: Cosine similarity threshold for memory consolidation (0-1)
            memory_window_minutes: Time window for active memories (minutes)
            phash_threshold: pHash difference threshold for duplicate detection
            embedding_model: Model name for embeddings
            mlx_model: MLX VLM model name (default: from config)
            verbose: If True, print detailed processing information
        """
        self.similarity_threshold = similarity_threshold
        self.memory_window_minutes = memory_window_minutes
        self.phash_threshold = phash_threshold
        self.embedding_model = embedding_model
        self.mlx_model = mlx_model or MLX_MODEL_NAME
        self.verbose = verbose
        
        # Get MLX processor (shared instance)
        self.mlx_processor = get_global_processor(
            model_name=self.mlx_model,
            max_new_tokens=MLX_MAX_NEW_TOKENS,
            verbose=verbose,
        )

        # Memory tracking
        self.memories: List[Memory] = []
        self.recent_hashes: List[Tuple[datetime, str]] = []  # (timestamp, phash)

        # Statistics
        self.stats = {
            "total_processed": 0,
            "duplicates_skipped": 0,
            "vlm_processed": 0,
            "memories_created": 0,
            "images_consolidated": 0,
        }

    def compute_phash(self, image: Image.Image) -> str:
        """Compute perceptual hash of an image."""
        return str(imagehash.phash(image))

    def is_duplicate(self, image: Image.Image, current_time: datetime) -> bool:
        """
        Check if image is a duplicate of recent images using pHash.

        Args:
            image: PIL Image to check
            current_time: Current timestamp

        Returns:
            True if image is a duplicate, False otherwise
        """
        # Compute hash
        image_hash = imagehash.phash(image)

        # Clean up old hashes (older than memory window)
        cutoff_time = current_time - timedelta(minutes=self.memory_window_minutes)
        self.recent_hashes = [
            (ts, h) for ts, h in self.recent_hashes if ts > cutoff_time
        ]

        # Check against recent hashes
        for ts, recent_hash_str in self.recent_hashes:
            recent_hash = imagehash.hex_to_hash(recent_hash_str)
            hash_diff = image_hash - recent_hash

            if hash_diff <= self.phash_threshold:
                return True

        # Not a duplicate, add to recent hashes
        self.recent_hashes.append((current_time, str(image_hash)))
        return False

    def downscale_image(
        self, image: Image.Image, tier: str = "balanced"
    ) -> Image.Image:
        """
        Downscale image to target resolution tier.

        Args:
            image: PIL Image to downscale
            tier: Resolution tier from config

        Returns:
            Downscaled PIL Image
        """
        target_width, target_height = RESOLUTION_TIERS.get(
            tier, RESOLUTION_TIERS["balanced"]
        )

        # Calculate scaling to maintain aspect ratio
        original_width, original_height = image.size
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        scale_ratio = min(width_ratio, height_ratio)

        # Only downscale if image is larger than target
        if scale_ratio >= 1.0:
            return image

        # Calculate new dimensions
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # Resize image
        return image.resize((new_width, new_height), resample=Image.LANCZOS)

    def process_with_vlm(
        self, image: Image.Image
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process image with MLX VLM model using structured prompt.

        Args:
            image: PIL Image to process

        Returns:
            Tuple of (structured_data dict with title/summary/bullets/tags/entities, stats dict)
        """
        # Load prompt from file
        try:
            prompt = load_prompt(SCREENSHOT_PROMPT_PATH)
        except FileNotFoundError:
            # Fallback to basic prompt if file not found
            print(f"⚠️  Prompt file not found: {SCREENSHOT_PROMPT_PATH}, using fallback")
            prompt = "Describe this screenshot as a JSON object with: title, summary, bullets, tags, entities"
        
        return self.mlx_processor.process_image_structured(image, prompt)

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_matching_memory(
        self, embedding: List[float], current_time: datetime
    ) -> Optional[Memory]:
        """
        Find an active memory that matches the given embedding.

        Args:
            embedding: Embedding vector to match
            current_time: Current timestamp

        Returns:
            Matching Memory object or None
        """
        best_match = None
        best_similarity = 0.0

        for memory in self.memories:
            # Only consider active memories
            if not memory.is_active(current_time, self.memory_window_minutes):
                continue

            # Calculate similarity
            similarity = self.cosine_similarity(embedding, memory.embedding)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = memory

        return best_match

    def save_image(
        self, image: Image.Image, output_dir: Path, prefix: str = "processed"
    ) -> Path:
        """
        Save image to disk.

        Args:
            image: PIL Image to save
            output_dir: Directory to save to
            prefix: Filename prefix

        Returns:
            Path to saved image
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = output_dir / filename
        image.save(filepath, format="PNG")
        return filepath

    def process_image(
        self,
        image: Image.Image,
        output_dir: Optional[Path] = None,
        save_images: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image through the complete pipeline.

        Args:
            image: PIL Image to process
            output_dir: Directory to save processed images
            save_images: Whether to save images to disk

        Returns:
            Dictionary with processing results and statistics, or None if skipped
        """
        start_time = time.time()
        current_time = datetime.now()
        self.stats["total_processed"] += 1

        print(f"\n📸 Processing image #{self.stats['total_processed']}...")

        # Step 1: Check for duplicates using pHash
        print("🔍 Checking for duplicates (pHash)...")
        if self.is_duplicate(image, current_time):
            print("⏭️  Skipping duplicate image")
            self.stats["duplicates_skipped"] += 1
            return None

        print("✅ Image is unique")

        # Step 2: Downscale image for processing
        print("🔽 Downscaling image for processing...")
        downscaled = self.downscale_image(image, tier="balanced")
        print(f"✅ Downscaled to {downscaled.size[0]}x{downscaled.size[1]}")

        # Step 3: Process with VLM model
        print("🤖 Processing with VLM model...")
        structured_data, vlm_stats = self.process_with_vlm(downscaled)
        processing_method = "vlm"
        self.stats["vlm_processed"] += 1

        if not structured_data:
            print("❌ Failed to generate structured output from VLM")
            return None

        # Extract fields from structured data
        title = structured_data.get("title", "Untitled")
        summary = structured_data.get("summary", "")
        bullets = structured_data.get("bullets", [])
        tags = structured_data.get("tags", [])
        entities = structured_data.get("entities", [])

        if not summary:
            print("❌ No summary in structured output")
            return None

        if self.verbose:
            print(f"📝 Title: {title}")
            print(f"📝 Summary (full): {summary}")
            print(f"📝 Tags: {tags}")
            print(f"📝 Entities: {entities}")
        else:
            print(f"📝 Title: {title}")
            print(f"📝 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")

        # Step 5: Create embedding
        print("🔄 Creating embedding...")
        embedding = create_embedding(summary, self.embedding_model)
        if not embedding:
            print("❌ Failed to create embedding")
            return None

        # Step 6: Find matching memory or create new one
        print("🔍 Searching for matching memory...")
        matching_memory = self.find_matching_memory(embedding, current_time)

        image_path = None
        is_new_memory = False
        memory_id = None

        if matching_memory:
            # Add to existing memory
            print(f"✅ Found matching memory: {matching_memory.memory_id}")

            # Save downscaled image for subsequent images in memory
            if save_images and output_dir:
                print("💾 Saving downscaled image...")
                image_path = self.save_image(
                    downscaled, output_dir, prefix="downscaled"
                )
                print(f"✅ Saved: {image_path}")

            matching_memory.add_image(
                summary,
                embedding,
                current_time,
                image_path=image_path,
                processing_method=processing_method,
                extracted_text=None,
                ocr_stats=None,
                llm_stats=vlm_stats,
                title=title,
                bullets=bullets,
                tags=tags,
                entities=entities,
            )
            memory_id = matching_memory.memory_id
            self.stats["images_consolidated"] += 1

        else:
            # Create new memory
            print("🆕 Creating new memory")
            memory_id = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            # Save original resolution for first image
            if save_images and output_dir:
                print("💾 Saving original resolution image...")
                image_path = self.save_image(image, output_dir, prefix="original")
                print(f"✅ Saved: {image_path}")

            new_memory = Memory(
                memory_id,
                current_time,
                summary,
                embedding,
                image_path,
                processing_method=processing_method,
                extracted_text=None,
                ocr_stats=None,
                llm_stats=vlm_stats,
                title=title,
                bullets=bullets,
                tags=tags,
                entities=entities,
            )
            self.memories.append(new_memory)
            is_new_memory = True
            self.stats["memories_created"] += 1

        total_time = time.time() - start_time

        # Compile results
        result = {
            "memory_id": memory_id,
            "is_new_memory": is_new_memory,
            "processing_method": processing_method,
            "title": title,
            "summary": summary,
            "bullets": bullets,
            "tags": tags,
            "entities": entities,
            "image_path": str(image_path) if image_path else None,
            "timestamp": current_time.isoformat(),
            "total_processing_time_seconds": round(total_time, 3),
            "vlm_stats": vlm_stats,
        }

        print(f"✅ Processing complete in {total_time:.2f}s")

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        current_time = datetime.now()
        active_memories = sum(
            1
            for m in self.memories
            if m.is_active(current_time, self.memory_window_minutes)
        )

        images_per_memory = {}
        for memory in self.memories:
            count = memory.image_count
            images_per_memory[count] = images_per_memory.get(count, 0) + 1

        return {
            **self.stats,
            "total_memories": len(self.memories),
            "active_memories": active_memories,
            "images_per_memory_distribution": images_per_memory,
        }

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("📊 PIPELINE STATISTICS")
        print("=" * 80)
        print(f"Total images processed: {stats['total_processed']}")
        print(f"Duplicates skipped: {stats['duplicates_skipped']}")
        print(f"Images processed with VLM: {stats['vlm_processed']}")
        print(f"\nMemories created: {stats['memories_created']}")
        print(
            f"Images consolidated into existing memories: {stats['images_consolidated']}"
        )
        print(f"Total memories: {stats['total_memories']}")
        print(f"Active memories: {stats['active_memories']}")

        if stats["images_per_memory_distribution"]:
            print(f"\nImages per memory distribution:")
            for count, num_memories in sorted(
                stats["images_per_memory_distribution"].items()
            ):
                print(f"  {count} images: {num_memories} memories")

        print("=" * 80)
