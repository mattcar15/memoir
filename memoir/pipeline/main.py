"""
Image processing pipeline for screenshot analysis and memory consolidation.

This module implements the complete pipeline for processing screenshots:
1. pHash-based deduplication (checks last 5 minutes)
2. Image downscaling for efficient processing
3. OCR text extraction using PaddleOCR
4. LLM processing fallback for image-heavy content
5. Memory consolidation using cosine similarity
"""

import time
import io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import imagehash
import numpy as np
from paddleocr import PaddleOCR

from ..config import RESOLUTION_TIERS
from ..processor import process_with_ollama
from ..embeddings import create_embedding


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
        ocr_text_threshold: int = 100,
        enable_llm: bool = True,
        embedding_model: str = "embeddinggemma",
        ollama_model: str = "gemma3:4b",
        verbose: bool = False,
    ):
        """
        Initialize the image pipeline.

        Args:
            similarity_threshold: Cosine similarity threshold for memory consolidation (0-1)
            memory_window_minutes: Time window for active memories (minutes)
            phash_threshold: pHash difference threshold for duplicate detection
            ocr_text_threshold: Minimum chars for OCR-only processing
            enable_llm: Whether to use LLM for image-heavy content
            embedding_model: Model name for embeddings
            ollama_model: Model name for Ollama LLM processing
        """
        self.similarity_threshold = similarity_threshold
        self.memory_window_minutes = memory_window_minutes
        self.phash_threshold = phash_threshold
        self.ocr_text_threshold = ocr_text_threshold
        self.enable_llm = enable_llm
        self.embedding_model = embedding_model
        self.ollama_model = ollama_model
        self.verbose = verbose

        # Initialize PaddleOCR
        print("🔧 Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True)
        print("✅ PaddleOCR initialized")

        # Memory tracking
        self.memories: List[Memory] = []
        self.recent_hashes: List[Tuple[datetime, str]] = []  # (timestamp, phash)

        # Statistics
        self.stats = {
            "total_processed": 0,
            "duplicates_skipped": 0,
            "ocr_only": 0,
            "llm_processed": 0,
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

    def extract_text_ocr(self, image: Image.Image) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from image using PaddleOCR.

        Args:
            image: PIL Image to extract text from

        Returns:
            Tuple of (extracted text, stats dict)
        """
        start_time = time.time()

        try:
            # PaddleOCR expects RGB images without alpha channel
            if image.mode != "RGB":
                print(f"  🔄 Converting image mode from {image.mode} to RGB for OCR")
                image = image.convert("RGB")

            # Convert PIL Image to numpy array for PaddleOCR (RGB)
            img_array = np.array(image)
            # print(
            #     f"  🔍 Image array shape: {img_array.shape}, dtype: {img_array.dtype}"
            # )

            # Run OCR
            print("  🔍 Running PaddleOCR...")
            result = self.ocr.ocr(img_array)
            # print(f"  🔍 Raw OCR result type: {type(result)}")
            # print(f"  🔍 Raw OCR result: {result}")

            # Extract text from result
            text_lines = []

            # PaddleOCR returns a list with one element per page/image
            # Each element is a list of detected text regions or None if nothing found
            if result is None:
                print("  ⚠️  OCR returned None (no text detected)")
            elif isinstance(result, list):
                print(f"  🔍 Result is list with length: {len(result)}")
                if len(result) > 0:
                    page_result = result[0]
                    print(f"  🔍 First page result type: {type(page_result)}")

                    # PaddleOCR >=3 returns OCRResult objects; convert to dict if possible
                    page_dict = None
                    if page_result is None:
                        print("  ⚠️  No text detected in image")
                    elif hasattr(page_result, "dict"):
                        try:
                            page_dict = page_result.dict()
                            print(
                                f"  🔍 First page dict keys: {list(page_dict.keys())}"
                            )
                        except Exception as convert_err:
                            print(
                                f"  ⚠️  Failed to convert OCRResult to dict: {convert_err}"
                            )
                    elif isinstance(page_result, dict):
                        page_dict = page_result
                        print(f"  🔍 First page dict keys: {list(page_dict.keys())}")

                    if page_dict is not None:
                        rec_texts = page_dict.get("rec_texts") or []
                        rec_scores = page_dict.get("rec_scores") or []
                        print(
                            f"  🔍 Found {len(rec_texts)} recognized entries in rec_texts"
                        )
                        for idx, text in enumerate(rec_texts):
                            score = (
                                rec_scores[idx] if idx < len(rec_scores) else "unknown"
                            )
                            print(
                                f"    ✅ rec_texts[{idx}]: '{text}' (confidence: {score})"
                            )
                            if text:
                                text_lines.append(text)

                        # Fallback for legacy structures within dicts
                        if not text_lines:
                            legacy_lines = page_dict.get("rec_result") or []
                            if legacy_lines:
                                print(
                                    f"  🔍 Fallback to rec_result with {len(legacy_lines)} entries"
                                )
                                for idx, line in enumerate(legacy_lines):
                                    print(
                                        f"    🔍 Legacy line {idx}: type={type(line)}, value={line}"
                                    )
                                    if (
                                        isinstance(line, (list, tuple))
                                        and len(line) >= 2
                                    ):
                                        text = line[0]
                                        score = line[1]
                                        print(
                                            f"      ✅ Legacy extracted: '{text}' (confidence: {score})"
                                        )
                                        if text:
                                            text_lines.append(text)

                    # Legacy PaddleOCR (<3.0) returns list of lines
                    if not text_lines and isinstance(page_result, list):
                        print(
                            f"  🔍 Processing {len(page_result)} detected text regions (legacy format)"
                        )
                        for idx, line in enumerate(page_result):
                            print(f"    🔍 Line {idx}: type={type(line)}, value={line}")

                            # Each line is [bbox, (text, confidence)]
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                text_info = line[1]
                                if (
                                    isinstance(text_info, (list, tuple))
                                    and len(text_info) >= 1
                                ):
                                    text = text_info[0]
                                    confidence = (
                                        text_info[1] if len(text_info) > 1 else 1.0
                                    )
                                    print(
                                        f"      ✅ Extracted: '{text}' (confidence: {confidence})"
                                    )
                                    if text:
                                        text_lines.append(text)
                                elif isinstance(text_info, str):
                                    print(f"      ✅ Extracted: '{text_info}'")
                                    if text_info:
                                        text_lines.append(text_info)
                            else:
                                print("      ⚠️  Unexpected line format")

            extracted_text = " ".join(text_lines)
            processing_time = time.time() - start_time

            stats = {
                "ocr_processing_time_seconds": round(processing_time, 3),
                "text_length_chars": len(extracted_text),
                "text_lines_found": len(text_lines),
            }

            return extracted_text, stats

        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            import traceback

            traceback.print_exc()
            return "", {
                "ocr_processing_time_seconds": time.time() - start_time,
                "text_length_chars": 0,
                "text_lines_found": 0,
                "error": str(e),
            }

    def process_with_llm(
        self, image: Image.Image
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process image with LLM.

        Args:
            image: PIL Image to process

        Returns:
            Tuple of (summary text, stats dict)
        """
        if not self.enable_llm:
            return None, None

        return process_with_ollama(image, self.ollama_model, power_efficient=False)

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

        # Step 3: Extract text using OCR
        print("📝 Extracting text with OCR...")
        extracted_text, ocr_stats = self.extract_text_ocr(downscaled)
        if self.verbose:
            print(
                f"✅ Extracted {ocr_stats['text_length_chars']} chars from {ocr_stats['text_lines_found']} lines"
            )
            if extracted_text:
                print(f"   🖨️ OCR Text: {extracted_text}")
        else:
            print(
                f"✅ Extracted {ocr_stats['text_length_chars']} chars from {ocr_stats['text_lines_found']} lines"
            )

        # Step 4: Decide on processing method
        summary = None
        llm_stats = None
        processing_method = None

        if extracted_text and len(extracted_text) >= self.ocr_text_threshold:
            # Sufficient text - use OCR result
            print(f"✅ Using OCR text (>= {self.ocr_text_threshold} chars)")
            summary = f"User is viewing/working with: {extracted_text}"
            processing_method = "ocr"
            self.stats["ocr_only"] += 1
        else:
            # Insufficient text - use LLM if enabled
            if self.enable_llm:
                print(f"🤖 Text too short ({len(extracted_text)} chars), using LLM...")
                summary, llm_stats = self.process_with_llm(downscaled)
                processing_method = "llm"
                self.stats["llm_processed"] += 1
            else:
                print(
                    f"⚠️  Text too short ({len(extracted_text)} chars), LLM disabled - using OCR anyway"
                )
                summary = f"User is viewing: {extracted_text if extracted_text else 'Unknown content'}"
                processing_method = "ocr_fallback"
                self.stats["ocr_only"] += 1

        if not summary:
            print("❌ Failed to generate summary")
            return None

        if self.verbose:
            print(f"📝 Summary (full): {summary}")
        else:
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
                extracted_text=extracted_text,
                ocr_stats=ocr_stats,
                llm_stats=llm_stats,
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
                extracted_text=extracted_text,
                ocr_stats=ocr_stats,
                llm_stats=llm_stats,
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
            "summary": summary,
            "extracted_text": extracted_text,
            "image_path": str(image_path) if image_path else None,
            "timestamp": current_time.isoformat(),
            "total_processing_time_seconds": round(total_time, 3),
            "ocr_stats": ocr_stats,
            "llm_stats": llm_stats,
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
        print(f"Images with sufficient OCR: {stats['ocr_only']}")
        print(f"Images needing LLM: {stats['llm_processed']}")
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
