"""
MLX-based vision-language model processor.

This module uses MLX VLM (Qwen3-VL-2B-Instruct-MLX-8bit) for processing
image-heavy content when OCR text is insufficient.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def load_prompt(prompt_path: Path) -> str:
    """
    Load a prompt from a file.
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Prompt text content
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8").strip()


def parse_vlm_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from VLM output.
    
    The VLM may return JSON wrapped in markdown code blocks or with
    extra text before/after. This function extracts the JSON object.
    
    Args:
        text: Raw VLM output text
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not text:
        return None
    
    # Try direct JSON parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object directly in text (between { and })
    # Find the first { and last } to extract potential JSON
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        potential_json = text[start_idx : end_idx + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass
    
    return None


def validate_structured_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a structured VLM response.
    
    Ensures all expected fields exist with appropriate defaults.
    
    Args:
        data: Parsed JSON response from VLM
        
    Returns:
        Normalized response with all required fields
    """
    return {
        "title": data.get("title", "Untitled"),
        "summary": data.get("summary", ""),
        "bullets": data.get("bullets", []),
        "tags": data.get("tags", []),
        "entities": data.get("entities", []),
    }


class MLXProcessor:
    """MLX VLM processor with model lifecycle management."""
    
    def __init__(
        self,
        model_name: str = "lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit",
        max_image_side: int = 1024,
        max_new_tokens: int = 128,
        verbose: bool = False,
    ):
        """
        Initialize the MLX processor.
        
        Args:
            model_name: HuggingFace model name or local path
            max_image_side: Maximum image dimension (images are downscaled if larger)
            max_new_tokens: Maximum tokens to generate
            verbose: If True, print debug messages
        """
        self.model_name = model_name
        self.max_image_side = max_image_side
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        
        # Model state
        self.model = None
        self.processor = None
        self.config = None
        self.is_loaded = False
    
    def _log(self, message: str):
        """Log message if verbose is enabled."""
        if self.verbose:
            print(f"[MLXProcessor] {message}")
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image if needed to fit within max_image_side.
        
        Args:
            img: PIL Image
            
        Returns:
            Resized PIL Image
        """
        width, height = img.size
        longest = max(width, height)
        
        if longest <= self.max_image_side:
            return img
        
        scale = self.max_image_side / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def load_model(self):
        """Load the MLX model into memory."""
        if self.is_loaded:
            self._log("Model already loaded")
            return
        
        self._log(f"Loading model: {self.model_name}")
        start_time = time.time()
        
        try:
            self.model, self.processor = load(self.model_name)
            
            # Get config
            try:
                self.config = self.model.config
            except AttributeError:
                self.config = load_config(self.model_name)
            
            self.is_loaded = True
            load_time = time.time() - start_time
            self._log(f"Model loaded in {load_time:.2f}s")
            
        except Exception as e:
            self._log(f"Error loading model: {e}")
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if not self.is_loaded:
            return
        
        self._log("Unloading model...")
        
        # Release references
        self.model = None
        self.processor = None
        self.config = None
        self.is_loaded = False
        
        # Force garbage collection and clear MLX cache
        import gc
        gc.collect()
        
        # Clear MLX memory cache
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
        
        self._log("Model unloaded")
    
    def process_image(
        self, image: Image.Image, prompt: str = "Summarize the user's current activity"
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process an image with the MLX VLM.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for the model
            
        Returns:
            Tuple of (generated text, stats dict) or (None, None) if failed
        """
        if not self.is_loaded:
            self._log("Model not loaded, loading now...")
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Prepare image
            img = ImageOps.exif_transpose(image)
            img = img.convert("RGB")
            img = self._resize_image(img)
            
            image_prep_time = time.time() - start_time
            
            # Format prompt
            formatted_prompt = apply_chat_template(
                self.processor,
                self.config,
                prompt,
                num_images=1,
            )
            
            # Run generation
            gen_start = time.time()
            
            # Reset peak memory tracking
            mx.reset_peak_memory()
            
            output = generate(
                self.model,
                self.processor,
                formatted_prompt,
                [img],
                max_tokens=self.max_new_tokens,
                verbose=False,
            )
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            
            # Get memory stats
            gpu_peak = mx.get_peak_memory()
            gpu_active = mx.get_active_memory()
            
            # Compile stats
            stats = {
                "mlx_processing_time_seconds": round(total_time, 3),
                "image_prep_time_seconds": round(image_prep_time, 3),
                "generation_time_seconds": round(gen_time, 3),
                "image_size": img.size,
                "prompt": prompt,
                "response_length_chars": len(output.text),
                "response_word_count": len(output.text.split()),
                "gpu_peak_bytes": gpu_peak,
                "gpu_active_bytes": gpu_active,
                "model_name": self.model_name,
            }
            
            return output.text, stats
            
        except Exception as e:
            self._log(f"Error processing image: {e}")
            error_time = time.time() - start_time
            return None, {
                "mlx_processing_time_seconds": round(error_time, 3),
                "error": str(e),
            }
    
    def process_image_from_path(
        self, image_path: str, prompt: str = "Summarize the user's current activity"
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process an image from file path.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for the model
            
        Returns:
            Tuple of (generated text, stats dict) or (None, None) if failed
        """
        try:
            img = Image.open(image_path)
            return self.process_image(img, prompt)
        except Exception as e:
            self._log(f"Error loading image from {image_path}: {e}")
            return None, {"error": str(e)}
    
    def process_image_structured(
        self, image: Image.Image, prompt: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process an image and return structured JSON output.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt that instructs the model to return JSON
            
        Returns:
            Tuple of (structured_data dict, stats dict) or (None, stats) if failed
        """
        raw_text, stats = self.process_image(image, prompt)
        
        if raw_text is None:
            return None, stats
        
        # Add raw response to stats for debugging
        if stats:
            stats["raw_response"] = raw_text
        
        # Parse JSON from response
        parsed = parse_vlm_json_response(raw_text)
        
        if parsed is None:
            self._log(f"Failed to parse JSON from response: {raw_text[:200]}...")
            if stats:
                stats["parse_error"] = "Failed to extract valid JSON from response"
            return None, stats
        
        # Validate and normalize the response
        structured = validate_structured_response(parsed)
        
        return structured, stats
    
    def should_keep_loaded(self, queue_has_items: bool, on_battery: bool) -> bool:
        """
        Determine if the model should stay loaded in memory.
        
        Args:
            queue_has_items: True if processing queue has pending items
            on_battery: True if system is on battery power
            
        Returns:
            True if model should stay loaded, False otherwise
        """
        # Keep loaded if there are items to process AND not on battery
        return queue_has_items and not on_battery
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
        return False


# Singleton instance for shared use
_global_processor: Optional[MLXProcessor] = None


def get_global_processor(
    model_name: str = "lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit",
    max_image_side: int = 1024,
    max_new_tokens: int = 128,
    verbose: bool = False,
) -> MLXProcessor:
    """
    Get or create the global MLX processor instance.
    
    This ensures only one model is loaded in memory at a time.
    
    Args:
        model_name: HuggingFace model name or local path
        max_image_side: Maximum image dimension
        max_new_tokens: Maximum tokens to generate
        verbose: If True, print debug messages
        
    Returns:
        MLXProcessor instance
    """
    global _global_processor
    
    if _global_processor is None:
        _global_processor = MLXProcessor(
            model_name=model_name,
            max_image_side=max_image_side,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
    
    return _global_processor


def process_with_mlx(
    image: Image.Image,
    model_name: str = "lmstudio-community/Qwen3-VL-2B-Instruct-MLX-8bit",
    prompt: str = "Summarize the user's current activity",
    max_new_tokens: int = 128,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Convenience function to process an image with MLX.
    
    Uses the global processor instance.
    
    Args:
        image: PIL Image to process
        model_name: HuggingFace model name or local path
        prompt: Text prompt for the model
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (generated text, stats dict) or (None, None) if failed
    """
    processor = get_global_processor(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    
    return processor.process_image(image, prompt)




