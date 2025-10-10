"""
Ollama/LLM processing functionality.
"""

import base64
import io
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


def image_to_base64(image):
    """
    Convert PIL Image to base64 string for Ollama.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def process_with_ollama(image, model_name="gemma3:4b"):
    """
    Process the screenshot with Ollama via LangChain.

    Args:
        image: PIL Image object
        model_name: Name of the Ollama model to use

    Returns:
        Tuple of (Model's response text, stats dict) or (None, None) if processing fails
    """
    try:
        start_time = time.time()

        # Initialize Ollama chat model with keep_alive to keep model loaded
        llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            keep_alive="1h",  # Keep model loaded for 1 hour
        )

        # Convert image to base64
        image_base64 = image_to_base64(image)
        image_base64_size = len(image_base64)

        # Create message with image and prompt
        prompt_text = "Summarize the user's current activity"
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_base64}",
                },
                {"type": "text", "text": prompt_text},
            ],
        )

        # Get response from model
        response = llm.invoke([message])
        processing_time = time.time() - start_time

        # Extract token information if available
        stats = {
            "ollama_processing_time_seconds": round(processing_time, 3),
            "prompt_text": prompt_text,
            "prompt_length_chars": len(prompt_text),
            "image_base64_size_bytes": image_base64_size,
            "image_base64_size_kb": round(image_base64_size / 1024, 2),
            "response_length_chars": len(response.content),
            "response_word_count": len(response.content.split()),
        }

        # Try to extract token usage from response metadata if available
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if "total_duration" in metadata:
                stats["model_total_duration_ns"] = metadata["total_duration"]
            if "load_duration" in metadata:
                stats["model_load_duration_ns"] = metadata["load_duration"]
            if "prompt_eval_count" in metadata:
                stats["prompt_eval_token_count"] = metadata["prompt_eval_count"]
            if "eval_count" in metadata:
                stats["response_token_count"] = metadata["eval_count"]

        return response.content, stats

    except Exception as e:
        print(f"❌ Error processing with Ollama: {e}")
        return None, None


def warmup_model(model_name="gemma3:4b"):
    """
    Warm up the Ollama model by running a tiny test inference.
    This loads the model into memory to avoid cold start delays.

    Args:
        model_name: Name of the Ollama model to warm up

    Returns:
        True if warmup successful, False otherwise
    """
    try:
        print(f"🔥 Warming up Ollama model ({model_name})...")
        start_time = time.time()

        # Initialize Ollama with keep_alive
        llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            keep_alive="1h",
        )

        # Send a minimal prompt to load the model
        message = HumanMessage(content="Hi")
        response = llm.invoke([message])

        warmup_time = time.time() - start_time
        print(f"✅ Model warmed up in {round(warmup_time, 2)}s")
        print(f"💡 Tip: Run 'ollama ps' in terminal to verify model is loaded")
        return True

    except Exception as e:
        print(f"⚠️  Model warmup failed: {e}")
        print("Continuing anyway - first capture may be slower")
        return False
