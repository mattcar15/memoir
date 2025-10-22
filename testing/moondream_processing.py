from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import cv2


def preprocess_image(path, size=(600, 400)):
    """Resize, lightly sharpen, normalize contrast."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)

    # Slight unsharp mask
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(img_cv, (0, 0), 4)
    sharpened = cv2.addWeighted(img_cv, 1.3, blurred, -0.3, 0)
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(sharpened)
    img = ImageOps.autocontrast(img)
    return img


if __name__ == "__main__":
    input_path = (
        Path(__file__).parent
        / "test_images"
        / "20251021_123325_226_Spotify_Taylor_Swift_-_The_Fate_of_Ophelia.png"
    )
    output_path = Path(__file__).parent / "processed_preview.jpg"

    processed = preprocess_image(input_path)
    processed.save(output_path, quality=95)
    print(f"✅ Saved preprocessed image to {output_path.resolve()}")
