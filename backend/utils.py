import base64
from io import BytesIO
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ddpm-app")

def array_to_base64_png(img_array: np.ndarray) -> str:
    # img_array is assumed to be shape (H, W, 1) and range [0, 1]
    if img_array.ndim == 3 and img_array.shape[2] == 1:
        img_array = img_array[:, :, 0]
    
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array, mode='L')
    
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def center_mnist_style(pil_img: Image.Image) -> Image.Image:
    """Preprocess image to match MNIST dataset standards:
    1. Crop to bounding box
    2. Resize so max dimension is 20 pixels
    3. Center it on a 28x28 grid by center of mass
    """
    bbox = pil_img.getbbox()
    if bbox is None:
        return pil_img.resize((28, 28), Image.LANCZOS)
        
    # Crop to bounding box
    cropped = pil_img.crop(bbox)
    
    # Resize so max dimension is 20
    w, h = cropped.size
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)
    
    # Find center of mass
    arr = np.array(resized)
    total = arr.sum()
    if total == 0:
        cy, cx = new_h / 2.0, new_w / 2.0
    else:
        y_coords, x_coords = np.indices(arr.shape)
        cy = (y_coords * arr).sum() / total
        cx = (x_coords * arr).sum() / total
        
    # Calculate offset to place center of mass at (14, 14)
    start_y = int(round(14.0 - cy))
    start_x = int(round(14.0 - cx))
    
    final_img = Image.new("L", (28, 28), color=0)
    final_img.paste(resized, (start_x, start_y))
    return final_img

def base64_png_to_array(b64_str: str) -> np.ndarray:
    """Convert a base64 PNG data URL from an HTML canvas to a DDPM-ready numpy array.

    Pipeline:
    1. Strip data URI prefix
    2. Decode to PIL Image -> grayscale
    3. Apply MNIST-style centering and resizing (center of mass in 28x28)
    4. Normalize to [-1.0, 1.0]
    5. Reshape to (1, 28, 28, 1)
    """
    # Strip data URI prefix if present
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(BytesIO(img_bytes)).convert("L")

    # Apply MNIST centering (handles resizing to 28x28 internally)
    pil_img = center_mnist_style(pil_img)

    # Normalize to [-1.0, 1.0]
    arr = np.array(pil_img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0

    # Reshape to (1, 28, 28, 1) for the model
    return arr.reshape(1, 28, 28, 1)
