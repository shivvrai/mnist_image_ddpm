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
