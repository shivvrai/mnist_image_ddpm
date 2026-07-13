import time
import os
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from tensorflow import keras

from schemas import GenerateRequest, GenerateFromSketchRequest, GenerateResponse, ClassifySketchRequest, ClassifySketchResponse
from utils import logger, array_to_base64_png, base64_png_to_array
from ddpm import build_unet2, gen_samples, gen_samples_from_image

class ModelManager:
    def __init__(self):
        self.models = {}
        self.classifier = None
        
    def load_model(self, model_id: str, filepath: str):
        logger.info(f"Loading model '{model_id}' from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found at {filepath}")
            
        model = build_unet2()
        model.load_weights(filepath)
        self.models[model_id] = model
        logger.info(f"Model '{model_id}' loaded successfully.")

    def load_classifier(self, filepath: str):
        logger.info(f"Loading classifier from {filepath}...")
        if not os.path.exists(filepath):
            logger.warning(f"Classifier not found at {filepath}. Sketch auto-prediction disabled.")
            return
        self.classifier = keras.models.load_model(filepath)
        logger.info("Classifier loaded successfully.")

    def classify(self, img_array: np.ndarray) -> tuple[int, float]:
        """Classify a sketch image. Returns (predicted_digit, confidence).
        
        Args:
            img_array: shape (1, 28, 28, 1), range [-1, 1] (DDPM format)
        Returns:
            (digit, confidence) tuple
        """
        if self.classifier is None:
            return None, None
        # Classifier expects [0, 1] range
        clf_input = (img_array + 1.0) / 2.0
        preds = self.classifier.predict(clf_input, verbose=0)
        digit = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return digit, confidence
        
    def get_model(self, model_id: str):
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not loaded.")
        return self.models[model_id]

# Global manager and cache
model_manager = ModelManager()

# Simple dict cache: key -> GenerateResponse
image_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the MNIST DDPM model
    weights_path = os.path.join("weights", "ddpm_mnist_cond_best.weights.h5")
    try:
        model_manager.load_model("mnist-ddpm", weights_path)
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

    # Load the sketch classifier
    classifier_path = os.path.join("weights", "mnist_classifier.keras")
    try:
        model_manager.load_classifier(classifier_path)
    except Exception as e:
        logger.error(f"Failed to load classifier on startup: {e}")

    yield
    # Shutdown
    model_manager.models.clear()
    model_manager.classifier = None

app = FastAPI(title="Handwriting DDPM API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "loaded_models": list(model_manager.models.keys())
    }

# Run generation in a threadpool so it doesn't block the async event loop
def do_generate(model_id: str, digit: int, guidance_scale: float, seed: int):
    model = model_manager.get_model(model_id)
    imgs = gen_samples(model, n_samples=1, conditioning=[digit], guidance_scale=guidance_scale, seed=seed)
    return imgs[0].numpy()

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # Check cache
    cache_key = f"{req.model_id}_{req.digit}_{req.guidance_scale}_{req.seed}"
    if cache_key in image_cache:
        logger.info(f"Cache hit for {cache_key}")
        return image_cache[cache_key]

    logger.info(f"Generating digit {req.digit} with seed {req.seed} and GS {req.guidance_scale}...")
    start_time = time.time()
    
    try:
        # Run synchronous generation in async executor
        img_array = await asyncio.to_thread(
            do_generate, 
            req.model_id, 
            req.digit, 
            req.guidance_scale, 
            req.seed
        )
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' is not available.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during generation.")
        
    generation_time_ms = int((time.time() - start_time) * 1000)
    
    # Convert to base64
    b64_image = array_to_base64_png(img_array)
    
    response = GenerateResponse(
        image_b64=b64_image,
        digit=req.digit,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        generation_time_ms=generation_time_ms,
        model_name=req.model_id
    )
    
    # Save to cache
    image_cache[cache_key] = response
    
    return response

# Sketch-to-digit generation (Image-to-Image)
def do_generate_from_sketch(model_id: str, sketch_b64: str, strength: float, digit: int, guidance_scale: float, seed: int):
    model = model_manager.get_model(model_id)
    x_start = base64_png_to_array(sketch_b64)
    imgs = gen_samples_from_image(model, x_start, strength=strength, conditioning=[digit], guidance_scale=guidance_scale, seed=seed)
    return imgs[0].numpy()

@app.post("/classify-sketch", response_model=ClassifySketchResponse)
async def classify_sketch(req: ClassifySketchRequest):
    sketch_array = base64_png_to_array(req.sketch_b64)
    predicted_digit, confidence = model_manager.classify(sketch_array)
    return ClassifySketchResponse(
        predicted_digit=predicted_digit,
        confidence=round(confidence, 4) if confidence is not None else None
    )

@app.post("/generate-from-sketch", response_model=GenerateResponse)
async def generate_from_sketch(req: GenerateFromSketchRequest):
    start_time = time.time()

    logger.info(
        f"Sketch-to-digit: conditioning={req.digit}, strength={req.strength}, "
        f"seed={req.seed}, GS={req.guidance_scale}"
    )

    try:
        img_array = await asyncio.to_thread(
            do_generate_from_sketch,
            req.model_id,
            req.sketch_b64,
            req.strength,
            req.digit,
            req.guidance_scale,
            req.seed
        )
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_id}' is not available.")
    except Exception as e:
        logger.error(f"Sketch generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during sketch generation.")

    generation_time_ms = int((time.time() - start_time) * 1000)
    b64_image = array_to_base64_png(img_array)

    return GenerateResponse(
        image_b64=b64_image,
        digit=req.digit,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        generation_time_ms=generation_time_ms,
        model_name=req.model_id,
        predicted_digit=None,
        confidence=None
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
