import time
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import GenerateRequest, GenerateResponse
from utils import logger, array_to_base64_png
from ddpm import build_unet2, gen_samples

class ModelManager:
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_id: str, filepath: str):
        logger.info(f"Loading model '{model_id}' from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found at {filepath}")
            
        model = build_unet2()
        model.load_weights(filepath)
        self.models[model_id] = model
        logger.info(f"Model '{model_id}' loaded successfully.")
        
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
    yield
    # Shutdown
    model_manager.models.clear()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
