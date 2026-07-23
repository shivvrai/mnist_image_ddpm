import time
import os
import json
import numpy as np

# ZeroGPU support — graceful fallback for local development
# IMPORTANT: import spaces MUST happen before import torch!
try:
    import spaces
except ImportError:
    # Provide a no-op decorator so the code works locally without ZeroGPU
    class _FakeSpaces:
        @staticmethod
        def GPU(fn=None, duration=120):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _FakeSpaces()

import torch
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import logger, array_to_base64_png, base64_png_to_array
from ddpm import UNet, gen_samples, gen_samples_from_image
from classifier import MNISTClassifier

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
unet_model = None
classifier_model = None

def load_models():
    global unet_model, classifier_model
    
    logger.info(f"Using device: {DEVICE}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    unet_path = os.path.join(base_dir, "weights", "ddpm_unet.pt")
    clf_path = os.path.join(base_dir, "weights", "mnist_classifier.pt")

    # Load UNet
    if os.path.exists(unet_path):
        logger.info(f"Loading UNet from {unet_path}...")
        unet_model = UNet()
        unet_model.load_state_dict(torch.load(unet_path, map_location="cpu", weights_only=True))
        unet_model.eval()
        unet_model.to(DEVICE)      # ZeroGPU: stays on CPU until @spaces.GPU runs
        logger.info("UNet loaded successfully.")
    else:
        logger.error(f"UNet weights not found at {unet_path}")

    # Load Classifier
    if os.path.exists(clf_path):
        logger.info(f"Loading Classifier from {clf_path}...")
        classifier_model = MNISTClassifier()
        classifier_model.load_state_dict(torch.load(clf_path, map_location="cpu", weights_only=True))
        classifier_model.eval()
        classifier_model.to(DEVICE)
        logger.info("Classifier loaded successfully.")
    else:
        logger.warning(f"Classifier weights not found at {clf_path}. Sketch prediction disabled.")

load_models()

# ---------------------------------------------------------------------------
# Simple in-memory cache
# ---------------------------------------------------------------------------
image_cache = {}

# ---------------------------------------------------------------------------
# API functions — @spaces.GPU gives us a free GPU for each call
# ---------------------------------------------------------------------------

def api_health():
    models = []
    if unet_model is not None:
        models.append("mnist-ddpm")
    return json.dumps({"status": "ok", "loaded_models": models, "device": DEVICE.type})


@spaces.GPU(duration=30)
def api_generate(request_json: str):
    try:
        req = json.loads(request_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON"})

    model_id = req.get("model_id", "mnist-ddpm")
    digit = int(req.get("digit", 7))
    guidance_scale = float(req.get("guidance_scale", 3.0))
    seed = int(req.get("seed", 42))

    cache_key = f"{model_id}_{digit}_{guidance_scale}_{seed}"
    if cache_key in image_cache:
        logger.info(f"Cache hit for {cache_key}")
        return image_cache[cache_key]

    if unet_model is None:
        return json.dumps({"error": "Model not loaded."})

    logger.info(f"Generating digit {digit} with seed {seed} and GS {guidance_scale}...")
    start_time = time.time()

    # Generate — model is on GPU inside @spaces.GPU context
    imgs = gen_samples(unet_model, n_samples=1, conditioning=[digit],
                       guidance_scale=guidance_scale, seed=seed)
    # imgs shape: (1, 1, 28, 28) on GPU, range [0, 1]
    img_np = imgs[0].cpu().permute(1, 2, 0).numpy()   # -> (28, 28, 1)
    generation_time_ms = int((time.time() - start_time) * 1000)

    b64_image = array_to_base64_png(img_np)
    response = json.dumps({
        "image_b64": b64_image,
        "digit": digit,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "generation_time_ms": generation_time_ms,
        "model_name": model_id,
    })
    image_cache[cache_key] = response
    return response


@spaces.GPU(duration=15)
def api_classify_sketch(request_json: str):
    try:
        req = json.loads(request_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON"})

    if classifier_model is None:
        return json.dumps({"predicted_digit": None, "confidence": None})

    sketch_b64 = req.get("sketch_b64", "")
    sketch_np = base64_png_to_array(sketch_b64)            # (1, 28, 28, 1) range [-1, 1]
    # Classifier expects [0, 1]
    clf_input = (sketch_np + 1.0) / 2.0
    # NHWC -> NCHW
    clf_tensor = torch.from_numpy(clf_input).permute(0, 3, 1, 2).to(DEVICE)

    with torch.no_grad():
        preds = classifier_model(clf_tensor)    # (1, 10) softmax output
    predicted_digit = int(preds[0].argmax().item())
    confidence = float(preds[0].max().item())

    return json.dumps({
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
    })


@spaces.GPU(duration=30)
def api_generate_from_sketch(request_json: str):
    try:
        req = json.loads(request_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON"})

    if unet_model is None:
        return json.dumps({"error": "Model not loaded."})

    model_id = req.get("model_id", "mnist-ddpm")
    sketch_b64 = req.get("sketch_b64", "")
    digit = int(req.get("digit", 7))
    strength = float(req.get("strength", 0.5))
    guidance_scale = float(req.get("guidance_scale", 3.0))
    seed = int(req.get("seed", 42))

    logger.info(f"Sketch-to-digit: conditioning={digit}, strength={strength}, seed={seed}, GS={guidance_scale}")
    start_time = time.time()

    # Prepare sketch tensor: NHWC numpy -> NCHW torch
    sketch_np = base64_png_to_array(sketch_b64)            # (1, 28, 28, 1) range [-1, 1]
    x_start = torch.from_numpy(sketch_np).permute(0, 3, 1, 2).to(DEVICE)

    imgs = gen_samples_from_image(unet_model, x_start, strength=strength,
                                  conditioning=[digit], guidance_scale=guidance_scale, seed=seed)
    img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
    generation_time_ms = int((time.time() - start_time) * 1000)

    b64_image = array_to_base64_png(img_np)
    return json.dumps({
        "image_b64": b64_image,
        "digit": digit,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "generation_time_ms": generation_time_ms,
        "model_name": model_id,
        "predicted_digit": None,
        "confidence": None,
    })

# ---------------------------------------------------------------------------
# Gradio Blocks — pure Gradio app (no FastAPI wrapper)
# The Gradio API protocol automatically creates POST endpoints at
# /api/<api_name> that accept {"data": [...]} and return {"data": [...]},
# which is exactly the format our React frontend already uses.
# ---------------------------------------------------------------------------
with gr.Blocks(title="MNIST Diffusion Backend") as demo:
    gr.Markdown("## MNIST Handwriting Diffusion - Backend API\nThis space serves the backend API. The frontend is deployed on Vercel.")

    with gr.Row(visible=False):
        health_out = gr.Textbox()
        gr.Button().click(fn=api_health, inputs=[], outputs=health_out, api_name="health")

    with gr.Row(visible=False):
        gen_in = gr.Textbox()
        gen_out = gr.Textbox()
        gr.Button().click(fn=api_generate, inputs=gen_in, outputs=gen_out, api_name="generate")

    with gr.Row(visible=False):
        cls_in = gr.Textbox()
        cls_out = gr.Textbox()
        gr.Button().click(fn=api_classify_sketch, inputs=cls_in, outputs=cls_out, api_name="classify-sketch")

    with gr.Row(visible=False):
        sketch_in = gr.Textbox()
        sketch_out = gr.Textbox()
        gr.Button().click(fn=api_generate_from_sketch, inputs=sketch_in, outputs=sketch_out, api_name="generate-from-sketch")

demo.launch()
