import streamlit as st
import os
import time
import numpy as np
from PIL import Image

# Force CPU — ZeroGPU only supports PyTorch, and we use TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras

from ddpm import build_unet2, gen_samples, gen_samples_from_image
from utils import center_mnist_style
import utils

from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MNIST Handwriting DDPM", layout="wide")
st.title("MNIST Handwriting Diffusion")
st.write("Generate MNIST digits using a DDPM from pure noise or guided by a sketch.")

# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Models...")
def get_model_manager():
    class ModelManager:
        def __init__(self):
            self.models = {}
            self.classifier = None

        def load_model(self, model_id: str, filepath: str):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Weights file not found at {filepath}")
            model = build_unet2()
            model.load_weights(filepath)
            self.models[model_id] = model

        def load_classifier(self, filepath: str):
            if not os.path.exists(filepath):
                return
            self.classifier = keras.models.load_model(filepath)

        def classify(self, img_array: np.ndarray):
            if self.classifier is None:
                return None, None
            clf_input = (img_array + 1.0) / 2.0
            preds = self.classifier.predict(clf_input, verbose=0)
            digit = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            return digit, confidence

        def get_model(self, model_id: str):
            return self.models[model_id]

    manager = ModelManager()
    weights_path = os.path.join("weights", "ddpm_mnist_cond_best.weights.h5")
    classifier_path = os.path.join("weights", "mnist_classifier.keras")
    try:
        manager.load_model("mnist-ddpm", weights_path)
    except Exception as e:
        st.error(f"Failed to load DDPM model on startup: {e}")

    try:
        manager.load_classifier(classifier_path)
    except Exception as e:
        st.warning(f"Failed to load classifier on startup: {e}")
        
    return manager

model_manager = get_model_manager()

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
mode = st.radio("Mode", ["From Noise", "Sketch-to-Digit"], horizontal=True)

if mode == "From Noise":
    col1, col2 = st.columns([1, 2])
    with col1:
        digit = st.number_input("Target Digit", min_value=0, max_value=9, value=7, step=1)
        guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 3.0, 0.1)
        seed = st.number_input("Seed", value=42, step=1)
        generate_btn = st.button("Generate", type="primary")

    with col2:
        if generate_btn:
            with st.spinner("Generating..."):
                start_time = time.time()
                try:
                    model = model_manager.get_model("mnist-ddpm")
                    # gen_samples takes conditioning as a list
                    imgs = gen_samples(model, n_samples=1, conditioning=[digit], guidance_scale=guidance_scale, seed=seed)
                    img_array = imgs[0].numpy()
                    
                    if img_array.ndim == 3 and img_array.shape[2] == 1:
                        img_array = img_array[:, :, 0]
                    img_array = np.clip((img_array + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                    
                    st.image(img, caption=f"Generated Digit: {digit} | Time: {int((time.time() - start_time) * 1000)}ms", width=200)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

else:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Draw a digit here:")
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#ffffff",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        strength = st.slider("Denoising Strength", 0.0, 1.0, 0.5, 0.05)
        guidance_scale = st.slider("Guidance Scale", 0.0, 10.0, 3.0, 0.1)
        seed = st.number_input("Seed", value=42, step=1)
        
    with col2:
        if canvas_result.image_data is not None:
            # st_canvas returns an RGBA array.
            img_rgba = canvas_result.image_data
            
            # Use Alpha channel if drawing on transparent, or Red channel if drawing white on black
            gray = img_rgba[:, :, 0].astype(np.uint8)
            pil_img = Image.fromarray(gray, mode="L")
            
            # Apply MNIST centering (handles resizing to 28x28 internally)
            pil_img = center_mnist_style(pil_img)

            # Normalize to [-1.0, 1.0] for DDPM
            arr = np.array(pil_img, dtype=np.float32)
            arr = (arr / 127.5) - 1.0

            # Reshape to (1, 28, 28, 1) for the model
            img_array_for_model = arr.reshape(1, 28, 28, 1)
            
            # Classify sketch
            predicted_digit, confidence = model_manager.classify(img_array_for_model)
            if predicted_digit is not None:
                st.info(f"Classifier detected: **{predicted_digit}** (Confidence: {confidence*100:.1f}%)")
                target_digit = st.number_input("Confirm/Override Target Digit", min_value=0, max_value=9, value=int(predicted_digit), step=1)
            else:
                target_digit = st.number_input("Target Digit", min_value=0, max_value=9, value=7, step=1)

            if st.button("Confirm & Diffuse", type="primary"):
                with st.spinner("Generating..."):
                    start_time = time.time()
                    try:
                        model = model_manager.get_model("mnist-ddpm")
                        
                        imgs = gen_samples_from_image(
                            model, 
                            img_array_for_model, 
                            strength=strength, 
                            conditioning=[target_digit], 
                            guidance_scale=guidance_scale, 
                            seed=seed
                        )
                        out_array = imgs[0].numpy()
                        
                        if out_array.ndim == 3 and out_array.shape[2] == 1:
                            out_array = out_array[:, :, 0]
                        out_array = np.clip((out_array + 1.0) * 127.5, 0, 255).astype(np.uint8)
                        img = Image.fromarray(out_array, mode='L')
                        
                        st.image(img, caption=f"Diffused from sketch | Digit: {target_digit} | Time: {int((time.time() - start_time) * 1000)}ms", width=200)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
