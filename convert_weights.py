"""
One-time script: Convert TF/Keras weights -> PyTorch state dicts.

Run this LOCALLY (where TensorFlow is installed):
    python convert_weights.py

Produces:
    backend/weights/ddpm_unet.pt
    backend/weights/mnist_classifier.pt
"""

import sys, os
import numpy as np
import h5py
import torch

# ── Make sure backend modules are importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from ddpm import UNet
from classifier import MNISTClassifier


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _conv_w(h5, key):
    """Load a TF Conv2D kernel (H,W,Cin,Cout) -> PyTorch (Cout,Cin,H,W)."""
    w = np.array(h5[f"layers/{key}/vars/0"])
    return torch.from_numpy(w.transpose(3, 2, 0, 1).copy())

def _conv_b(h5, key):
    """Load a Conv2D bias."""
    return torch.from_numpy(np.array(h5[f"layers/{key}/vars/1"]))

def _dense_w(h5, key):
    """Load a TF Dense weight (in, out) -> PyTorch (out, in)."""
    w = np.array(h5[f"layers/{key}/vars/0"])
    return torch.from_numpy(w.T.copy())

def _dense_b(h5, key):
    """Load a Dense bias."""
    return torch.from_numpy(np.array(h5[f"layers/{key}/vars/1"]))


# ─────────────────────────────────────────────────────────────────────────────
#  Convert UNet
# ─────────────────────────────────────────────────────────────────────────────

def convert_unet(h5_path: str, out_path: str):
    print(f"Converting UNet: {h5_path} -> {out_path}")
    f = h5py.File(h5_path, "r")
    model = UNet()

    sd = {}

    # Embedding
    sd["class_embed.weight"] = torch.from_numpy(np.array(f["layers/embedding/vars/0"]))

    # Conditioning MLP: dense -> cond_mlp.0, dense_1 -> cond_mlp.2
    sd["cond_mlp.0.weight"] = _dense_w(f, "dense")
    sd["cond_mlp.0.bias"]   = _dense_b(f, "dense")
    sd["cond_mlp.2.weight"] = _dense_w(f, "dense_1")
    sd["cond_mlp.2.bias"]   = _dense_b(f, "dense_1")

    # Init conv (conv2d -> init_conv)
    sd["init_conv.weight"] = _conv_w(f, "conv2d")
    sd["init_conv.bias"]   = _conv_b(f, "conv2d")

    # Encoder level 1 — ResBlocks with 64 channels
    # ResBlock pattern: conv1 (odd idx), conv2 (even idx), t_proj (dense_N)
    res_blocks = [
        ("enc1_res1", "conv2d_1", "conv2d_2", "dense_2"),
        ("enc1_res2", "conv2d_3", "conv2d_4", "dense_3"),
    ]

    # Down1 (conv2d_5)
    sd["down1.weight"] = _conv_w(f, "conv2d_5")
    sd["down1.bias"]   = _conv_b(f, "conv2d_5")

    # Encoder level 2 — ResBlocks with 128 channels
    res_blocks += [
        ("enc2_res1", "conv2d_6", "conv2d_7", "dense_4"),
        ("enc2_res2", "conv2d_8", "conv2d_9", "dense_5"),
    ]

    # Down2 (conv2d_10)
    sd["down2.weight"] = _conv_w(f, "conv2d_10")
    sd["down2.bias"]   = _conv_b(f, "conv2d_10")

    # Bottleneck — ResBlocks with 256 channels
    res_blocks += [
        ("bottle_res1", "conv2d_11", "conv2d_12", "dense_6"),
        ("bottle_res2", "conv2d_13", "conv2d_14", "dense_7"),
    ]

    # Decoder level 1 — ResBlocks with 384 channels (256+128 after concat)
    res_blocks += [
        ("dec1_res1", "conv2d_15", "conv2d_16", "dense_8"),
        ("dec1_res2", "conv2d_17", "conv2d_18", "dense_9"),
    ]

    # Decoder level 2 — ResBlocks with 448 channels (384+64 after concat)
    res_blocks += [
        ("dec2_res1", "conv2d_19", "conv2d_20", "dense_10"),
        ("dec2_res2", "conv2d_21", "conv2d_22", "dense_11"),
    ]

    # Map all ResBlocks
    for pt_name, cv1_key, cv2_key, dense_key in res_blocks:
        sd[f"{pt_name}.conv1.weight"] = _conv_w(f, cv1_key)
        sd[f"{pt_name}.conv1.bias"]   = _conv_b(f, cv1_key)
        sd[f"{pt_name}.conv2.weight"] = _conv_w(f, cv2_key)
        sd[f"{pt_name}.conv2.bias"]   = _conv_b(f, cv2_key)
        sd[f"{pt_name}.t_proj.weight"] = _dense_w(f, dense_key)
        sd[f"{pt_name}.t_proj.bias"]   = _dense_b(f, dense_key)

    # Final conv (conv2d_23)
    sd["final_conv.weight"] = _conv_w(f, "conv2d_23")
    sd["final_conv.bias"]   = _conv_b(f, "conv2d_23")

    f.close()

    # Validate: every key in the model must be covered
    model_keys = set(model.state_dict().keys())
    converted_keys = set(sd.keys())
    missing = model_keys - converted_keys
    extra = converted_keys - model_keys
    if missing:
        print(f"  [WARN] MISSING keys: {missing}")
    if extra:
        print(f"  [WARN] EXTRA keys: {extra}")

    # Shape check
    for key in model_keys:
        expected = model.state_dict()[key].shape
        actual = sd[key].shape
        if expected != actual:
            print(f"  [WARN] Shape mismatch for '{key}': expected {expected}, got {actual}")
            return False

    model.load_state_dict(sd)
    torch.save(model.state_dict(), out_path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [OK] UNet converted successfully — {param_count:,} parameters")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Convert Classifier
# ─────────────────────────────────────────────────────────────────────────────

def convert_classifier(keras_path: str, out_path: str):
    print(f"Converting Classifier: {keras_path} -> {out_path}")

    # Use TensorFlow to load the .keras model and extract weights
    import tensorflow as tf
    tf_model = tf.keras.models.load_model(keras_path)

    pt_model = MNISTClassifier()
    sd = {}

    # Map Keras sequential layers (skipping Dropout, MaxPool, Flatten — no params)
    # Layer order in Keras: conv2d, conv2d_1, maxpool, dropout, conv2d_2, conv2d_3, maxpool, dropout, flatten, dense, dropout, dense_1
    # Layers with weights:  conv2d(0), conv2d_1(1), conv2d_2(4), conv2d_3(5), dense(9), dense_1(11)

    keras_conv_layers = [l for l in tf_model.layers if "conv2d" in l.name]
    keras_dense_layers = [l for l in tf_model.layers if "dense" in l.name]

    # features: Conv2d(0), ReLU(1), Conv2d(2), ReLU(3), MaxPool(4), Conv2d(5), ReLU(6), Conv2d(7), ReLU(8), MaxPool(9)
    pt_conv_keys = [
        "features.0",   # Conv2d(1→32)
        "features.2",   # Conv2d(32→32)
        "features.5",   # Conv2d(32→64)
        "features.7",   # Conv2d(64→64)
    ]

    for pt_key, keras_layer in zip(pt_conv_keys, keras_conv_layers):
        w, b = keras_layer.get_weights()
        sd[f"{pt_key}.weight"] = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
        sd[f"{pt_key}.bias"] = torch.from_numpy(b.copy())

    # head: Linear(0), ReLU(1), Linear(2)
    pt_dense_keys = ["head.0", "head.2"]
    for pt_key, keras_layer in zip(pt_dense_keys, keras_dense_layers):
        w, b = keras_layer.get_weights()
        sd[f"{pt_key}.weight"] = torch.from_numpy(w.T.copy())
        sd[f"{pt_key}.bias"] = torch.from_numpy(b.copy())

    # Validate
    model_keys = set(pt_model.state_dict().keys())
    converted_keys = set(sd.keys())
    missing = model_keys - converted_keys
    if missing:
        print(f"  [WARN] MISSING keys: {missing}")
        return False

    for key in model_keys:
        expected = pt_model.state_dict()[key].shape
        actual = sd[key].shape
        if expected != actual:
            print(f"  [WARN] Shape mismatch for '{key}': expected {expected}, got {actual}")
            return False

    pt_model.load_state_dict(sd)
    torch.save(pt_model.state_dict(), out_path)
    param_count = sum(p.numel() for p in pt_model.parameters())
    print(f"  [OK] Classifier converted successfully — {param_count:,} parameters")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unet_ok = convert_unet(
        h5_path="backend/weights/ddpm_mnist_cond_best.weights.h5",
        out_path="backend/weights/ddpm_unet.pt",
    )
    print()
    clf_ok = convert_classifier(
        keras_path="backend/weights/mnist_classifier.keras",
        out_path="backend/weights/mnist_classifier.pt",
    )
    print()
    if unet_ok and clf_ok:
        print("🎉 All conversions successful!")
    else:
        print("[FAIL] Some conversions failed. Check errors above.")
