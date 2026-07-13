import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SZ2 = 28
CH2 = 1
CLS = 10
NULL_CLS = CLS
TIMESTEPS2 = 300

def cosine_beta_schedule2(timesteps, s=0.008):
    x = tf.cast(tf.range(timesteps + 1), tf.float32)
    ac = tf.cos(((x / (timesteps + 1)) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return tf.clip_by_value(betas, 1e-8, 0.999)

betas2 = cosine_beta_schedule2(TIMESTEPS2).numpy()
alphas2 = 1.0 - betas2
alpha_hats2 = np.cumprod(alphas2)

def timestep_embedding2(t, dim=128):
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(half, dtype=tf.float32) / float(half))
    t = tf.cast(t, tf.float32)
    t = tf.reshape(t, (-1, 1))
    x = t * tf.reshape(freqs, (1, -1))
    return tf.concat([tf.sin(x), tf.cos(x)], axis=1)

def ResBlock2(x, cond_emb):
    c = x.shape[-1]
    h = layers.Activation("silu")(x)
    h = layers.Conv2D(c, 3, padding="same")(h)
    t_proj = layers.Reshape((1, 1, c))(layers.Dense(c)(cond_emb))
    h = layers.Add()([h, t_proj])
    h = layers.Activation("silu")(h)
    h = layers.Conv2D(c, 3, padding="same")(h)
    return layers.Add()([x, h])

def build_unet2(base_filters=64, cond_dim=128):
    img = keras.Input((IMG_SZ2, IMG_SZ2, CH2))
    t = keras.Input((), dtype=tf.int32)
    y = keras.Input((), dtype=tf.int32)

    t_emb = layers.Lambda(lambda tt: timestep_embedding2(tt, cond_dim), output_shape=(cond_dim,))(t)
    y_emb = layers.Embedding(CLS + 1, cond_dim)(y)
    cond_emb = layers.Add()([t_emb, y_emb])
    cond_emb = layers.Dense(cond_dim, activation="silu")(cond_emb)
    cond_emb = layers.Dense(cond_dim, activation="silu")(cond_emb)

    x = layers.Conv2D(base_filters, 3, padding="same")(img)
    x = ResBlock2(x, cond_emb)
    s1 = ResBlock2(x, cond_emb)

    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same")(s1)
    x = ResBlock2(x, cond_emb)
    s2 = ResBlock2(x, cond_emb)

    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same")(s2)
    x = ResBlock2(x, cond_emb)
    x = ResBlock2(x, cond_emb)

    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, s2])
    x = ResBlock2(x, cond_emb)
    x = ResBlock2(x, cond_emb)

    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, s1])
    x = ResBlock2(x, cond_emb)
    x = ResBlock2(x, cond_emb)

    out = layers.Conv2D(CH2, 3, padding="same")(x)
    return keras.Model([img, t, y], out)

def p_sample_step(x, t, eps, beta, alpha, alpha_cum):
    coef1 = 1.0 / tf.sqrt(alpha)
    coef2 = (1 - alpha) / tf.sqrt(1 - alpha_cum)
    mean = coef1 * (x - coef2 * eps)
    sigma = tf.sqrt(beta)
    return mean + sigma * tf.random.normal(tf.shape(x))

def gen_samples(gen_model, n_samples, conditioning, guidance_scale, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
        
    x = tf.random.normal((n_samples, IMG_SZ2, IMG_SZ2, CH2))
    cond = tf.convert_to_tensor(conditioning, dtype=tf.int32)
    for step in reversed(range(TIMESTEPS2)):
        t_batch = tf.fill((n_samples,), step)
        eps_u = gen_model([x, t_batch, tf.fill((n_samples,), NULL_CLS)], training=False)
        eps_c = gen_model([x, t_batch, cond], training=False)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = p_sample_step(x, t_batch, eps, betas2[step], alphas2[step], alpha_hats2[step])
    return (x + 1.0) * 0.5

def gen_samples_from_image(gen_model, x_start, strength, conditioning, guidance_scale, seed=None):
    """Image-to-image generation: add noise to a sketch, then denoise it."""
    if seed is not None:
        tf.random.set_seed(seed)

    n_samples = tf.shape(x_start)[0]
    x_start = tf.cast(x_start, tf.float32)

    t_start = int(strength * TIMESTEPS2)

    if t_start == 0:
        return (x_start + 1.0) * 0.5

    noise = tf.random.normal(tf.shape(x_start))
    alpha_hat_t = alpha_hats2[t_start - 1]
    x = tf.sqrt(alpha_hat_t) * x_start + tf.sqrt(1 - alpha_hat_t) * noise

    cond = tf.convert_to_tensor(conditioning, dtype=tf.int32)
    for step in reversed(range(t_start)):
        t_batch = tf.fill((n_samples,), step)
        eps_u = gen_model([x, t_batch, tf.fill((n_samples,), NULL_CLS)], training=False)
        eps_c = gen_model([x, t_batch, cond], training=False)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = p_sample_step(x, t_batch, eps, betas2[step], alphas2[step], alpha_hats2[step])

    return (x + 1.0) * 0.5
