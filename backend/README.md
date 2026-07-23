---
title: MNIST Handwriting Diffusion Backend API
emoji: 🖊️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: MNIST Handwriting Diffusion - Backend API
---

# MNIST Handwriting Diffusion - Backend API

This space serves the backend API. The frontend is deployed on Vercel.

## Endpoints

The API endpoints are served via Gradio Blocks:

- `/api/health` - Check backend status and loaded models
- `/api/generate` - Generate digits using DDPM conditional diffusion
- `/api/classify-sketch` - Classify custom user sketch using PyTorch MNIST classifier
- `/api/generate-from-sketch` - Image-to-image generation from user sketch
