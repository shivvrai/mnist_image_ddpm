"""Test with very high guidance to see if conditioning works at all."""
import torch
import numpy as np
from PIL import Image
from ddpm import UNet, gen_samples

model = UNet()
model.load_state_dict(torch.load("weights/ddpm_unet.pt", map_location="cpu", weights_only=True))
model.eval()

# Test with high guidance scale - this should force strong conditioning
grid_size = 28
scales = [0.0, 1.0, 3.0, 5.0, 10.0]
grid = Image.new('L', (5 * grid_size, 4 * grid_size), 0)

for row, digit in enumerate([3, 7, 0, 1]):
    for col, gs in enumerate(scales):
        print(f"Digit {digit}, GS={gs}...")
        imgs = gen_samples(model, n_samples=1, conditioning=[digit], guidance_scale=gs, seed=42)
        img_np = imgs[0, 0].cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='L')
        grid.paste(pil_img, (col * grid_size, row * grid_size))

grid_up = grid.resize((5 * grid_size * 4, 4 * grid_size * 4), Image.NEAREST)
grid_up.save("scratch_guidance_test.png")
print("Saved to scratch_guidance_test.png")
print("Columns: GS=0, 1, 3, 5, 10")
print("Rows: digits 3, 7, 0, 1")
