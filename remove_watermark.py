import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(__file__)
for name in ['mac-static-new.png', 'mac-explosion-new.png']:
    p = os.path.join(ROOT, 'assets', name)
    img = Image.open(p).convert('RGBA')
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Watermark bounding box (bottom-right corner ~6% width x 10% height)
    cw = int(w * 0.06)
    ch = int(h * 0.10)
    x0 = w - cw
    y0 = h - ch

    # Sample bg from the strip just above the watermark (clean area)
    sample_strip = arr[max(0, y0 - 30):y0, x0:w, :3]
    bg_color = sample_strip.mean(axis=(0, 1)).astype(np.uint8)

    # Fill the watermark region with that color (add tiny noise to avoid banding)
    fill = np.tile(bg_color, (ch, cw, 1)).astype(np.int16)
    noise = np.random.randint(-3, 4, size=(ch, cw, 3))
    arr[y0:h, x0:w, :3] = np.clip(fill + noise, 0, 255).astype(np.uint8)

    Image.fromarray(arr, 'RGBA').save(p)
    print(f'Cleaned {name}: bg={tuple(int(c) for c in bg_color)}')
