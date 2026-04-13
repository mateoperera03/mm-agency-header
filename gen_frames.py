import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(__file__)
STATIC = os.path.join(ROOT, 'assets', 'mac-static-new.png')
EXPLOSION = os.path.join(ROOT, 'assets', 'mac-explosion-new.png')
OUT_DIR = os.path.join(ROOT, 'assets', 'frames')

# Clean out old frames
if os.path.isdir(OUT_DIR):
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
else:
    os.makedirs(OUT_DIR)

FRAME_COUNT = 100
static = Image.open(STATIC).convert('RGB')
explosion = Image.open(EXPLOSION).convert('RGB')
W, H = static.size
if explosion.size != static.size:
    explosion = explosion.resize(static.size, Image.LANCZOS)

s_arr = np.array(static).astype(np.float32)
e_arr = np.array(explosion).astype(np.float32)

def ease_in_out_cubic(t):
    return 4*t*t*t if t < 0.5 else 1 - ((-2*t + 2) ** 3) / 2

for i in range(FRAME_COUNT):
    t = i / (FRAME_COUNT - 1)
    et = ease_in_out_cubic(t)

    # Crossfade
    blend = s_arr * (1 - et) + e_arr * et

    # Subtle zoom-in towards the explosion (1.0 -> 1.06)
    zoom = 1.0 + 0.06 * et
    img = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8), 'RGB')
    if zoom > 1.001:
        zw, zh = int(W * zoom), int(H * zoom)
        zoomed = img.resize((zw, zh), Image.LANCZOS)
        # center crop back to W,H
        left = (zw - W) // 2
        top = (zh - H) // 2
        img = zoomed.crop((left, top, left + W, top + H))

    out_path = os.path.join(OUT_DIR, f'frame_{i+1:04d}.webp')
    img.save(out_path, 'WEBP', quality=95, method=6)
    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/{FRAME_COUNT}]')

print(f'Done. {FRAME_COUNT} frames at {W}x{H}')
