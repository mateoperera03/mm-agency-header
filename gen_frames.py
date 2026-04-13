import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(__file__)
STATIC = os.path.join(ROOT, 'assets', 'mac-static-new.png')
EXPLOSION = os.path.join(ROOT, 'assets', 'mac-explosion-new.png')
OUT_DIR = os.path.join(ROOT, 'assets', 'frames')

if os.path.isdir(OUT_DIR):
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
else:
    os.makedirs(OUT_DIR)

FRAME_COUNT = 100
static = Image.open(STATIC).convert('RGB')
explosion = Image.open(EXPLOSION).convert('RGB')
if explosion.size != static.size:
    explosion = explosion.resize(static.size, Image.LANCZOS)
W, H = static.size

s_arr = np.array(static).astype(np.float32)
e_arr = np.array(explosion).astype(np.float32)

# Precompute distance from center (for radial mask)
cx, cy = W / 2, H / 2
ys = np.arange(H).reshape(-1, 1)
xs = np.arange(W).reshape(1, -1)
dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
max_dist = np.sqrt(cx ** 2 + cy ** 2)
norm_dist = dist / max_dist  # 0 at center, 1 at corners

def ease_in_out_cubic(t):
    return 4*t*t*t if t < 0.5 else 1 - ((-2*t + 2) ** 3) / 2

# Timing: 0-15% hold static, 15-85% radial reveal, 85-100% hold explosion
for i in range(FRAME_COUNT):
    t = i / (FRAME_COUNT - 1)

    if t < 0.15:
        phase = 0.0
    elif t > 0.85:
        phase = 1.0
    else:
        phase = ease_in_out_cubic((t - 0.15) / 0.70)

    # Radius of the reveal circle (normalized). Feather width for soft edge.
    radius = phase * 1.15  # goes slightly past corners so full reveal at phase=1
    feather = 0.08

    # Mask: 1 inside radius, 0 outside, smooth transition across feather
    mask = np.clip((radius - norm_dist) / feather + 0.5, 0, 1)
    mask = mask[:, :, None]  # broadcast to 3 channels

    blend = s_arr * (1 - mask) + e_arr * mask

    # Subtle zoom for drama
    zoom = 1.0 + 0.05 * phase
    img = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8), 'RGB')
    if zoom > 1.001:
        zw, zh = int(W * zoom), int(H * zoom)
        zoomed = img.resize((zw, zh), Image.LANCZOS)
        left = (zw - W) // 2
        top = (zh - H) // 2
        img = zoomed.crop((left, top, left + W, top + H))

    out_path = os.path.join(OUT_DIR, f'frame_{i+1:04d}.webp')
    img.save(out_path, 'WEBP', quality=95, method=6)
    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/{FRAME_COUNT}]')

print(f'Done. {FRAME_COUNT} frames at {W}x{H}')
