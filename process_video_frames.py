import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames')
files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.png')))
total = len(files)
print(f'Procesando {total} frames...')

TOL = 28
BLACK = np.array([10, 10, 10], dtype=np.float32)

for i, f in enumerate(files, 1):
    img = Image.open(f).convert('RGB')
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    # Sample bg color as median of outer border
    border = np.concatenate([
        arr[0:2, :].reshape(-1, 3),
        arr[h-2:h, :].reshape(-1, 3),
        arr[:, 0:2].reshape(-1, 3),
        arr[:, w-2:w].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    diff = np.abs(arr - bg).max(axis=2)
    similar = diff < TOL

    # Flood fill from border to keep the center burst intact
    labels, _ = ndimage.label(similar)
    border_labels = set()
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    border_labels.discard(0)
    bg_mask = np.isin(labels, list(border_labels))

    # Hard replace all flood-filled bg pixels with black
    out = arr.astype(np.float32)
    out[bg_mask] = BLACK
    # Soft feather at boundary: pixels adjacent to bg_mask get partial blend
    # based on how similar they are to bg
    edge = ndimage.binary_dilation(bg_mask, iterations=2) & ~bg_mask
    if edge.any():
        edge_diff = diff[edge].astype(np.float32)
        edge_alpha = np.clip(1 - (edge_diff / (TOL * 1.5)), 0, 1)
        out[edge] = out[edge] * (1 - edge_alpha[:, None]) + BLACK * edge_alpha[:, None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Mask out the Kling watermark (bottom-right corner)
    wm_w = int(w * 0.12)
    wm_h = int(h * 0.08)
    out[h - wm_h:h, w - wm_w:w] = BLACK

    out_path = f.replace('.png', '.webp')
    Image.fromarray(out, 'RGB').save(out_path, 'WEBP', quality=95, method=6)
    os.remove(f)
    if i % 20 == 0:
        print(f'  [{i}/{total}]')

print('Done.')
