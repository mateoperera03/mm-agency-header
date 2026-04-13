import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames_clean')
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.webp')))
total = len(files)
print(f'Procesando {total} frames...')

TOL = 28

for i, f in enumerate(files, 1):
    name = os.path.basename(f)
    out_path = os.path.join(OUT_DIR, name)
    img = Image.open(f).convert('RGBA')
    arr = np.array(img)
    h, w = arr.shape[:2]

    border = np.concatenate([
        arr[0:2, :, :3].reshape(-1, 3),
        arr[h-2:h, :, :3].reshape(-1, 3),
        arr[:, 0:2, :3].reshape(-1, 3),
        arr[:, w-2:w, :3].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)

    rgb = arr[:, :, :3].astype(np.int16)
    diff = np.abs(rgb - bg).max(axis=2)
    similar = diff < TOL  # pixels similar to bg color

    # Flood fill from border: label connected components of `similar`, keep only
    # components that touch the image border. Interior white pools (the burst) are
    # NOT connected to the border and therefore stay opaque.
    labels, n = ndimage.label(similar)
    border_labels = set()
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    border_labels.discard(0)

    mask = np.isin(labels, list(border_labels))
    arr[mask, 3] = 0

    Image.fromarray(arr, 'RGBA').save(out_path, 'WEBP', quality=90)
    print(f'  [{i}/{total}] {name} bg={tuple(int(c) for c in bg)} removed={int(mask.sum())}')

print('Done.')
