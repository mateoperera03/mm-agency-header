import os
import glob
import numpy as np
from PIL import Image

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames')
files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.png')))
total = len(files)
print(f'Procesando {total} frames...')

BLACK = (0, 0, 0)

for i, f in enumerate(files, 1):
    img = Image.open(f).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Sample the very corner to know if bg is already black
    if i == 1:
        print('corner sample:', tuple(int(c) for c in arr[5, 5]))

    # Mask out Veo watermark in bottom-right (~8% width x 8% height)
    wm_w = int(w * 0.08)
    wm_h = int(h * 0.09)
    arr[h - wm_h:h, w - wm_w:w] = BLACK

    out_path = f.replace('.png', '.webp')
    Image.fromarray(arr, 'RGB').save(out_path, 'WEBP', quality=95, method=6)
    os.remove(f)
    if i % 30 == 0:
        print(f'  [{i}/{total}]')

print('Done.')
