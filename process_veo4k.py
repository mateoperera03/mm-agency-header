import os
import glob
import numpy as np
from PIL import Image

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames')
files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.png')))
total = len(files)
print(f'Procesando {total} frames 4K...')

# Target size: 2560x1440 (sharper than any realistic browser, half the size of raw 4K)
TARGET_W, TARGET_H = 2560, 1440
BLACK = (0, 0, 0)

for i, f in enumerate(files, 1):
    img = Image.open(f).convert('RGB')
    w, h = img.size
    if i == 1:
        print(f'  source size: {w}x{h}')

    # Downscale with high-quality Lanczos
    img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    arr = np.array(img)
    wh, ww = arr.shape[:2]

    # Mask out watermark in bottom-right
    wm_w = int(ww * 0.08)
    wm_h = int(wh * 0.09)
    arr[wh - wm_h:wh, ww - wm_w:ww] = BLACK

    out_path = f.replace('.png', '.webp')
    Image.fromarray(arr).save(out_path, 'WEBP', quality=88, method=6)
    os.remove(f)
    if i % 30 == 0:
        print(f'  [{i}/{total}]')

print('Done.')
