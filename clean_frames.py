import os
import glob
from rembg import remove, new_session
from PIL import Image

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'frames_clean')
os.makedirs(OUT_DIR, exist_ok=True)

session = new_session('u2net')

files = sorted(glob.glob(os.path.join(FRAMES_DIR, 'frame_*.webp')))
total = len(files)
print(f'Procesando {total} frames...')

for i, f in enumerate(files, 1):
    name = os.path.basename(f)
    out_path = os.path.join(OUT_DIR, name)
    with open(f, 'rb') as inp:
        data = inp.read()
    result = remove(data, session=session)
    img = Image.open(__import__('io').BytesIO(result))
    img.save(out_path, 'WEBP', quality=90)
    print(f'  [{i}/{total}] {name}')

print('Done.')
