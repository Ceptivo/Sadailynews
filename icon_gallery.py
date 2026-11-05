"""Preview every icon in icons.py on the reel background: python3 icon_gallery.py -> build/icon_gallery.jpg"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import icons

DIR = os.path.dirname(os.path.abspath(__file__))
CELL, COLS = 360, 5


def compose(ic, size):
    w, h = ic.size
    k = min((size - 40) / w, (size - 70) / h)
    nw, nh = int(w * k), int(h * k)
    body = cv2.resize(ic.body, (nw, nh), interpolation=cv2.INTER_AREA)
    gl = cv2.resize(ic.glow, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.zeros((size, size, 3), np.float32)
    cell[:] = (12, 18, 44)
    x, y = (size - nw) // 2, (size - 50 - nh) // 2 + 10
    neon = np.zeros_like(cell)
    neon[y:y + nh, x:x + nw] = gl
    halo = cv2.GaussianBlur(neon, (0, 0), 6) * 1.2 + cv2.GaussianBlur(neon, (0, 0), 18) * 0.8
    cell = np.clip(cell + neon + halo, 0, 255)
    a = body[..., 3:4] / 255.0
    cell[y:y + nh, x:x + nw] = cell[y:y + nh, x:x + nw] * (1 - a) + body[..., :3] * a
    return cell.astype(np.uint8)


def main(names=None):
    items = [(n, f) for n, f in icons.GALLERY if not names or n.split()[0] in names]
    rows = (len(items) + COLS - 1) // COLS
    sheet = np.zeros((rows * CELL, COLS * CELL, 3), np.uint8)
    font = ImageFont.truetype(f"{DIR}/fonts/Inter-Bold.otf", 22)
    for i, (name, fn) in enumerate(items):
        cell = compose(fn(), CELL)
        im = Image.fromarray(cell)
        ImageDraw.Draw(im).text((CELL / 2, CELL - 24), name, font=font, fill=(200, 210, 235), anchor="mm")
        r, c = divmod(i, COLS)
        sheet[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = np.array(im)
    os.makedirs(f"{DIR}/build", exist_ok=True)
    out = f"{DIR}/build/icon_gallery.jpg"
    Image.fromarray(sheet).save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
