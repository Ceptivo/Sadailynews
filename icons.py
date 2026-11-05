"""Detailed, shaded illustrations for story scenes (the "neon realism" style).

Every icon is drawn once at 3x supersampling and cached, then placed with render.place_icon().
Each Icon has two layers:
  body  RGBA solid illustration (gradients, bevels, highlights, soft shadow), composited crisp on top
  glow  RGB rim light in the accent colour, added to the neon layer so it blooms
Style rules for new icons: a solid gradient body (lit from the top-left), a darker side or
underside for depth, at least one white highlight/sheen, small real-world details (screws,
seams, windows, labels), a soft ground shadow, and an accent-coloured rim stroke with glow=1.

Preview them all: python3 icon_gallery.py  ->  build/icon_gallery.jpg
"""
import json
import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIR = os.path.dirname(os.path.abspath(__file__))
F_BLACK = f"{DIR}/fonts/InterDisplay-Black.otf"
F_BOLD = f"{DIR}/fonts/Inter-Bold.otf"
F_MONO = f"{DIR}/fonts/DejaVuSansMono-Bold.ttf"

CYAN = (0, 235, 255)
MAG = (255, 40, 190)
GOLD = (255, 186, 30)
GREEN = (0, 215, 120)
RED = (255, 50, 70)
BLUE = (60, 110, 255)
WHITE = (255, 255, 255)
NAVY = (10, 16, 40)
SS = 3
AA = cv2.LINE_AA


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def lt(c, t):
    return mix(c, WHITE, t)


def dk(c, t):
    return mix(c, (0, 0, 0), t)


class Icon:
    def __init__(self, body, glow):
        self.body, self.glow = body, glow

    @property
    def size(self):
        return self.body.shape[1], self.body.shape[0]


# ---------- shape point helpers (all in icon pixel units) ----------
def circ(cx, cy, r, n=120):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def ell(cx, cy, rx, ry, rot=0.0, a0=0.0, a1=360.0, n=120):
    t = np.radians(np.linspace(a0, a1, n))
    x, y = rx * np.cos(t), ry * np.sin(t)
    r = math.radians(rot)
    return np.column_stack([cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)])


def rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)


def rrect(x0, y0, x1, y1, r, n=10):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]:
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(pts)


def star(cx, cy, ro, ri, n=5, rot=-90):
    pts = []
    for i in range(n * 2):
        a = math.radians(rot + i * 180 / n)
        r = ro if i % 2 == 0 else ri
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def bez(p0, p1, p2, p3=None, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    if p3 is None:
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    p3 = np.array(p3)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def drop(cx, cy, r, n=90):
    """Teardrop with the point at the top; (cx, cy) is the centre of the round part."""
    pts = []
    th = math.acos(1 / 2.1)
    for a in np.linspace(-90 + math.degrees(th), 270 - math.degrees(th), n):
        pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array([(cx, cy - 2.1 * r)] + pts)


def rot_pts(pts, cx, cy, deg):
    a = math.radians(deg)
    p = np.asarray(pts, float) - (cx, cy)
    return np.column_stack([p[:, 0] * math.cos(a) - p[:, 1] * math.sin(a),
                            p[:, 0] * math.sin(a) + p[:, 1] * math.cos(a)]) + (cx, cy)


# ---------- supersampled canvas ----------
class Cv:
    def __init__(self, w, h):
        self.w, self.h = w, h
        H, W = h * SS, w * SS
        self.rgb = np.zeros((H, W, 3), np.float32)
        self.a = np.zeros((H, W), np.float32)
        self.g = np.zeros((H, W, 3), np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx / SS, yy / SS

    # masks
    def _polys(self, polys):
        if isinstance(polys, np.ndarray) and polys.ndim == 2:
            polys = [polys]
        elif polys and not hasattr(polys[0][0], "__len__"):
            polys = [polys]
        return [np.round(np.asarray(p, float) * SS * 4).astype(np.int32) for p in polys]

    def mask(self, polys):
        m = np.zeros(self.a.shape, np.uint8)
        for p in self._polys(polys):  # union (one fillPoly call per shape, so overlaps don't cancel)
            cv2.fillPoly(m, [p], 255, AA, shift=2)
        return m.astype(np.float32) / 255

    def line_mask(self, polys, th, closed=True):
        m = np.zeros(self.a.shape, np.uint8)
        cv2.polylines(m, self._polys(polys), closed, 255, max(1, int(round(th * SS))), AA, shift=2)
        return m.astype(np.float32) / 255

    # colour fields
    def lin(self, p1, p2, c1, c2):
        d = np.array(p2, float) - p1
        t = ((self.xx - p1[0]) * d[0] + (self.yy - p1[1]) * d[1]) / max(d @ d, 1e-6)
        t = np.clip(t, 0, 1)[..., None]
        return np.array(c1, np.float32) + (np.array(c2, np.float32) - c1) * t

    def lin_a(self, p1, p2, a1, a2):
        d = np.array(p2, float) - p1
        t = np.clip(((self.xx - p1[0]) * d[0] + (self.yy - p1[1]) * d[1]) / max(d @ d, 1e-6), 0, 1)
        return a1 + (a2 - a1) * t

    def rad(self, c, r, c1, c2):
        t = np.clip(np.hypot(self.xx - c[0], self.yy - c[1]) / r, 0, 1)[..., None]
        return np.array(c1, np.float32) + (np.array(c2, np.float32) - c1) * t

    def rad_a(self, c, r, a1, a2, rx=None, ry=None):
        rx, ry = rx or r, ry or r
        t = np.clip(np.hypot((self.xx - c[0]) / rx, (self.yy - c[1]) / ry), 0, 1)
        return a1 + (a2 - a1) * t

    def cyl(self, x0, x1, light, dark, axis="x"):
        """Cylinder shading across x (vertical cylinder) or y (horizontal cylinder)."""
        v = self.xx if axis == "x" else self.yy
        t = np.clip((v - x0) / (x1 - x0), 0, 1)
        k = np.clip(np.sin(np.pi * np.clip(t * 1.15 - 0.05, 0, 1)), 0, 1) ** 0.8
        k = k[..., None]
        return np.array(dark, np.float32) + (np.array(light, np.float32) - dark) * k

    # painting
    def paint(self, m, color, alpha=1.0, clip=None):
        k = m * alpha
        if clip is not None:
            k = k * clip
        ys, xs = np.nonzero(k > 0.002)
        if len(ys) == 0:
            return
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        kk = k[y0:y1, x0:x1, None]
        C = color[y0:y1, x0:x1] if isinstance(color, np.ndarray) else np.array(color, np.float32)
        self.rgb[y0:y1, x0:x1] = self.rgb[y0:y1, x0:x1] * (1 - kk) + C * kk
        self.a[y0:y1, x0:x1] += kk[..., 0] * (1 - self.a[y0:y1, x0:x1])

    def fill(self, polys, color, alpha=1.0, clip=None):
        m = self.mask(polys)
        self.paint(m, color, alpha, clip)
        return m

    def stroke(self, polys, color, th, alpha=1.0, closed=True, glow=0.0, clip=None):
        m = self.line_mask(polys, th, closed)
        self.paint(m, color, alpha, clip)
        if glow:
            self.add_glow(m if clip is None else m * clip, color, glow)
        return m

    def add_glow(self, m, color, k=1.0):
        C = np.array(color, np.float32) if not isinstance(color, np.ndarray) else color
        self.g = np.maximum(self.g, m[..., None] * C * k)

    def glow_fill(self, polys, color, k=1.0, blur=0):
        m = self.mask(polys)
        if blur:
            m = cv2.GaussianBlur(m, (0, 0), blur * SS)
        self.add_glow(m, color, k)

    def soft(self, polys, color, alpha=0.6, blur=10):
        m = cv2.GaussianBlur(self.mask(polys), (0, 0), blur * SS)
        self.paint(m, color, alpha)

    def shadow(self, cx, cy, rx, ry, alpha=0.55, blur=10):
        self.soft(ell(cx, cy, rx, ry), (0, 0, 0), alpha, blur)

    def text(self, txt, x, y, size, color, font=F_BLACK, alpha=1.0, glow=0.0):
        f = ImageFont.truetype(font, int(size * SS))
        im = Image.new("L", (self.w * SS, self.h * SS), 0)
        d = ImageDraw.Draw(im)
        d.text((x * SS, y * SS), txt, font=f, fill=255, anchor="mm")
        m = np.array(im).astype(np.float32) / 255
        self.paint(m, color, alpha)
        if glow:
            self.add_glow(m, color, glow)

    def done(self):
        a = self.a
        pre = self.rgb * a[..., None]
        sz = (self.w, self.h)
        A = cv2.resize(a, sz, interpolation=cv2.INTER_AREA)
        P = cv2.resize(pre, sz, interpolation=cv2.INTER_AREA)
        rgb = P / np.maximum(A[..., None], 1e-4)
        body = np.dstack([np.clip(rgb, 0, 255), np.clip(A * 255, 0, 255)]).astype(np.uint8)
        g = np.clip(cv2.resize(self.g, sz, interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
        return Icon(body, g)


def sparkle(c, x, y, r, alpha=1.0, color=WHITE):
    pts = star(x, y, r, r * 0.18, 4, -90)
    c.fill(pts, color, alpha)
    c.glow_fill(pts, color, 0.9 * alpha)


# =====================================================================
# ICONS
# =====================================================================
@lru_cache(maxsize=8)
def ballot_box(acc=GOLD, label="VOTE"):
    """Ballot box with a ballot (ticked) going into the slot. 540x500."""
    c = Cv(540, 500)
    c.shadow(275, 458, 235, 26)
    F = rect(70, 215, 390, 450)
    TOP = np.array([(70, 215), (160, 150), (480, 150), (390, 215)])
    SIDE = np.array([(390, 215), (480, 150), (480, 385), (390, 450)])
    navy = (44, 66, 138)
    c.fill(SIDE, c.lin((390, 0), (480, 0), dk(navy, 0.45), dk(navy, 0.7)))
    c.fill(F, c.lin((0, 215), (0, 450), lt(navy, 0.08), dk(navy, 0.5)))
    c.fill(F, WHITE, alpha=c.lin_a((70, 215), (260, 450), 0.18, 0.0))
    c.fill(TOP, c.lin((0, 150), (0, 215), lt(navy, 0.4), lt(navy, 0.12)))
    # label band wraps round the side
    c.fill(rect(70, 300, 390, 362), c.lin((0, 300), (0, 362), lt(acc, 0.25), dk(acc, 0.2)))
    c.fill([(390, 300), (480, 235), (480, 297), (390, 362)], c.lin((390, 0), (480, 0), dk(acc, 0.3), dk(acc, 0.5)))
    c.stroke([[(70, 302), (390, 302)]], WHITE, 2, 0.55, closed=False)
    c.text(label, 230, 333, 44, NAVY)
    # screws
    for x, y in [(92, 236), (368, 236), (92, 430), (368, 430)]:
        c.fill(circ(x, y, 6), c.rad((x - 2, y - 2), 7, (210, 220, 240), (90, 100, 130)))
    # slot + ballot paper
    c.fill([(195, 194), (222, 174), (354, 174), (327, 194)], (6, 8, 20))
    P = np.array([(224, 188), (240, 34), (362, 48), (340, 188)])
    c.fill(P, c.lin((0, 40), (0, 188), (252, 252, 248), (190, 198, 220)))
    c.fill(P, (0, 0, 0), alpha=c.lin_a((0, 150), (0, 188), 0.0, 0.35))
    c.stroke(P, (140, 150, 185), 2)
    bx = np.array([(258, 74), (292, 77), (289, 111), (255, 108)])
    c.stroke(bx, (80, 92, 125), 3)
    c.stroke([[(262, 92), (272, 104), (300, 62)]], (0, 175, 95), 7, closed=False)
    c.stroke([[(304, 88), (342, 92)]], (150, 160, 185), 4, closed=False)
    c.stroke([[(304, 102), (334, 105)]], (150, 160, 185), 4, closed=False)
    c.stroke([[(250, 135), (338, 143)]], (170, 178, 200), 3, closed=False)
    c.stroke([[(248, 152), (330, 159)]], (170, 178, 200), 3, closed=False)
    c.fill([(199, 191), (331, 191), (327, 194), (195, 194)], lt(navy, 0.15))
    # rims
    c.stroke([F, TOP, SIDE], lt(acc, 0.25), 3, glow=1.0)
    c.stroke([[(70, 215), (390, 215)]], WHITE, 2, 0.7, closed=False)
    return c.done()


@lru_cache(maxsize=8)
def fuel_pump(acc=MAG, price="-1c", sub="PER LITRE"):
    """Petrol pump with lit price display, hose and nozzle. 520x580."""
    c = Cv(520, 580)
    c.shadow(250, 548, 210, 22)
    c.fill(rrect(92, 505, 388, 542, 8), c.lin((0, 505), (0, 542), (78, 88, 120), (28, 34, 54)))
    c.stroke([[(96, 507), (384, 507)]], WHITE, 2, 0.4, closed=False)
    c.fill([(345, 88), (377, 108), (377, 498), (345, 505)], c.lin((345, 0), (377, 0), dk(acc, 0.55), dk(acc, 0.78)))
    body = rrect(125, 70, 345, 505, 26)
    c.fill(body, c.lin((125, 0), (345, 0), lt(dk(acc, 0.12), 0.12), dk(acc, 0.55)))
    c.fill(rrect(140, 100, 162, 490, 10), WHITE, alpha=0.16)
    c.fill(rrect(110, 46, 360, 102, 18), c.lin((0, 46), (0, 102), (68, 78, 110), (22, 28, 48)))
    c.stroke([[(124, 50), (346, 50)]], WHITE, 2, 0.5, closed=False)
    c.fill(rrect(150, 124, 320, 248, 14), (6, 9, 22))
    c.fill(rrect(160, 134, 310, 238, 8), c.lin((0, 134), (0, 238), (16, 42, 64), (6, 16, 28)))
    c.glow_fill(rrect(160, 134, 310, 238, 8), CYAN, 0.18)
    c.text(price, 235, 176, 46, CYAN, glow=0.9)
    c.text(sub, 235, 220, 15, (120, 175, 205), F_BOLD)
    c.fill([(160, 134), (310, 134), (310, 160), (160, 190)], WHITE, alpha=0.06)
    for i in range(3):
        x = 185 + i * 50
        c.fill(circ(x, 282, 13), c.rad((x - 4, 278), 15, (225, 230, 245), (90, 96, 120)))
    c.fill(circ(235, 380, 60), c.rad((218, 362), 75, (255, 255, 255), (190, 196, 215)))
    c.stroke(circ(235, 380, 60), dk(acc, 0.3), 3)
    c.fill(drop(235, 392, 26), c.lin((0, 335), (0, 420), lt(acc, 0.25), dk(acc, 0.3)))
    c.fill(ell(226, 388, 6, 11), WHITE, 0.6)
    # holster, hose, nozzle
    c.fill(rrect(338, 250, 392, 332, 8), c.lin((338, 0), (392, 0), (70, 76, 100), (26, 30, 46)))
    hose = bez((345, 140), (500, 140), (505, 430), (408, 318))
    c.stroke([hose], (20, 22, 34), 22, closed=False)
    c.stroke([hose - (3, 4)], (92, 98, 124), 7, closed=False)
    noz = np.array([(370, 296), (424, 286), (432, 306), (404, 316), (402, 372), (382, 375), (380, 322), (368, 318)])
    c.fill(noz, c.lin((370, 0), (432, 0), (225, 230, 242), (90, 96, 116)))
    c.stroke(noz, (40, 44, 60), 2)
    c.stroke([body], lt(acc, 0.2), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=8)
def arrow3d(up=True, color=None):
    """Extruded 3D arrow. 220x280. Green down / red up by default."""
    color = color or (RED if up else GREEN)
    c = Cv(220, 280)
    pts = np.array([(110, 20), (200, 120), (145, 120), (145, 255), (75, 255), (75, 120), (20, 120)], float)
    if not up:
        pts[:, 1] = 275 - pts[:, 1]
    c.fill(pts + (10, 8), dk(color, 0.6))
    c.fill(pts, c.lin((0, 20), (0, 260), lt(color, 0.3), dk(color, 0.25)))
    c.fill(pts, WHITE, alpha=c.lin_a((20, 20), (140, 200), 0.3, 0.0))
    c.stroke(pts, lt(color, 0.4), 3, glow=1.1)
    return c.done()


@lru_cache(maxsize=4)
def parliament(acc=BLUE):
    """Classical Parliament building with columns, pediment, lit doorway and SA flag. 660x540."""
    c = Cv(660, 540)
    st1, st2 = (238, 240, 248), (148, 156, 184)
    c.shadow(330, 515, 310, 18)
    for x0, x1, y in [(72, 588, 456), (52, 608, 476), (32, 628, 496)]:
        c.fill(rect(x0, y, x1, y + 20), c.lin((0, y), (0, y + 20), st1, st2))
        c.stroke([[(x0, y), (x1, y)]], WHITE, 2, 0.8, closed=False)
    c.fill(rect(90, 232, 570, 456), c.lin((0, 232), (0, 456), (24, 32, 70), (10, 14, 34)))
    door = rrect(298, 330, 362, 456, 30)
    c.fill(door, c.lin((0, 330), (0, 456), (255, 214, 130), (210, 120, 40)))
    c.glow_fill(door, GOLD, 0.9, blur=6)
    c.stroke([[(330, 336), (330, 456)]], dk(GOLD, 0.5), 2, closed=False)
    for i in range(8):
        x = 112 + i * 62.3
        c.fill(rect(x - 17, 252, x + 17, 442), c.cyl(x - 17, x + 17, st1, (120, 128, 158)))
        for dx in (-8, 0, 8):
            c.stroke([[(x + dx, 256), (x + dx, 438)]], (90, 98, 128), 1.5, 0.35, closed=False)
        c.fill(rect(x - 24, 238, x + 24, 252), c.lin((0, 238), (0, 252), st1, st2))
        c.fill(rect(x - 24, 442, x + 24, 456), c.lin((0, 442), (0, 456), st1, st2))
    c.fill(rect(60, 196, 600, 238), c.lin((0, 196), (0, 238), st1, st2))
    c.stroke([[(60, 222), (600, 222)]], (110, 118, 150), 2, 0.6, closed=False)
    for i in range(27):
        x = 70 + i * 20
        c.fill(rect(x, 226, x + 10, 236), (120, 128, 158), 0.5)
    tri = np.array([(48, 198), (330, 96), (612, 198)])
    c.fill(tri, c.lin((0, 96), (0, 198), st1, st2))
    c.fill([(112, 188), (330, 118), (548, 188)], c.lin((0, 118), (0, 188), (172, 180, 204), (118, 126, 156)))
    c.fill(circ(330, 160, 20), c.rad((324, 154), 24, (255, 230, 140), (180, 110, 10)))
    c.stroke(circ(330, 160, 20), dk(GOLD, 0.4), 2)
    # flag on the apex
    c.stroke([[(330, 98), (330, 14)]], (200, 205, 220), 4, closed=False)
    fx, fy, fw, fh = 333, 16, 66, 44
    c.fill(rect(fx, fy, fx + fw, fy + fh / 2), (224, 60, 49))
    c.fill(rect(fx, fy + fh / 2, fx + fw, fy + fh), (0, 35, 149))
    c.fill([(fx, fy), (fx + 26, fy + 18), (fx + fw, fy + 18), (fx + fw, fy + 26), (fx + 26, fy + 26), (fx, fy + fh)], WHITE)
    c.fill([(fx, fy + 4), (fx + 23, fy + 20), (fx + fw, fy + 20), (fx + fw, fy + 24), (fx + 23, fy + 24), (fx, fy + fh - 4)], (0, 122, 77))
    c.fill([(fx, fy + 10), (fx + 16, fy + 22), (fx, fy + 34)], (255, 184, 28))
    c.fill([(fx, fy + 13), (fx + 12, fy + 22), (fx, fy + 31)], (0, 0, 0))
    c.stroke([tri, rect(60, 196, 600, 238), rect(32, 456, 628, 516)], lt(acc, 0.35), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=8)
def medal(c1=GREEN, c2=GOLD, metal="gold"):
    """Medal on a two-colour ribbon. 440x600. metal: gold | silver | bronze."""
    pal = {"gold": ((255, 238, 150), (172, 112, 10)), "silver": ((250, 252, 255), (120, 130, 150)),
           "bronze": ((255, 196, 140), (130, 70, 30))}[metal]
    c = Cv(440, 600)
    R = np.array([(250, 0), (330, 0), (262, 338), (196, 338)])
    L = np.array([(110, 0), (190, 0), (244, 338), (178, 338)])
    c.fill(R, c.lin((250, 0), (330, 0), dk(c2, 0.15), dk(c2, 0.5)))
    c.fill(L, c.lin((110, 0), (190, 0), lt(c1, 0.1), dk(c1, 0.35)))
    c.stroke([[(150, 0), (211, 338)]], WHITE, 6, 0.7, closed=False)
    c.fill(L, (0, 0, 0), alpha=c.lin_a((0, 230), (0, 338), 0.0, 0.45))
    c.stroke(circ(220, 344, 18), c.lin((200, 326), (240, 362), pal[0], pal[1]), 8)
    c.shadow(220, 588, 120, 12)
    Rm = 130
    cx, cy = 220, 455
    c.fill(circ(cx, cy, Rm), c.rad((cx - 50, cy - 55), Rm * 1.5, pal[0], pal[1]))
    c.fill(circ(cx, cy, Rm * 0.8), c.rad((cx + 40, cy + 45), Rm * 1.2, lt(pal[0], 0.1), pal[1]))
    for i in range(48):
        a = math.radians(i * 7.5)
        p1 = (cx + math.cos(a) * Rm * 0.84, cy + math.sin(a) * Rm * 0.84)
        p2 = (cx + math.cos(a) * Rm * 0.92, cy + math.sin(a) * Rm * 0.92)
        c.stroke([[p1, p2]], dk(pal[1], 0.2), 2, 0.5, closed=False)
    s = star(cx, cy, Rm * 0.55, Rm * 0.23)
    c.fill(s + (4, 5), dk(pal[1], 0.4), 0.7)
    c.fill(s, c.lin((cx - 60, cy - 60), (cx + 60, cy + 60), lt(pal[0], 0.3), pal[1]))
    c.stroke(s, lt(pal[0], 0.5), 2)
    c.fill(ell(cx - 45, cy - 62, 52, 26, -35), WHITE, alpha=0.35)
    sparkle(c, cx + 88, cy - 92, 22)
    sparkle(c, cx - 98, cy + 70, 14, 0.8)
    c.stroke(circ(cx, cy, Rm), lt(pal[0], 0.2), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def trophy(acc=GOLD):
    """Trophy cup on a plinth. 400x520."""
    c = Cv(400, 520)
    g1, g2 = (255, 236, 150), (160, 100, 10)
    c.shadow(200, 500, 150, 14)
    c.fill(rrect(90, 420, 310, 495, 8), c.lin((0, 420), (0, 495), (90, 56, 30), (40, 22, 12)))
    c.fill(rect(140, 440, 260, 470), c.lin((0, 440), (0, 470), g1, g2))
    c.fill(rect(130, 395, 270, 422), c.cyl(130, 270, g1, g2))
    c.fill([(180, 300), (220, 300), (232, 395), (168, 395)], c.cyl(168, 232, g1, g2))
    for side in (-1, 1):
        arm = bez((200 + side * 110, 90), (200 + side * 190, 90), (200 + side * 170, 230), (200 + side * 80, 250))
        c.stroke([arm], dk(g2, 0.2), 20, closed=False)
        c.stroke([arm], c.lin((0, 80), (0, 260), g1, g2), 13, closed=False)
    cup = np.vstack([[(80, 60), (320, 60)], bez((320, 60), (320, 250), (240, 300), (200, 305))[1:],
                     bez((200, 305), (160, 300), (80, 250), (80, 60))[1:]])
    c.fill(cup, c.cyl(80, 320, g1, g2))
    c.fill(ell(200, 60, 120, 18), c.lin((0, 42), (0, 78), dk(g2, 0.4), g2))
    c.fill([(108, 80), (130, 80), (150, 260), (135, 250)], WHITE, alpha=0.4)
    c.fill(star(200, 165, 42, 18), dk(g2, 0.1), 0.7)
    c.stroke(star(200, 165, 42, 18), lt(g1, 0.4), 2)
    sparkle(c, 300, 70, 20)
    c.stroke([cup], lt(acc, 0.3), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def storm_cloud(bolt=True, rain=True):
    """Volumetric storm cloud with lightning and rain. 600x560."""
    c = Cv(600, 560)
    cx, cy = 300, 205
    blobs = [(-150, 40, 85), (-60, -15, 115), (60, -35, 120), (165, 20, 90), (10, 60, 100), (-205, 85, 60), (225, 85, 60)]
    flat = rect(0, 0, 600, cy + 140)
    back = c.mask([circ(cx + dx + 22, cy + dy - 38, r * 0.9) for dx, dy, r in blobs]) * c.mask(flat)
    c.paint(back, c.lin((0, 40), (0, 330), (120, 132, 175), (48, 56, 90)))
    m = c.mask([circ(cx + dx, cy + dy, r) for dx, dy, r in blobs]) * c.mask(flat)
    c.paint(m, c.lin((0, 70), (0, 345), (244, 247, 255), (112, 124, 170)))
    for dx, dy, r in blobs[:4]:
        c.paint(m, WHITE, alpha=c.rad_a((cx + dx - r * 0.3, cy + dy - r * 0.4), r * 0.9, 0.55, 0.0))
    c.paint(m, (20, 26, 60), alpha=c.lin_a((0, 250), (0, 345), 0.0, 0.5))
    edge = cv2.Canny((m * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8))
    c.add_glow(edge, CYAN, 0.8)
    if rain:
        for i, (x, y) in enumerate([(150, 400), (205, 455), (420, 410), (470, 470), (380, 495), (120, 480)]):
            d = drop(x, y, 11)
            c.fill(d, c.lin((0, y - 25), (0, y + 11), (180, 240, 255), (30, 140, 220)))
            c.glow_fill(d, CYAN, 0.6)
    if bolt:
        b = np.array([(318, 285), (258, 410), (306, 410), (270, 540), (378, 375), (326, 375), (362, 285)])
        c.fill(b, c.lin((0, 285), (0, 540), (255, 248, 190), (255, 160, 20)))
        c.stroke(b, (255, 255, 230), 2)
        c.glow_fill(b, GOLD, 1.3, blur=2)
    return c.done()


@lru_cache(maxsize=2)
def sun():
    """Glowing sun with rays. 360x360."""
    c = Cv(360, 360)
    cx = cy = 180
    for i in range(12):
        a = math.radians(i * 30)
        tri = [(cx + math.cos(a - 0.12) * 98, cy + math.sin(a - 0.12) * 98),
               (cx + math.cos(a) * (170 if i % 2 == 0 else 145), cy + math.sin(a) * (170 if i % 2 == 0 else 145)),
               (cx + math.cos(a + 0.12) * 98, cy + math.sin(a + 0.12) * 98)]
        c.fill(tri, c.rad((cx, cy), 170, (255, 230, 120), (255, 120, 20)))
        c.glow_fill(tri, GOLD, 0.8)
    c.fill(circ(cx, cy, 92), c.rad((cx - 30, cy - 30), 130, (255, 250, 200), (255, 120, 20)))
    c.fill(ell(cx - 32, cy - 36, 34, 20, -35), WHITE, 0.45)
    c.glow_fill(circ(cx, cy, 100), (255, 140, 30), 1.0, blur=8)
    return c.done()


@lru_cache(maxsize=2)
def tap_drop():
    """Chrome tap with a falling water drop. 520x560."""
    c = Cv(520, 560)
    ch1, ch2 = (245, 248, 255), (90, 100, 125)
    c.fill(rrect(36, 112, 92, 248, 10), c.cyl(36, 92, ch1, ch2))
    c.fill(rect(92, 150, 300, 202), c.cyl(150, 202, ch1, ch2, axis="y"))
    r = np.hypot(c.xx - 300, c.yy - 252)
    t = np.clip((r - 50) / 52, 0, 1)
    k = (np.clip(np.sin(np.pi * t), 0, 1) ** 0.8)[..., None]
    arc_col = np.array(ch2, np.float32) + (np.array(ch1, np.float32) - ch2) * k
    outer = [(300 + 102 * math.cos(math.radians(a)), 252 + 102 * math.sin(math.radians(a))) for a in np.linspace(-90, 0, 30)]
    inner = [(300 + 50 * math.cos(math.radians(a)), 252 + 50 * math.sin(math.radians(a))) for a in np.linspace(0, -90, 30)]
    c.fill(np.array(outer + inner), arc_col)
    c.fill(rect(350, 250, 402, 300), c.cyl(350, 402, ch1, ch2))
    c.fill(rrect(342, 296, 410, 314, 5), c.cyl(342, 410, ch1, ch2))
    c.fill(rect(182, 112, 206, 152), c.cyl(182, 206, ch1, ch2))
    c.fill(rrect(132, 92, 256, 116, 12), c.cyl(92, 116, ch1, ch2, axis="y"))
    c.fill(circ(194, 104, 9), (60, 130, 255))
    d = drop(376, 420, 34)
    c.fill(d, c.lin((0, 345), (0, 455), (200, 248, 255), (20, 130, 230)))
    c.fill(ell(364, 418, 8, 15, 15), WHITE, 0.7)
    c.glow_fill(d, CYAN, 0.9)
    for i in range(3):
        c.stroke(ell(376, 520, 40 + i * 38, 9 + i * 7), CYAN, 3, 0.7 - i * 0.2, glow=0.8 - i * 0.2)
    c.stroke([np.array(outer)], CYAN, 2, 0.6, closed=False, glow=0.7)
    return c.done()


@lru_cache(maxsize=2)
def soccer_ball():
    """Shaded football with panel pattern. 340x360 (spin it with place_icon(angle=...))."""
    c = Cv(340, 360)
    cx, cy, R = 170, 165, 150
    c.shadow(cx, 340, 115, 14)
    sph = c.fill(circ(cx, cy, R), c.rad((cx - 50, cy - 55), R * 1.6, (255, 255, 255), (150, 160, 190)))
    dark = c.lin((0, cy - R), (0, cy + R), (36, 42, 66), (14, 16, 30))
    cp = star(cx, cy, R * 0.34, R * 0.34, 5)[::2]
    c.fill(cp, dark, clip=sph)
    for i in range(5):
        a = math.radians(-90 + i * 72)
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        ctr = np.array([cx, cy]) + u * R * 0.93
        pts = []
        for j in range(5):
            b = math.radians(90 + j * 72)
            lx, ly = math.cos(b) * R * 0.3, math.sin(b) * R * 0.3
            pts.append(ctr + u * (-ly * 0.55) + v * lx)
        c.fill(np.array(pts), dark, clip=sph)
        c.stroke([[cp[i], np.array([cx, cy]) + u * R * 0.66]], (50, 56, 80), 3, closed=False, clip=sph)
        nxt = math.radians(-90 + (i + 1) * 72)
        un = np.array([math.cos(nxt), math.sin(nxt)])
        mid = np.array([cx, cy]) + (u + un) / np.linalg.norm(u + un) * R * 0.8
        c.stroke([[np.array([cx, cy]) + u * R * 0.66, mid, np.array([cx, cy]) + un * R * 0.66]], (60, 66, 92), 3,
                 closed=False, clip=sph)
        c.stroke([[mid, np.array([cx, cy]) + (u + un) / np.linalg.norm(u + un) * R * 1.05]], (60, 66, 92), 3,
                 closed=False, clip=sph)
    c.paint(sph, (10, 14, 40), alpha=c.rad_a((cx - 40, cy - 45), R * 1.25, 0.0, 0.55))
    c.fill(ell(cx - 55, cy - 70, 45, 26, -35), WHITE, 0.55)
    c.stroke(circ(cx, cy, R), lt(GREEN, 0.3), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def rugby_ball(band=GREEN, stripe=GOLD):
    """Rugby ball (horizontal; tilt with place_icon(angle=...)). 460x300."""
    c = Cv(460, 300)
    cx, cy, rx, ry = 230, 140, 200, 112
    c.shadow(cx, 280, 170, 12)
    body = ell(cx, cy, rx, ry)
    m = c.fill(body, c.rad((cx - 60, cy - 50), rx * 1.25, (255, 255, 255), (150, 160, 190)))
    for k in (0.3, 0.62):
        up = ell(cx, cy, rx, ry * k, a0=180, a1=360)
        lo = ell(cx, cy, rx, ry * (k + 0.16), a0=180, a1=360)
        band_poly = np.vstack([up, lo[::-1]])
        c.fill(band_poly, c.lin((0, cy - ry), (0, cy), lt(band, 0.1), dk(band, 0.35)), clip=m)
        c.stroke([ell(cx, cy, rx, ry * (k + 0.08), a0=185, a1=355)], stripe, 3, clip=m, closed=False)
    c.stroke([ell(cx, cy, rx, ry * 0.02, a0=180, a1=360)], (120, 130, 160), 2, closed=False, clip=m)
    for i in range(-3, 4):
        x = cx + i * 20
        c.stroke([[(x - 6, cy - 12), (x + 6, cy + 10)]], (60, 66, 90), 6, closed=False)
        c.stroke([[(x - 6, cy - 12), (x + 6, cy + 10)]], WHITE, 3, closed=False)
    c.paint(m, (10, 14, 40), alpha=c.rad_a((cx - 40, cy - 30), 0, 0.0, 0.5, rx=rx * 1.15, ry=ry * 1.15))
    c.fill(ell(cx - 60, cy - 55, 70, 20, -8), WHITE, 0.5)
    c.stroke(body, lt(band, 0.3), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=40)
def clapperboard(angle=28, title="TAKE 1", acc=RED):
    """Film clapperboard; angle = how far the top stick is open (0-28). 520x460."""
    c = Cv(520, 460)
    c.shadow(260, 440, 210, 14)
    body = rrect(60, 175, 460, 425, 12)
    c.fill(body, c.lin((0, 175), (0, 425), (52, 56, 76), (18, 20, 32)))
    for y in (290, 345):
        c.stroke([[(80, y), (440, y)]], WHITE, 2, 0.35, closed=False)
    c.stroke([[(260, 290), (260, 400)]], WHITE, 2, 0.35, closed=False)
    c.text(title, 260, 255, 40, WHITE, alpha=0.95)
    c.text("SCENE", 160, 318, 18, (200, 205, 220), F_BOLD, 0.8)
    c.text("ROLL", 360, 318, 18, (200, 205, 220), F_BOLD, 0.8)
    bar = rect(60, 175, 460, 222)
    mb = c.fill(bar, (14, 14, 18))
    for k in range(7):
        x = 60 + k * 64
        c.fill([(x + 10, 175), (x + 42, 175), (x + 22, 222), (x - 10, 222)], (245, 245, 250), clip=mb)
    stick = np.array([(60, 120), (460, 120), (460, 168), (60, 168)])
    stick = rot_pts(stick, 60, 168, -angle)
    ms = c.fill(stick, (14, 14, 18))
    for k in range(7):
        x = 60 + k * 64
        s = rot_pts([(x - 10, 120), (x + 22, 120), (x + 42, 168), (x + 10, 168)], 60, 168, -angle)
        c.fill(s, (245, 245, 250), clip=ms)
    c.fill(circ(66, 170, 11), c.rad((62, 166), 13, (230, 235, 245), (90, 96, 120)))
    c.stroke([body, stick], lt(acc, 0.3), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=64)
def bars3d(heights, colors):
    """3D bar chart; heights in px (quantise them for caching), one colour per bar."""
    n = len(heights)
    w = 80 + n * 200
    c = Cv(w, 520)
    base = 470
    c.shadow(w / 2, base + 14, w / 2 - 30, 14)
    c.fill([(30, base), (w - 40, base), (w - 10, base - 20), (60, base - 20)], (40, 50, 90), 0.8)
    for i, (h, col) in enumerate(zip(heights, colors)):
        x0, x1, d = 60 + i * 200, 190 + i * 200, (30, -20)
        top = base - max(h, 2)
        F = rect(x0, top, x1, base)
        S = [(x1, top), (x1 + d[0], top + d[1]), (x1 + d[0], base + d[1]), (x1, base)]
        Tp = [(x0, top), (x0 + d[0], top + d[1]), (x1 + d[0], top + d[1]), (x1, top)]
        c.fill(S, dk(col, 0.55))
        c.fill(F, c.lin((0, top), (0, base), lt(col, 0.2), dk(col, 0.35)))
        c.fill(rect(x0 + 10, top + 8, x0 + 26, base - 8), WHITE, 0.18)
        c.fill(Tp, lt(col, 0.45))
        c.stroke([F, S, Tp], lt(col, 0.35), 2.5, glow=1.0)
    return c.done()


@lru_cache(maxsize=8)
def map_pin(color=MAG):
    """3D map pin. 320x450."""
    c = Cv(320, 450)
    cx, cy, r = 160, 150, 110
    c.shadow(cx, 418, 70, 14)
    d = 250
    th = math.degrees(math.acos(r / d))
    arc = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
           for a in np.linspace(90 - th, 90 + th - 360, 100)]
    pin = np.array([(cx, cy + d)] + arc)
    c.fill(pin, c.rad((cx - 45, cy - 50), 300, lt(color, 0.35), dk(color, 0.55)))
    c.fill(pin, WHITE, alpha=c.lin_a((cx - 110, 0), (cx + 20, 0), 0.18, 0.0))
    c.fill(circ(cx, cy, 52), c.rad((cx - 15, cy - 15), 70, (255, 255, 255), (200, 205, 225)))
    c.fill(circ(cx, cy, 38), c.lin((0, cy - 38), (0, cy + 38), (8, 12, 30), (40, 50, 90)))
    c.fill(ell(cx - 45, cy - 62, 36, 18, -30), WHITE, 0.45)
    c.stroke(pin, lt(color, 0.35), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def skyline(color=BLUE, tower=True):
    """City skyline with lit windows and a Hillbrow-style tower. 820x460."""
    c = Cv(820, 460)
    rng = np.random.default_rng(3)
    base = 430
    glass = mix(color, (40, 60, 110), 0.45)
    blds = [(40, 140, 250), (150, 240, 170), (250, 330, 290), (345, 420, 200), (520, 600, 140),
            (610, 700, 230), (710, 790, 300)]
    sil = []
    for x0, x1, top in blds:
        F = rect(x0, top, x1, base)
        c.fill([(x1, top), (x1 + 10, top - 8), (x1 + 10, base - 8), (x1, base)], dk(glass, 0.6))
        c.fill([(x0, top), (x0 + 10, top - 8), (x1 + 10, top - 8), (x1, top)], lt(glass, 0.35))
        c.fill(F, c.lin((0, top), (0, base), lt(glass, 0.15), dk(glass, 0.6)))
        c.fill(F, WHITE, alpha=c.lin_a((x0, 0), (x1, 0), 0.12, 0.0))
        for yy in range(int(top) + 14, base - 14, 22):
            for xx in range(x0 + 10, x1 - 14, 18):
                on = rng.random() < 0.38
                col = (255, 214, 130) if on else (30, 44, 80)
                c.fill(rect(xx, yy, xx + 9, yy + 12), col)
                if on:
                    c.glow_fill(rect(xx, yy, xx + 9, yy + 12), GOLD, 0.5)
        sil.append(F)
    if tower:
        tx = 470
        c.fill(rect(tx - 14, 120, tx + 14, base), c.cyl(tx - 14, tx + 14, (200, 210, 235), (70, 80, 110)))
        c.fill(rrect(tx - 34, 120, tx + 34, 175, 12), c.cyl(tx - 34, tx + 34, (220, 228, 248), (70, 80, 110)))
        c.fill(rect(tx - 30, 140, tx + 30, 152), (255, 210, 120))
        c.glow_fill(rect(tx - 30, 140, tx + 30, 152), GOLD, 0.8)
        c.stroke([[(tx, 120), (tx, 30)]], (220, 225, 240), 4, closed=False)
        c.fill(circ(tx, 30, 6), RED)
        c.glow_fill(circ(tx, 30, 9), RED, 1.2)
        sil.append(rect(tx - 34, 120, tx + 34, base))
    c.fill(rect(20, base, 800, base + 8), (30, 40, 80))
    c.stroke(sil, lt(color, 0.35), 2.5, glow=1.0)
    return c.done()


@lru_cache(maxsize=2)
def table_mountain():
    """Table Mountain with Lion's Head, tablecloth cloud, city lights and sea. 820x460."""
    c = Cv(820, 460)
    rng = np.random.default_rng(5)
    back = np.array([(0, 300), (80, 230), (140, 250), (200, 170), (260, 200), (820, 240), (820, 330), (0, 330)])
    c.fill(back, c.lin((0, 170), (0, 330), (60, 72, 120), (24, 30, 60)))
    mtn = np.array([(40, 330), (120, 250), (175, 150), (215, 105), (250, 120), (600, 112), (640, 150), (690, 200),
                    (720, 165), (745, 135), (770, 160), (800, 240), (820, 330)])
    m = c.fill(mtn, c.lin((0, 105), (0, 330), (138, 150, 190), (40, 50, 86)))
    c.paint(m, WHITE, alpha=c.lin_a((150, 0), (520, 0), 0.15, 0.0))
    for i in range(60):
        x = rng.uniform(260, 590)
        y0 = 118 + rng.uniform(0, 10)
        c.stroke([[(x, y0), (x + rng.uniform(-4, 4), y0 + rng.uniform(25, 70))]], (30, 38, 70), 1.5, 0.18,
                 closed=False, clip=m)
    c.stroke([[(255, 160), (590, 152)]], (30, 38, 70), 2, 0.4, closed=False)
    cloud = [ell(250 + i * 40, 106 + (i % 3) * 3, 46, 15) for i in range(10)]
    cm = cv2.GaussianBlur(c.mask(cloud), (0, 0), 6 * SS)
    c.paint(cm, (240, 245, 255), 0.8)
    for x, ln in ((300, 26), (380, 40), (460, 30), (540, 22)):
        c.soft([ell(x, 118 + ln / 2, 22, ln)], (230, 238, 255), 0.35, 8)
    c.fill(rect(0, 330, 820, 360), (12, 16, 34))
    for i in range(80):
        x, y = rng.uniform(10, 810), rng.uniform(334, 356)
        c.fill(circ(x, y, 2), (255, 210, 120))
    c.glow_fill(rect(0, 332, 820, 358), GOLD, 0.25)
    c.fill(rect(0, 360, 820, 460), c.lin((0, 360), (0, 460), (20, 70, 120), (6, 18, 44)))
    for i in range(14):
        y = 372 + i * 6
        x = rng.uniform(80, 700)
        c.stroke([[(x, y), (x + rng.uniform(40, 140), y)]], CYAN, 2, 0.35, closed=False)
    c.stroke([mtn[1:-1]], lt(CYAN, 0.2), 3, closed=False, glow=1.0)
    return c.done()


@lru_cache(maxsize=40)
def gavel(angle=0):
    """Judge's gavel and sound block; angle tilts the gavel up (0 = striking). 560x480."""
    c = Cv(560, 480)
    w1, w2 = (176, 104, 52), (72, 36, 14)
    c.shadow(300, 455, 190, 14)
    c.fill(ell(300, 420, 150, 30), c.lin((0, 390), (0, 450), dk(w2, 0.2), dk(w2, 0.5)))
    c.fill(rect(150, 395, 450, 420), c.lin((0, 395), (0, 420), w2, dk(w2, 0.4)))
    c.fill(ell(300, 395, 150, 30), c.rad((260, 385), 170, w1, w2))
    c.stroke(ell(300, 395, 150, 30), lt(w1, 0.3), 2, 0.6)
    pv = (470, 330)
    a = angle
    handle = rot_pts(rrect(260, 312, 500, 338, 13), *pv, a)
    c.fill(handle, c.lin(rot_pts([(0, 312)], *pv, a)[0], rot_pts([(0, 338)], *pv, a)[0], w1, w2))
    head = rot_pts(rrect(150, 250, 290, 375, 18), *pv, a)
    c.fill(head, c.cyl(0, 1, w1, w2) if False else c.lin(rot_pts([(150, 0)], *pv, a)[0], rot_pts([(290, 0)], *pv, a)[0], w1, w2))
    for x in (172, 268):
        band = rot_pts(rect(x - 8, 248, x + 8, 377), *pv, a)
        c.fill(band, c.lin(rot_pts([(x - 8, 0)], *pv, a)[0], rot_pts([(x + 8, 0)], *pv, a)[0], (255, 230, 140), (160, 100, 10)))
    c.fill(rot_pts(rrect(160, 262, 280, 280, 8), *pv, a), WHITE, 0.25)
    c.stroke([head, handle], lt(GOLD, 0.2), 2.5, glow=0.9)
    return c.done()


@lru_cache(maxsize=4)
def police_light(active="red"):
    """Police siren light; active = 'red' or 'blue' (alternate for a flashing effect). 420x300."""
    c = Cv(420, 300)
    c.shadow(210, 285, 170, 12)
    c.fill(rrect(40, 210, 380, 270, 14), c.lin((0, 210), (0, 270), (80, 88, 116), (24, 28, 44)))
    c.stroke([[(52, 214), (368, 214)]], WHITE, 2, 0.5, closed=False)
    dome = np.vstack([ell(210, 210, 150, 160, a0=180, a1=360), [(360, 210), (60, 210)]])
    left = c.mask(rect(0, 0, 210, 210))
    right = c.mask(rect(210, 0, 420, 210))
    dm = c.mask(dome)
    for side, colr, msk in (("red", RED, left), ("blue", BLUE, right)):
        on = active == side
        c.paint(dm, c.rad((210 + (-60 if side == "red" else 60), 130), 190,
                          lt(colr, 0.5) if on else dk(colr, 0.45), dk(colr, 0.3) if on else dk(colr, 0.75)), clip=msk)
        if on:
            c.add_glow(cv2.GaussianBlur(dm * msk, (0, 0), 8 * SS), colr, 1.3)
    for i in range(5):
        x = 110 + i * 50
        c.stroke([[(x, 205), (x, 90 + abs(i - 2) * 22)]], WHITE, 2, 0.2, closed=False)
    c.fill(ell(160, 110, 40, 22, -30), WHITE, 0.4)
    c.stroke([dome], (230, 235, 250), 2.5, 0.8)
    return c.done()


@lru_cache(maxsize=2)
def pylon():
    """Electricity pylon with insulators and sagging cables. 600x500."""
    c = Cv(600, 500)
    steel = c.lin((0, 40), (0, 470), (220, 228, 245), (100, 110, 140))
    L, R = [(220, 470), (270, 70)], [(380, 470), (330, 70)]
    c.stroke([L, R], steel, 7, closed=False)
    lvl = [70, 130, 190, 260, 330, 400, 470]
    for i in range(len(lvl) - 1):
        y0, y1 = lvl[i], lvl[i + 1]
        xl0, xl1 = 270 - (y0 - 70) / 8, 270 - (y1 - 70) / 8
        xr0, xr1 = 330 + (y0 - 70) / 8, 330 + (y1 - 70) / 8
        c.stroke([[(xl0, y0), (xr1, y1)], [(xr0, y0), (xl1, y1)], [(xl1, y1), (xr1, y1)]], steel, 3, closed=False)
    for y, span in ((130, 150), (200, 120)):
        c.stroke([[(300 - span, y), (300 + span, y)], [(300 - span, y), (262, y - 30)], [(300 + span, y), (338, y - 30)]],
                 steel, 5, closed=False)
        for x in (300 - span + 10, 300 + span - 10):
            for k in range(4):
                c.fill(ell(x, y + 10 + k * 9, 9, 4), c.lin((0, y), (0, y + 50), (150, 220, 255), (40, 110, 200)))
            cab_y = y + 46
            c.stroke([bez((x, cab_y), (x - 160 if x < 300 else x + 160, cab_y + 70), (0 if x < 300 else 600, cab_y + 20))],
                     (160, 170, 200), 2, closed=False)
    c.fill([(285, 70), (300, 30), (315, 70)], steel)
    c.stroke([[(270, 70), (220, 470)], [(330, 70), (380, 470)]], lt(GOLD, 0.2), 2, 0.0, closed=False, glow=0.8)
    return c.done()


@lru_cache(maxsize=4)
def bolt(color=GOLD):
    """Lightning bolt (electricity / load-shedding). 260x420."""
    c = Cv(260, 420)
    b = np.array([(150, 10), (40, 230), (120, 230), (70, 410), (225, 160), (140, 160), (200, 10)])
    c.fill(b + (8, 8), dk(color, 0.6))
    c.fill(b, c.lin((0, 10), (0, 410), lt(color, 0.6), dk(color, 0.1)))
    c.fill(b, WHITE, alpha=c.lin_a((60, 0), (200, 0), 0.35, 0.0))
    c.stroke(b, lt(color, 0.5), 3, glow=1.3)
    return c.done()


@lru_cache(maxsize=4)
def house(acc=CYAN):
    """House with tiled roof, lit windows and chimney. 560x480."""
    c = Cv(560, 480)
    wall1, wall2 = (236, 226, 210), (150, 140, 135)
    roof1, roof2 = (200, 80, 60), (90, 30, 26)
    c.shadow(290, 452, 240, 16)
    side = [(280, 230), (480, 190), (480, 390), (280, 430)]
    c.fill(side, c.lin((280, 0), (480, 0), dk(wall2, 0.1), dk(wall2, 0.45)))
    front = np.array([(80, 230), (180, 140), (280, 230), (280, 430), (80, 430)])
    c.fill(front, c.lin((0, 140), (0, 430), wall1, wall2))
    c.fill(rect(318, 270, 380, 330), (255, 212, 130))
    c.glow_fill(rect(318, 270, 380, 330), GOLD, 0.7)
    c.stroke([[(349, 270), (349, 330)], [(318, 300), (380, 300)]], (120, 80, 40), 3, closed=False)
    roof = [(180, 132), (380, 92), (500, 196), (290, 238)]
    rm = c.fill(roof, c.lin((180, 100), (420, 240), roof1, roof2))
    for k in range(1, 7):
        t = k / 7
        a = (180 + (290 - 180) * t, 132 + (238 - 132) * t)
        b = (380 + (500 - 380) * t, 92 + (196 - 92) * t)
        c.stroke([[a, b]], dk(roof2, 0.3), 2, 0.6, closed=False, clip=rm)
    c.fill([(338, 110), (372, 104), (372, 60), (338, 66)], c.lin((338, 0), (372, 0), (170, 90, 70), (90, 40, 30)))
    c.fill(rect(150, 330, 210, 430), c.lin((0, 330), (0, 430), (120, 70, 40), (60, 30, 16)))
    c.fill(circ(198, 382, 4), (255, 220, 140))
    c.fill(circ(180, 205, 20), (255, 212, 130))
    c.glow_fill(circ(180, 205, 20), GOLD, 0.7)
    c.fill(rect(100, 260, 140, 300), (255, 212, 130))
    c.glow_fill(rect(100, 260, 140, 300), GOLD, 0.6)
    c.stroke([front, side, roof], lt(acc, 0.3), 2.5, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def medical(acc=RED):
    """Medical cross badge with heartbeat line. 420x420."""
    c = Cv(420, 420)
    c.shadow(210, 400, 150, 12)
    c.fill(circ(210, 200, 175), c.rad((160, 150), 260, (255, 255, 255), (170, 180, 205)))
    c.stroke(circ(210, 200, 175), (120, 130, 160), 3)
    cross = np.array([(170, 80), (250, 80), (250, 160), (330, 160), (330, 240), (250, 240), (250, 320),
                      (170, 320), (170, 240), (90, 240), (90, 160), (170, 160)])
    c.fill(cross + (8, 8), dk(acc, 0.6))
    c.fill(cross, c.lin((0, 80), (0, 320), lt(acc, 0.25), dk(acc, 0.2)))
    c.fill(cross, WHITE, alpha=c.lin_a((90, 80), (250, 250), 0.3, 0.0))
    ekg = [(60, 205), (150, 205), (170, 170), (195, 250), (220, 140), (240, 205), (360, 205)]
    c.stroke([ekg], WHITE, 6, closed=False, glow=1.0)
    c.stroke(circ(210, 200, 175), lt(acc, 0.3), 3, 0.0, glow=1.0)
    return c.done()


@lru_cache(maxsize=64)
def rate_gauge(angle=270):
    """Interest-rate dial with needle; angle 180 (left/low) .. 360 (right/high). 520x340."""
    c = Cv(520, 340)
    cx, cy = 260, 280
    c.shadow(cx, 318, 230, 12)
    c.fill(np.vstack([ell(cx, cy, 240, 240, a0=180, a1=360), [(500, 300), (20, 300)]]),
           c.lin((0, 40), (0, 300), (70, 80, 115), (20, 24, 40)))
    c.fill(np.vstack([ell(cx, cy, 215, 215, a0=180, a1=360), [(475, 280), (45, 280)]]),
           c.lin((0, 60), (0, 280), (14, 20, 44), (6, 8, 20)))
    for i in range(36):
        a0, a1 = 180 + i * 5, 185 + i * 5
        t = i / 35
        colr = mix(GREEN, GOLD, t * 2) if t < 0.5 else mix(GOLD, RED, (t - 0.5) * 2)
        seg = np.vstack([ell(cx, cy, 200, 200, a0=a0, a1=a1, n=6), ell(cx, cy, 168, 168, a0=a1, a1=a0, n=6)])
        c.fill(seg, colr)
        c.glow_fill(seg, colr, 0.5)
    for i in range(10):
        a = math.radians(180 + i * 20)
        c.stroke([[(cx + math.cos(a) * 160, cy + math.sin(a) * 160), (cx + math.cos(a) * 140, cy + math.sin(a) * 140)]],
                 WHITE, 3, 0.8, closed=False)
    c.text("%", cx, cy - 70, 64, WHITE, alpha=0.9)
    a = math.radians(angle)
    tip = (cx + math.cos(a) * 175, cy + math.sin(a) * 175)
    n = np.array([tip, (cx + math.cos(a + 1.57) * 12, cy + math.sin(a + 1.57) * 12),
                  (cx - math.cos(a) * 24, cy - math.sin(a) * 24), (cx + math.cos(a - 1.57) * 12, cy + math.sin(a - 1.57) * 12)])
    c.fill(n, c.lin((cx, cy), tip, (255, 255, 255), (255, 80, 90)))
    c.glow_fill(n, RED, 0.8)
    c.fill(circ(cx, cy, 24), c.rad((cx - 8, cy - 8), 30, (240, 244, 255), (80, 90, 120)))
    c.stroke(np.vstack([ell(cx, cy, 240, 240, a0=180, a1=360), [(500, 300), (20, 300)]]), lt(GOLD, 0.2), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def coin(symbol="R"):
    """Gold coin (rand) with milled edge and embossed symbol. 380x380."""
    c = Cv(380, 380)
    g1, g2 = (255, 236, 150), (165, 105, 10)
    cx, cy, rx, ry, th = 190, 170, 160, 150, 28
    c.shadow(cx, 352, 150, 14)
    edge = np.vstack([ell(cx, cy, rx, ry, a0=0, a1=180), ell(cx, cy + th, rx, ry, a0=180, a1=0)])
    c.fill(ell(cx, cy + th, rx, ry), c.lin((cx - rx, 0), (cx + rx, 0), dk(g2, 0.1), dk(g2, 0.5)))
    for i in range(40):
        a = math.radians(5 + i * 4.3)
        x = cx + rx * math.cos(a)
        y = cy + ry * math.sin(a)
        c.stroke([[(x, y), (x, y + th)]], dk(g2, 0.55), 2, 0.6, closed=False)
    c.fill(ell(cx, cy, rx, ry), c.rad((cx - 60, cy - 60), rx * 1.6, g1, g2))
    c.stroke(ell(cx, cy, rx * 0.84, ry * 0.84), dk(g2, 0.2), 3, 0.7)
    c.text(symbol, cx + 5, cy + 7, 170, dk(g2, 0.45), alpha=0.7)
    c.text(symbol, cx, cy, 170, c.rad((cx - 40, cy - 60), 200, lt(g1, 0.3), g2) if False else lt(g1, 0.15))
    c.fill(ell(cx - 60, cy - 70, 50, 24, -35), WHITE, 0.4)
    sparkle(c, cx + 110, cy - 105, 20)
    c.stroke([ell(cx, cy, rx, ry), edge], lt(GOLD, 0.3), 3, glow=1.0)
    return c.done()


_AFRICA = [(-17, 21), (-16, 12), (-12, 7), (-8, 4.5), (-2, 5), (5, 6), (9, 4), (9.5, 1), (12, -5), (13, -12),
           (12, -18), (15, -27), (18, -34.5), (20, -35), (26, -34), (32, -29), (35, -24), (35, -20), (40, -15),
           (40, -10), (39, -5), (42, 0), (46, 2), (51, 11), (43, 12), (39, 16), (35, 23), (33, 29), (32, 31),
           (25, 32), (20, 31), (11, 33), (10, 37), (3, 37), (-6, 35.8), (-10, 30), (-13, 27)]


@lru_cache(maxsize=2)
def globe():
    """Shaded globe centred on Africa, South Africa highlighted. 400x420."""
    c = Cv(400, 420)
    cx, cy, R = 200, 195, 180
    lon0, lat0 = 20, -5
    c.shadow(cx, 400, 140, 12)

    def proj(lo, la):
        lo, la, l0, p0 = map(math.radians, (lo, la, lon0, lat0))
        x = math.cos(la) * math.sin(lo - l0)
        y = math.cos(p0) * math.sin(la) - math.sin(p0) * math.cos(la) * math.cos(lo - l0)
        return cx + x * R, cy - y * R

    sph = c.fill(circ(cx, cy, R), c.rad((cx - 60, cy - 70), R * 1.6, (60, 150, 240), (6, 24, 70)))
    for lat in range(-60, 90, 30):
        c.stroke([[proj(lo, lat) for lo in range(lon0 - 89, lon0 + 90, 3)]], WHITE, 1.5, 0.25, closed=False, clip=sph)
    for lo in range(lon0 - 90, lon0 + 91, 30):
        c.stroke([[proj(lo, la) for la in range(-89, 90, 3)]], WHITE, 1.5, 0.25, closed=False, clip=sph)
    af = np.array([proj(lo, la) for lo, la in _AFRICA])
    c.fill(af, c.lin((0, cy - R), (0, cy + R), (90, 200, 120), (20, 100, 60)), clip=sph)
    try:
        geo = json.load(open(f"{DIR}/assets/ZAF.geo.json"))["features"][0]["geometry"]["coordinates"]
        za = np.array([proj(lo, la) for lo, la in geo[0]])
        c.fill(za, (255, 200, 60))
        c.glow_fill(za, GOLD, 1.0)
    except Exception:
        pass
    c.paint(sph, (4, 8, 30), alpha=c.rad_a((cx - 50, cy - 60), R * 1.3, 0.0, 0.6))
    c.fill(ell(cx - 70, cy - 90, 60, 30, -35), WHITE, 0.35)
    c.stroke(circ(cx, cy, R), lt(CYAN, 0.2), 3, glow=1.0)
    return c.done()


@lru_cache(maxsize=4)
def school(acc=GOLD):
    """School building with bell tower, clock and flag. 600x480."""
    c = Cv(600, 480)
    br1, br2 = (210, 110, 80), (110, 50, 36)
    c.shadow(300, 455, 270, 14)
    c.fill(rect(60, 250, 540, 440), c.lin((0, 250), (0, 440), br1, br2))
    for y in range(262, 440, 16):
        c.stroke([[(60, y), (540, y)]], dk(br2, 0.3), 1.2, 0.35, closed=False)
    c.fill([(40, 255), (300, 200), (560, 255)], c.lin((0, 200), (0, 255), (90, 100, 130), (40, 46, 70)))
    c.fill(rect(240, 150, 360, 440), c.lin((0, 150), (0, 440), lt(br1, 0.1), br2))
    c.fill([(225, 155), (300, 70), (375, 155)], c.lin((0, 70), (0, 155), (100, 110, 140), (40, 46, 70)))
    c.fill(circ(300, 200, 30), c.rad((292, 192), 36, (255, 255, 255), (200, 205, 220)))
    c.stroke([[(300, 200), (300, 180)], [(300, 200), (315, 208)]], NAVY, 3, closed=False)
    c.fill(rrect(270, 350, 330, 440, 26), c.lin((0, 350), (0, 440), (120, 70, 40), (60, 30, 16)))
    for x in (90, 150, 390, 450):
        for y in (290, 360):
            c.fill(rect(x, y, x + 44, y + 44), (255, 212, 130))
            c.glow_fill(rect(x, y, x + 44, y + 44), GOLD, 0.5)
            c.stroke([[(x + 22, y), (x + 22, y + 44)], [(x, y + 22), (x + 44, y + 22)]], WHITE, 2.5, closed=False)
    c.stroke([[(300, 72), (300, 20)]], (210, 215, 230), 3, closed=False)
    c.fill([(302, 22), (345, 32), (302, 42)], acc)
    c.stroke([rect(60, 250, 540, 440), rect(240, 150, 360, 440)], lt(acc, 0.3), 2.5, glow=1.0)
    return c.done()


@lru_cache(maxsize=2)
def handshake_globe():
    """Two shaded arrows passing between globes (trade / international deals). 600x380."""
    c = Cv(600, 380)
    for x, colr in ((130, GREEN), (470, BLUE)):
        c.fill(circ(x, 190, 110), c.rad((x - 35, 150), 180, lt(colr, 0.45), dk(colr, 0.65)))
        for lat in (-60, -30, 0, 30, 60):
            c.stroke(ell(x, 190 - lat * 1.5, 110 * math.cos(math.radians(lat)), 12), WHITE, 1.5, 0.3, closed=False)
        c.stroke(ell(x, 190, 40, 110), WHITE, 1.5, 0.3)
        c.fill(ell(x - 40, 140, 30, 16, -30), WHITE, 0.4)
        c.stroke(circ(x, 190, 110), lt(colr, 0.3), 3, glow=1.0)
    for y, d, colr in ((120, 1, GOLD), (260, -1, CYAN)):
        x0, x1 = (230, 370) if d > 0 else (370, 230)
        shaft = rect(min(x0, x1), y - 10, max(x0, x1), y + 10)
        head = [(x1, y - 26), (x1 + d * 34, y), (x1, y + 26)]
        c.fill(shaft, c.lin((0, y - 10), (0, y + 10), lt(colr, 0.3), dk(colr, 0.3)))
        c.fill(head, lt(colr, 0.1))
        c.glow_fill(shaft, colr, 0.8)
        c.glow_fill(head, colr, 0.8)
    return c.done()


@lru_cache(maxsize=8)
def sphere(color=BLUE, label=""):
    """Shaded globe-ball with optional label (country codes etc). 260x260."""
    c = Cv(260, 260)
    cx = cy = 130
    m = c.fill(circ(cx, cy, 110), c.rad((cx - 40, cy - 45), 190, lt(color, 0.45), dk(color, 0.7)))
    for lat in (-60, -30, 0, 30, 60):
        c.stroke(ell(cx, cy - lat * 1.5, 110 * math.cos(math.radians(lat)), 12), WHITE, 1.5, 0.25, clip=m)
    c.stroke(ell(cx, cy, 45, 110), WHITE, 1.5, 0.25, clip=m)
    c.stroke(ell(cx, cy, 85, 110), WHITE, 1.5, 0.2, clip=m)
    if label:
        c.text(label, cx + 3, cy + 4, 70, (0, 0, 0), alpha=0.4)
        c.text(label, cx, cy, 70, WHITE)
    c.fill(ell(cx - 42, cy - 55, 34, 18, -30), WHITE, 0.4)
    c.stroke(circ(cx, cy, 110), lt(color, 0.3), 3, glow=1.0)
    return c.done()


# registry used by icon_gallery.py
GALLERY = [
    ("ballot_box", lambda: ballot_box()), ("fuel_pump", lambda: fuel_pump()), ("arrow3d up", lambda: arrow3d(True)),
    ("arrow3d down", lambda: arrow3d(False)), ("parliament", lambda: parliament()), ("medal", lambda: medal()),
    ("trophy", lambda: trophy()), ("storm_cloud", lambda: storm_cloud()), ("sun", lambda: sun()),
    ("tap_drop", lambda: tap_drop()), ("soccer_ball", lambda: soccer_ball()), ("rugby_ball", lambda: rugby_ball()),
    ("clapperboard", lambda: clapperboard(20)), ("bars3d", lambda: bars3d((380, 330, 80), (GREEN, BLUE, GOLD))),
    ("map_pin", lambda: map_pin()), ("skyline", lambda: skyline()), ("table_mountain", lambda: table_mountain()),
    ("gavel", lambda: gavel(25)), ("police_light", lambda: police_light()), ("pylon", lambda: pylon()),
    ("bolt", lambda: bolt()), ("house", lambda: house()), ("medical", lambda: medical()),
    ("rate_gauge", lambda: rate_gauge(300)), ("coin", lambda: coin()), ("globe", lambda: globe()),
    ("school", lambda: school()), ("handshake_globe", lambda: handshake_globe()),
    ("sphere", lambda: sphere(GREEN, "ZA")),
]
