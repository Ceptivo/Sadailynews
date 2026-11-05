"""Neon news reel renderer: one encoded file per scene."""
import json, math, os, subprocess, sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import DIR, OUT, FPS, ORDER, EPISODE, Y_SHIFT, caption_times, timeline
import icons as IC

W, H = 1080, 1920
AA = cv2.LINE_AA
TL = timeline()

F_BLACK = f"{DIR}/fonts/InterDisplay-Black.otf"
F_XB = f"{DIR}/fonts/InterDisplay-ExtraBold.otf"
F_BOLD = f"{DIR}/fonts/Inter-Bold.otf"
F_SEMI = f"{DIR}/fonts/Inter-SemiBold.otf"
F_MONO = f"{DIR}/fonts/DejaVuSansMono-Bold.ttf"

CYAN = (0, 235, 255)
MAG = (255, 40, 190)
GOLD = (255, 186, 30)
GREEN = (0, 215, 120)
RED = (255, 50, 70)
BLUE = (60, 110, 255)
WHITE = (255, 255, 255)

ACCENT = {"intro": CYAN, "vote": GOLD, "agoa": MAG, "water": CYAN, "film": RED, "rugby": GREEN, "outro": MAG}


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(s, t0, d):
    return clamp((s - t0) / d)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_back(x):
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2 if x > 0 else 0.0


def col(c, a):
    a = clamp(a, 0, 2)
    return tuple(int(min(255, v * a)) for v in c)


# ---------- sprites ----------
@lru_cache(maxsize=512)
def text_sprite(text, font, size, color=WHITE, track=0):
    f = ImageFont.truetype(font, size)
    if track:
        widths = [f.getbbox(ch)[2] if ch != " " else size * 0.28 for ch in text]
        tw = int(sum(widths) + track * (len(text) - 1)) + 8
    else:
        bb = f.getbbox(text)
        tw = bb[2] + 8
    asc, desc = f.getmetrics()
    th = asc + desc + 6
    im = Image.new("L", (max(tw, 1), th), 0)
    d = ImageDraw.Draw(im)
    if track:
        x = 0
        for ch, w_ in zip(text, widths):
            d.text((x, 2), ch, font=f, fill=255)
            x += w_ + track
    else:
        d.text((0, 2), text, font=f, fill=255)
    a = np.array(im)
    rgba = np.zeros((th, tw, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = a
    return rgba


@lru_cache(maxsize=128)
def pill_sprite(w, h, r, fill=None, outline=None, ow=3):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((ow // 2, ow // 2, w - 1 - ow // 2, h - 1 - ow // 2), r,
                        fill=fill, outline=outline, width=ow if outline else 0)
    return np.array(im)


def blit(dst, spr, x, y, a=1.0, xw=None):
    """Alpha-over sprite at (x, y). xw crops sprite width (wipe reveal)."""
    if a <= 0.003:
        return
    if xw is not None:
        spr = spr[:, :max(0, int(xw))]
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    al = s[..., 3:4].astype(np.float32) * (a / 255.0)
    reg = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (reg * (1 - al) + s[..., :3] * al).astype(np.uint8)


def blit_scaled(dst, spr, cx, cy, scale, a=1.0):
    if scale <= 0.01:
        return
    h, w = spr.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    s = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blit(dst, s, cx - nw / 2, cy - nh / 2, a)


# ---------- logo (from assets/logo_source.png) ----------
_LOGO = {k: np.array(Image.open(f"{DIR}/assets/logo_{k}.png").convert("RGBA")) for k in ("map", "wordmark")}


@lru_cache(maxsize=16)
def logo_sprite(name, height):
    a = _LOGO[name]
    w = int(round(a.shape[1] * height / a.shape[0]))
    return cv2.resize(a, (w, height), interpolation=cv2.INTER_AREA)


# ---------- static assets ----------
def make_bg():
    y = np.linspace(0, 1, H)[:, None]
    top = np.array([7, 11, 26], float)
    bot = np.array([9, 14, 34], float)
    g = top + (bot - top) * y[..., None]
    g = np.repeat(g, W, axis=1)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy, r, c in [(120, 380, 760, (16, 40, 95)), (980, 1150, 820, (10, 45, 80)), (540, 1850, 700, (30, 20, 70))]:
        d = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (r ** 2)))
        g += d[..., None] * np.array(c, float) * 0.6
    grid = ((xx % 54) == 0) | ((yy % 54) == 0)
    g[grid] += 5
    vign = 1 - 0.35 * (((xx - W / 2) / (W * 0.8)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2)
    g *= np.clip(vign, 0.5, 1)[..., None]
    return np.clip(g, 0, 255).astype(np.uint8)


BG = make_bg()
# scenes are drawn on a background raised by Y_SHIFT, then the frame is moved down by Y_SHIFT
BG_R = np.vstack([BG[Y_SHIFT:], np.repeat(BG[-1:], Y_SHIFT, axis=0)]) if Y_SHIFT else BG

rng = np.random.default_rng(7)
PART = np.column_stack([rng.uniform(0, W, 34), rng.uniform(0, H, 34), rng.uniform(8, 30, 34),
                        rng.uniform(1, 2.2, 34), rng.uniform(0, 6.28, 34)])

# South Africa outline (Natural Earth via world.geo.json), projected
_geo = json.load(open(f"{DIR}/assets/ZAF.geo.json"))["features"][0]["geometry"]["coordinates"]


def project(lon, lat, cx, cy, width):
    k = math.cos(math.radians(29))
    x0, x1, y0, y1 = 16.4 * k, 32.95 * k, -22.1, -34.85
    s = width / (x1 - x0)
    return cx + (lon * k - (x0 + x1) / 2) * s, cy + (-(lat) - (-(y0 + y1) / 2)) * s


def map_polys(cx, cy, width):
    out = []
    for ring in _geo:
        out.append(np.array([project(lo, la, cx, cy, width) for lo, la in ring], np.float32))
    return out


CITIES = [("JHB", 28.05, -26.2), ("CPT", 18.42, -33.92), ("DBN", 31.03, -29.86)]

TICKER = ("SPECIAL VOTING DAY 2   ■   LOCAL ELECTIONS WED 4 NOV   ■   US MOVES TO DROP SA FROM AGOA   ■   "
          "TSHWANE: BRONKHORSTSPRUIT TAP WATER STILL UNSAFE   ■   FILM INCENTIVE TALKS COLLAPSE AGAIN   ■   "
          "RASSIE NAMES 6 UNCAPPED BOKS   ■   ")
NAVY = (10, 16, 40)


# ---------- drawing helpers ----------
def poly_partial(img, pts, frac, color, th):
    if frac <= 0:
        return
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum() * frac
    acc, out = 0.0, [pts[0]]
    for i, l in enumerate(seg):
        if acc + l >= L:
            r = (L - acc) / max(l, 1e-6)
            out.append(pts[i] + (pts[i + 1] - pts[i]) * r)
            break
        acc += l
        out.append(pts[i + 1])
    p = (np.array(out) * 4).astype(np.int32)
    cv2.polylines(img, [p], False, color, th, AA, shift=2)
    return out[-1]


# logo colours around the outline, by compass angle from the map centre (image coords, degrees)
FLAG_STOPS = [(-180, (255, 196, 0)), (-125, (255, 170, 0)), (-75, (235, 30, 40)), (-20, (235, 30, 40)),
              (25, (0, 140, 255)), (85, (0, 150, 255)), (130, (0, 200, 90)), (160, (0, 200, 90)),
              (180, (255, 196, 0))]


def flag_color(ang):
    for (a0, c0), (a1, c1) in zip(FLAG_STOPS, FLAG_STOPS[1:]):
        if a0 <= ang <= a1:
            u = (ang - a0) / (a1 - a0)
            return tuple(c0[i] + (c1[i] - c0[i]) * u for i in range(3))
    return FLAG_STOPS[-1][1]


def poly_partial_flag(img, pts, frac, th, gain=1.0):
    """Like poly_partial, but each segment takes the logo's flag colour for its position."""
    if frac <= 0:
        return
    ctr = pts.mean(axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum() * frac
    acc = 0.0
    for i, l in enumerate(seg):
        a, b = pts[i], pts[i + 1]
        if acc + l >= L:
            b = a + (b - a) * ((L - acc) / max(l, 1e-6))
        m = (a + b) / 2 - ctr
        c = col(flag_color(math.degrees(math.atan2(m[1], m[0]))), gain)
        cv2.line(img, tuple((a * 4).astype(int)), tuple((b * 4).astype(int)), c, th, AA, shift=2)
        acc += l
        if acc >= L:
            break


def iline(img, a, b, c, th):
    cv2.line(img, (int(a[0] * 4), int(a[1] * 4)), (int(b[0] * 4), int(b[1] * 4)), c, th, AA, shift=2)


def icircle(img, c, r, color, th):
    cv2.circle(img, (int(c[0] * 4), int(c[1] * 4)), int(r * 4), color, th, AA, shift=2)


def hud_ring(neon, cx, cy, r, t, c, a=1.0):
    a *= 0.55
    for i in range(4):
        st = (t * 25 + i * 90) % 360
        cv2.ellipse(neon, (cx, cy), (r, r), 0, st, st + 50, col(c, 0.5 * a), 2, AA)
    for i in range(72):
        ang = math.radians(i * 5 - t * 8)
        r1, r2 = r + 14, r + (26 if i % 6 == 0 else 19)
        iline(neon, (cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
              (cx + math.cos(ang) * r2, cy + math.sin(ang) * r2), col(WHITE, 0.16 * a), 1)


def draw_background(base, neon, T, accent):
    for x, y, sp, r, ph0 in PART:
        yy = (y - T * sp) % H
        tw = 0.18 + 0.12 * math.sin(T * 1.6 + ph0)
        icircle(base, (x + math.sin(T * 0.5 + ph0) * 10, yy), r, col((170, 200, 255), tw), -1)


def card(base, accent, s, y0=560, y1=1062):
    p = ease_out(prog(s, 0.15, 0.5))
    if p <= 0:
        return
    sp = pill_sprite(960, y1 - y0, 28, (255, 255, 255, 11), (255, 255, 255, 30), 2)
    blit(base, sp, 60, y0 + (1 - p) * 40, p)


def card_accent(neon, accent, s, y0=560):
    p = ease_out(prog(s, 0.3, 0.6))
    if p > 0:
        iline(neon, (88, y0 + 1), (88 + 180 * p, y0 + 1), col(accent, 0.9), 3)


def draw_hud(neon, crisp, T, accent, scene_idx, s, dur):
    # segmented progress
    n = len(ORDER)
    gap, x0, x1, y = 8, 70, W - 70, 50
    sw = (x1 - x0 - gap * (n - 1)) / n
    for i in range(n):
        xa = x0 + i * (sw + gap)
        cv2.rectangle(crisp, (int(xa), y), (int(xa + sw), y + 4), (60, 68, 95), -1)
        f = 1 if i < scene_idx else (s / dur if i == scene_idx else 0)
        if f > 0:
            cv2.rectangle(crisp, (int(xa), y), (int(xa + sw * f), y + 4), (240, 244, 255), -1)
    # channel bug
    blit(crisp, logo_sprite("map", 70), 66, 88)
    cv2.rectangle(crisp, (150, 92), (151, 172), (235, 238, 245), -1)
    blit(crisp, text_sprite("SA DAILY BRIEF", F_BLACK, 42, WHITE), 168, 92)
    blit(crisp, text_sprite(f"NEWS  ·  {EPISODE['hud_date']}", F_SEMI, 22, (165, 178, 205), track=1), 169, 146)
    # LIVE
    blit(crisp, pill_sprite(142, 50, 8, (230, 30, 50, 255)), W - 212, 98)
    on = (math.sin(T * 5) > -0.3)
    cv2.circle(crisp, (W - 184, 123), 8, (255, 255, 255) if on else (255, 150, 160), -1, AA)
    blit(crisp, text_sprite("LIVE", F_BLACK, 27, WHITE, track=2), W - 166, 106)
    # thin divider with a soft neon sweep
    cv2.rectangle(crisp, (70, 204), (W - 70, 205), (40, 50, 80), -1)
    sx = 70 + ((T * 300) % (W - 140))
    iline(neon, (max(70, sx - 80), 205), (min(W - 70, sx + 80), 205), col(CYAN, 0.7), 2)


def draw_ticker(crisp, neon, T):
    y0, h = 1488, 66
    blit(crisp, pill_sprite(W, h, 1, (244, 246, 251, 255)), 0, y0)
    strip = text_sprite(TICKER, F_BOLD, 28, NAVY)
    sw = strip.shape[1]
    off = int(T * 140) % sw
    view_w = W - 236
    ext = np.concatenate([strip, strip], axis=1)[:, off:off + view_w]
    blit(crisp, ext, 236, y0 + 14)
    blit(crisp, pill_sprite(222, h, 1, (226, 28, 48, 255)), 0, y0)
    blit(crisp, text_sprite("BREAKING", F_BLACK, 28, WHITE, track=2), 34, y0 + 14)
    cv2.rectangle(crisp, (0, y0 - 6), (W, y0 - 1), NAVY, -1)
    iline(neon, (0, y0 - 4), (W, y0 - 4), col(CYAN, 0.35), 1)


def draw_caption(crisp, key, info, s):
    for text, t0, t1 in caption_times(key, info):
        if t0 <= s < t1:
            spr = text_sprite(text, F_BOLD, 50, WHITE)
            w = spr.shape[1]
            p = ease_out(prog(s, t0, 0.16))
            box = pill_sprite(w + 52, 82, 14, (6, 10, 26, 215), None)
            blit_scaled(crisp, box, W / 2, 1390 + (1 - p) * 12, 1, p)
            blit_scaled(crisp, spr, W / 2, 1390 + (1 - p) * 12, 1, p)


def tag_text_color(accent):
    return WHITE if accent in (MAG, RED, BLUE) else NAVY


def story_header(crisp, neon, s, num, tag, lines, accent):
    p = ease_out(prog(s, 0.1, 0.4))
    ts = text_sprite(f"{tag}", F_BLACK, 27, tag_text_color(accent), track=3)
    ns = text_sprite(f"{num:02d}", F_BLACK, 27, WHITE)
    pw = ts.shape[1] + 40
    x = 80 - (1 - p) * 200
    blit(crisp, pill_sprite(70, 52, 6, (255, 255, 255, 40)), x, 250, p)
    blit(crisp, ns, x + 17, 259, p)
    blit(crisp, pill_sprite(pw, 52, 6, accent + (255,)), x + 78, 250, p)
    blit(crisp, ts, x + 98, 259, p)
    for i, ln in enumerate(lines):
        spr = text_sprite(ln, F_BLACK, 82, WHITE)
        q = prog(s, 0.3 + i * 0.22, 0.45)
        if q <= 0:
            continue
        xw = spr.shape[1] * ease_out(q)
        y = 330 + i * 96
        blit(crisp, spr, 80, y, 1, xw=xw)
        if q < 1:
            cv2.rectangle(crisp, (int(80 + xw), y + 16), (int(80 + xw + 10), y + 92), accent, -1)


def chip(crisp, text, x, y, s, t0, color, size=30, fill=(14, 20, 44, 235)):
    p = ease_back(prog(s, t0, 0.35))
    if p <= 0:
        return 0
    ts = text_sprite(text, F_BOLD, size, WHITE, track=1)
    w, h = ts.shape[1] + 46, ts.shape[0] + 22
    sp = pill_sprite(w, h, 10, fill, color + (210,), 2)
    blit_scaled(crisp, sp, x + w / 2, y + h / 2, p, min(1, p))
    blit_scaled(crisp, ts, x + w / 2, y + h / 2 + 2, p, min(1, p))
    return w


def big_stat(crisp, neon, text, s, t0, color, y=1130, size=128, label=None):
    p = ease_out(prog(s, t0, 0.4))
    if p <= 0:
        return
    spr = text_sprite(text, F_BLACK, size, WHITE)
    sh = text_sprite(text, F_BLACK, size, (0, 0, 0))
    x = 104 + (1 - p) * 40
    h = spr.shape[0]
    cv2.rectangle(crisp, (80, int(y + 18)), (90, int(y + 18 + (h - 30) * p)), color, -1)
    blit(crisp, sh, x + 4, y + 5, 0.45 * p)
    blit(crisp, spr, x, y, p)
    if label:
        ls = text_sprite(label, F_SEMI, 26, (180, 192, 218), track=1)
        blit(crisp, ls, 106, y + h - 2, p)


# ---------- scenes ----------
def title_block(crisp, s, t0, height=290, y=205):
    p = ease_out(prog(s, t0, 0.5))
    if p <= 0:
        return
    wm = logo_sprite("wordmark", height)
    blit(crisp, wm, 540 - wm.shape[1] / 2, y + (1 - p) * 30, p)

def sc_intro(neon, crisp, s, T, info):
    polys = map_polys(540, 900, 780)
    f = ease_out(prog(s, 0.15, 2.4))
    for i, pts in enumerate(polys):
        if i == 0:
            poly_partial_flag(neon, pts, f, 5)
        else:
            poly_partial(neon, pts, f, col(GOLD, 1.0), 5)
        if f >= 1:
            # inner echo
            ctr = pts.mean(axis=0)
            e = ctr + (pts - ctr) * 0.94
            poly_partial(neon, e, 1, col(CYAN, 0.25 + 0.1 * math.sin(T * 4)), 2)
    hud_ring(neon, 540, 900, 470, T, CYAN, prog(s, 0.0, 1.0))
    for j, (nm, lo, la) in enumerate(CITIES):
        q = prog(s, 1.5 + j * 0.25, 0.3)
        if q <= 0:
            continue
        x, y = project(lo, la, 540, 900, 780)
        icircle(neon, (x, y), 8, col(GOLD, 1.3 * q), -1)
        for k in range(2):
            rr = ((s * 0.9 + k * 0.5) % 1.0)
            icircle(neon, (x, y), 10 + rr * 60, col(GOLD, (1 - rr) * 0.9 * q), 2)
        blit(crisp, text_sprite(nm, F_MONO, 24, WHITE), x + 16, y - 34, q)
    # title
    title_block(crisp, s, 0.25)
    q = prog(s, 0.8, 0.5)
    sub = text_sprite(f"YOUR 60-SECOND NEWS UPDATE  ·  {EPISODE['short_date']}", F_BOLD, 30, (200, 210, 230), track=2)
    blit(crisp, sub, 540 - sub.shape[1] / 2, 512, q, xw=sub.shape[1] * ease_out(q))


# ---------- detailed icons (icons.py) ----------
def place_icon(neon, crisp, ic, cx, cy, scale=1.0, a=1.0, angle=0.0, glow=1.0):
    """Draw an icons.Icon centred at (cx, cy): solid body on the crisp layer, rim light into the neon (glow) layer."""
    if scale <= 0.01 or a <= 0.003:
        return
    body, g = ic.body, ic.glow
    if angle:
        h, w = body.shape[:2]
        d = int(math.hypot(w, h)) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        M[0, 2] += (d - w) / 2
        M[1, 2] += (d - h) / 2
        body = cv2.warpAffine(body, M, (d, d), flags=cv2.INTER_LINEAR)
        g = cv2.warpAffine(g, M, (d, d), flags=cv2.INTER_LINEAR)
    blit_scaled(crisp, body, cx, cy, scale, a)
    h, w = g.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    gs = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + nw, W), min(y0 + nh, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = gs[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32) * (a * glow)
    neon[Y0:Y1, X0:X1] = np.maximum(neon[Y0:Y1, X0:X1], np.clip(sub, 0, 255).astype(np.uint8))


def pop(s, t0=0.3, d=0.55):
    """Pop-in scale (0 -> overshoot -> 1) for icons."""
    return ease_back(prog(s, t0, d))


ICON_Y = 815  # centre of the card area (card spans y 560-1062)


def ripples(neon, cx, cy, s, color, n=3, rx=260, ry=50):
    for k in range(n):
        rr = ((s * 0.6 + k / n) % 1.0)
        cv2.ellipse(neon, (int(cx), int(cy)), (int(30 + rr * rx), int(8 + rr * ry)), 0, 0, 360,
                    col(color, (1 - rr) * 0.8), 3, AA)


def sc_vote(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 1, "LOCAL ELECTIONS 2026", ["Special voting", "wraps up today"], GOLD)
    p = pop(s)
    hud_ring(neon, 540, ICON_Y, 300, T, GOLD, 0.45 * p)
    place_icon(neon, crisp, IC.ballot_box(GOLD, "VOTE"), 540, ICON_Y + 4 * math.sin(T * 2), 0.85 * p, min(1, p * 2))
    chip(crisp, "ELECTION DAY  ·  WED 4 NOV", 80, 1060, s, 2.6, GOLD)
    t0 = info["vo_start"] + 4.4
    q = prog(s, t0, 1.6)
    if q > 0:
        n = int(700000 * ease_out(q) / 1000) * 1000
        big_stat(crisp, neon, f"{n:,}+", s, t0, GOLD, y=1140, size=118, label="VOTES CAST ON SPECIAL VOTING DAY 1 · IEC")



def sc_agoa(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 2, "TRADE", ["US moves to drop", "SA from AGOA"], MAG)
    a, b = np.array([250., 860.]), np.array([830., 860.])
    pa = ease_back(prog(s, 0.4, 0.5))
    for c, pt, lab, cc in [(GREEN, a, "ZA", GOLD), (BLUE, b, "US", CYAN)]:
        if pa > 0:
            place_icon(neon, crisp, IC.sphere(c, lab), pt[0], pt[1], 0.85 * pa, min(1, pa * 2))
            for i in range(12):
                ang = math.radians(i * 30 + T * 50 * (1 if lab == "ZA" else -1))
                iline(neon, pt + np.array([math.cos(ang), math.sin(ang)]) * 122 * pa,
                      pt + np.array([math.cos(ang), math.sin(ang)]) * 136 * pa, col(cc, 0.6), 3)
    # arc
    tt = np.linspace(0, 1, 80)
    ctrl = np.array([540., 560.])
    arc = ((1 - tt) ** 2)[:, None] * (a + [90, -40]) + (2 * (1 - tt) * tt)[:, None] * ctrl + (tt ** 2)[:, None] * (b + [-90, -40])
    brk = info["vo_start"] + 1.6
    fa = ease_out(prog(s, 0.9, 1.0))
    broken = s >= brk
    if not broken:
        poly_partial(neon, arc.astype(np.float32), fa, col(CYAN, 1.0), 5)
        for k in range(5):
            u = ((s * 0.6 + k / 5) % 1.0)
            if u < fa:
                i = int(u * 79)
                icircle(neon, arc[i], 9, col(GOLD, 1.4), -1)
    else:
        q = prog(s, brk, 0.6)
        shake = (1 - q) * 18
        left, right = arc[:34].copy(), arc[46:].copy()
        left[-12:, 1] += np.linspace(0, 40 * ease_out(q), 12)
        right[:12, 1] += np.linspace(40 * ease_out(q), 0, 12)
        poly_partial(neon, left.astype(np.float32), 1, col(RED, 1.0), 5)
        poly_partial(neon, right.astype(np.float32), 1, col(RED, 1.0), 5)
        cx, cy = 540 + np.random.default_rng(int(s * 60)).uniform(-shake, shake), 640
        r = 54 * ease_back(q)
        iline(neon, (cx - r, cy - r), (cx + r, cy + r), col(RED, 1.5), 12)
        iline(neon, (cx + r, cy - r), (cx - r, cy + r), col(RED, 1.5), 12)
        icircle(neon, (cx, cy), 90 * q + 20, col(RED, (1 - q) * 1.2), 4)
        for k in range(14):
            ang = k * 0.45
            d = 30 + 220 * ease_out(q)
            icircle(neon, (cx + math.cos(ang) * d, cy + math.sin(ang) * d), 4, col(GOLD, 1 - q), -1)
    big_stat(crisp, neon, "01.01.2027", s, brk + 0.5, MAG, y=1120, size=118, label="SET EXIT DATE · USTR NOTICE, 2 NOV")
    chip(crisp, "SA GOVT: SUPPORT FOR AFFECTED SECTORS", 80, 1040, s, info["vo_start"] + 6.4, GREEN, size=26)


def drop_pts(cx, cy, r):
    pts = []
    for i in range(80):
        ang = math.pi / 2 + (i / 79) * 2 * math.pi
        th = (i / 79) * 2 * math.pi
        # teardrop: r*(1-sin)... parametric
        x = cx + r * math.sin(th) * (math.sin(th / 2) ** 1.0)
        y = cy - r * 1.35 * math.cos(th / 2) ** 2 + r * 0.55
        pts.append((x, y))
    return np.array(pts, np.float32)


def sc_water(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 3, "TSHWANE", ["Tap water still", "unsafe to drink"], CYAN)
    p = pop(s)
    place_icon(neon, crisp, IC.tap_drop(), 500, ICON_Y, 0.78 * p, min(1, p * 2))
    q = ease_back(prog(s, info["vo_start"] + 2.6, 0.4))
    if q > 0:
        bx, by, R = 800, 930, 80 * q
        pulse = 0.8 + 0.4 * (math.sin(T * 10) > 0)
        tri = np.array([[bx, by - R], [bx + R * 1.1, by + R * 0.8], [bx - R * 1.1, by + R * 0.8]], np.float32)
        cv2.fillPoly(neon, [(tri * 4).astype(np.int32)], col((60, 0, 8), 1), AA, shift=2)
        cv2.polylines(neon, [(tri * 4).astype(np.int32)], True, col(RED, 1.2 * pulse), 7, AA, shift=2)
        ex = text_sprite("!", F_BLACK, int(90 * q) + 1, WHITE)
        blit(crisp, ex, bx - ex.shape[1] / 2 + 4, by - ex.shape[0] / 2 + 10)
    t0 = info["vo_start"] + 2.9
    if s > t0:
        flash = 1.0 if (int((s - t0) * 4) % 2 == 0) else 0.75
        chip(crisp, "DO NOT DRINK  ·  BRONKHORSTSPRUIT", 80, 1100, s, t0, RED, size=34, fill=(int(70 * flash), 0, 12, 235))
    chip(crisp, "CITY OF TSHWANE ADVISORY", 80, 1195, s, t0 + 0.4, CYAN, size=26)


def sc_film(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 4, "FILM & TV", ["Incentive talks", "collapse again"], RED)
    p = pop(s, 0.25, 0.5)
    snaps = [info["vo_start"] + 0.2, info["vo_start"] + 1.4, info["vo_start"] + 2.9]
    ang, done = 28, 0
    for st in snaps:
        done += s >= st
        d = s - st
        if -0.35 < d < 0:
            ang = 28 * (-d / 0.35) ** 0.5
        elif 0 <= d < 0.4:
            ang = 0 if d < 0.12 else 28 * ease_out((d - 0.12) / 0.28)
    if s > snaps[-1]:
        ang = 0
    cx, cy = 540, ICON_Y + 10
    place_icon(neon, crisp, IC.clapperboard(int(round(ang)), f"TAKE {max(1, done)}", RED), cx, cy, 1.0 * p, min(1, p * 2))
    for st in snaps:
        d = s - st
        if 0 <= d < 0.25:
            icircle(neon, (cx - 180, cy - 90), 40 + d * 500, col(RED, (1 - d / 0.25) * 1.3), 5)
    hud_ring(neon, cx, cy, 330, T, RED, 0.35)
    big_stat(crisp, neon, "3RD", s, snaps[-1], RED, y=1150, size=118, label="BREAKDOWN IN INCENTIVE TALKS · COALITION")


def sc_rugby(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 5, "SPRINGBOKS", ["Rassie names six", "uncapped Boks"], GREEN)
    p = pop(s)
    cx, cy = 580, ICON_Y - 10
    for k in range(5):
        yy = cy - 100 + k * 50
        ln = 120 + 60 * math.sin(T * 8 + k)
        iline(neon, (max(110, cx - 330 - ln), yy), (cx - 300, yy), col(GREEN, 0.7 * p), 3)
    place_icon(neon, crisp, IC.rugby_ball(GREEN, GOLD), cx, cy, 1.2 * p, min(1, p * 2), angle=-25 + 8 * math.sin(T * 2.2))
    big_stat(crisp, neon, "6 UNCAPPED", s, info["vo_start"] + 2.3, GREEN, y=1110, size=104, label="37-MAN NATIONS CHAMPIONSHIP SQUAD")
    t0 = info["vo_start"] + 4.6
    x = 80
    for i, c in enumerate(["ITALY", "FRANCE", "IRELAND"]):
        w = chip(crisp, c, x, 1265 - 220, s, t0 + i * 0.45, GOLD, size=28)
        x += (w or 170) + 14


# ---- reusable scene library (adapt the text, register in SCENES). Every icon lives in icons.py ----
def sc_weather(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 2, "WEATHER", ["Storms and heat", "on voting day"], CYAN)
    p = pop(s)
    flash = 1.6 if (s > 0.8 and int((s - 0.8) * 3) % 3 == 0) else 0.8
    q = pop(s, info["vo_start"] + 4.6, 0.5)
    place_icon(neon, crisp, IC.sun(), 820, 700, 0.62 * q, min(1, q * 2), angle=T * 12)
    place_icon(neon, crisp, IC.storm_cloud(), 450, ICON_Y, 0.76 * p, min(1, p * 2), glow=flash)
    chip(crisp, "LEVEL 4 STORMS · MPUMALANGA & NW KZN", 80, 1100, s, info["vo_start"] + 1.2, GOLD, size=28)
    chip(crisp, "EXTREME HEAT · WESTERN & NORTHERN CAPE", 80, 1195, s, info["vo_start"] + 5.0, RED, size=28)


def sc_fuel(neon, crisp, s, T, info):
    # price increase: IC.fuel_pump(MAG, "+50c") and IC.arrow3d(up=True); decrease: arrow3d(up=False)
    story_header(crisp, neon, s, 3, "FUEL PRICES", ["Diesel down,", "petrol barely"], MAG)
    p = pop(s)
    place_icon(neon, crisp, IC.fuel_pump(MAG, "-1c"), 420, ICON_Y, 0.76 * p, min(1, p * 2))
    q = pop(s, info["vo_start"] + 1.4, 0.5)
    place_icon(neon, crisp, IC.arrow3d(up=False), 800, ICON_Y + 10 * math.sin(T * 5), 0.85 * q, min(1, q * 2))
    chip(crisp, "DIESEL: DOWN OVER R1 A LITRE", 80, 1100, s, info["vo_start"] + 1.6, GREEN, size=30)
    chip(crisp, "PETROL: DOWN 1c   ·   PARAFFIN: UP", 80, 1195, s, info["vo_start"] + 4.6, RED, size=28)


def sc_parliament(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 4, "PARLIAMENT", ["NCOP Secretary", "dies suddenly"], BLUE)
    p = ease_out(prog(s, 0.3, 0.9))
    hud_ring(neon, 540, ICON_Y, 330, T * 0.4, BLUE, 0.3 * p)
    place_icon(neon, crisp, IC.parliament(BLUE), 540, ICON_Y + (1 - p) * 40, 0.8, p)
    big_stat(crisp, neon, "SINCE 2010", s, info["vo_start"] + 3.8, BLUE, y=1110, size=104, label="SECRETARY TO THE NCOP · PARLIAMENT")


def sc_medal(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 5, "YOUTH OLYMPICS", ["Team SA's first", "gold in Dakar"], GREEN)
    p = pop(s)
    hud_ring(neon, 540, ICON_Y, 300, T, GREEN, 0.4 * p)
    place_icon(neon, crisp, IC.medal(GREEN, GOLD, "gold"), 540, ICON_Y - 10, 0.7 * p, min(1, p * 2), angle=4 * math.sin(T * 2))
    chip(crisp, "CALLUM DREYER · 16 · ROWING", 80, 1060, s, info["vo_start"] + 1.4, GOLD, size=28)
    big_stat(crisp, neon, "GOLD", s, info["vo_start"] + 3.0, GOLD, y=1140, size=110, label="COASTAL BEACH SPRINT · 2:15.65")


def sc_results(neon, crisp, s, T, info):
    """Election results / polls: growing 3D bars with % labels."""
    story_header(crisp, neon, s, 1, "ELECTION RESULTS", ["ANC ahead,", "DA close behind"], GOLD)
    bars = [("ANC", 32, GREEN), ("DA", 28, BLUE), ("MK", 6, GOLD)]
    q = ease_out(prog(s, 0.5, 1.4))
    k, cx, cy = 0.82, 540, ICON_Y + 10
    hs = tuple(int(v * 11 * q / 4) * 4 for _, v, _ in bars)
    ic = IC.bars3d(hs, tuple(c for _, _, c in bars))
    w = ic.size[0]
    place_icon(neon, crisp, ic, cx, cy, k)
    for i, ((nm, v, c), h) in enumerate(zip(bars, hs)):
        bx = cx + (60 + i * 200 + 65 - w / 2) * k
        top = cy + (470 - h - 260) * k
        lab = text_sprite(nm, F_BLACK, 40, WHITE)
        blit(crisp, lab, bx - lab.shape[1] / 2, cy + (480 - 260) * k, min(1, q * 2))
        pct = text_sprite(f"~{int(round(v * q))}%", F_BLACK, 44, WHITE)
        blit(crisp, pct, bx - pct.shape[1] / 2, top - 70, min(1, q * 2))
    chip(crisp, "~1/3 OF VOTING DISTRICTS COUNTED", 80, 1110, s, info["vo_start"] + 0.6, GOLD, size=28)
    chip(crisp, "RESULTS DUE SUNDAY 8 NOV · IEC", 80, 1205, s, info["vo_start"] + 6.0, CYAN, size=28)


def sc_capetown(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 2, "CAPE TOWN", ["DA claims", "early victory"], CYAN)
    p = ease_out(prog(s, 0.3, 0.9))
    place_icon(neon, crisp, IC.table_mountain(), 540, ICON_Y + (1 - p) * 30, 0.98, p)
    big_stat(crisp, neon, "~58%", s, info["vo_start"] + 4.6, CYAN, y=1120, size=124, label="DA'S OWN PROJECTION · NOT AN OFFICIAL RESULT")


def sc_pin(neon, crisp, s, T, info):
    """A town / municipality / place story: bouncing map pin."""
    story_header(crisp, neon, s, 3, "KWAZULU-NATAL", ["MK wins its", "first council"], MAG)
    p = pop(s)
    ripples(neon, 540, ICON_Y + 185, s, MAG, rx=240, ry=40)
    bounce = abs(math.sin(T * 2.4)) * 16
    place_icon(neon, crisp, IC.map_pin(MAG), 540, ICON_Y - 15 - bounce, 0.88 * p, min(1, p * 2))
    chip(crisp, "RICHMOND LOCAL MUNICIPALITY", 80, 1070, s, info["vo_start"] + 1.6, MAG, size=28)
    big_stat(crisp, neon, "56%", s, info["vo_start"] + 2.6, MAG, y=1150, size=110, label="MK PARTY · 10 OF 17 SEATS")


def sc_city(neon, crisp, s, T, info):
    """Metro / city story: skyline (Joburg-style tower)."""
    story_header(crisp, neon, s, 4, "GAUTENG METROS", ["DA leads Joburg,", "Tshwane tight"], BLUE)
    p = ease_out(prog(s, 0.3, 0.9))
    place_icon(neon, crisp, IC.skyline(BLUE), 540, ICON_Y - 30 + (1 - p) * 30, 0.95, p)
    t0 = info["vo_start"] + 1.4
    chip(crisp, "JOBURG · DA ~40%", 80, 1040, s, t0, BLUE, size=28)
    chip(crisp, "EKURHULENI · DA ~41%", 80, 1125, s, t0 + 0.6, BLUE, size=28)
    chip(crisp, "TSHWANE · ACTIONSA & ANC ~27% EACH", 80, 1210, s, info["vo_start"] + 3.8, GOLD, size=28)


def sc_soccer(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 5, "BAFANA BAFANA", ["Squad named", "for Kenya"], GREEN)
    p = pop(s)
    cx = 560
    cy = ICON_Y - 30 + 20 * abs(math.sin(T * 3))
    for k in range(5):
        yy = cy - 100 + k * 50
        ln = 90 + 50 * math.sin(T * 8 + k)
        iline(neon, (max(110, cx - 230 - ln), yy), (cx - 210, yy), col(GREEN, 0.7 * p), 3)
    place_icon(neon, crisp, IC.soccer_ball(), cx, cy, 1.05 * p, min(1, p * 2), angle=T * 70)
    big_stat(crisp, neon, "27-MAN SQUAD", s, info["vo_start"] + 2.0, GREEN, y=1050, size=100,
             label="AFCON 2027 QUALIFIERS · COACH PITSO MOSIMANE")
    x = 80
    for i, c in enumerate(["FRI 13 NOV · ORLANDO", "TUE 17 NOV · NAIROBI"]):
        w = chip(crisp, c, x, 1230, s, info["vo_start"] + 4.4 + i * 0.45, GOLD, size=26)
        x += (w or 330) + 14


def sc_court(neon, crisp, s, T, info):
    """Courts / judgments: gavel strikes on the voiceover beat."""
    story_header(crisp, neon, s, 2, "COURTS", ["Judgment handed", "down today"], GOLD)
    p = ease_out(prog(s, 0.3, 0.6))
    hit = info["vo_start"] + 1.5
    d = s - hit
    ang = 28 if d < -0.25 else (28 * (-d / 0.25) if d < 0 else (6 * math.exp(-d * 6) * abs(math.cos(d * 20))))
    place_icon(neon, crisp, IC.gavel(int(round(ang))), 540, ICON_Y - 10, 1.05 * p, p)
    if 0 <= d < 0.4:
        icircle(neon, (575, ICON_Y + 190), 60 + d * 600, col(GOLD, (1 - d / 0.4) * 1.3), 5)
    chip(crisp, "HIGH COURT · PRETORIA", 80, 1100, s, hit + 0.3, GOLD, size=28)


def sc_crime(neon, crisp, s, T, info):
    """Crime / police: flashing siren light (no graphic imagery)."""
    story_header(crisp, neon, s, 2, "POLICE", ["Arrests after", "CIT heist"], RED)
    p = pop(s)
    on = "red" if int(T * 4) % 2 == 0 else "blue"
    place_icon(neon, crisp, IC.police_light(on), 540, ICON_Y + 20, 1.15 * p, min(1, p * 2))
    chip(crisp, "SAPS STATEMENT", 80, 1100, s, info["vo_start"] + 1.0, RED, size=28)


def sc_power(neon, crisp, s, T, info):
    """Electricity / load-shedding: pylon with a flashing bolt."""
    story_header(crisp, neon, s, 3, "ESKOM", ["Load-shedding", "returns tonight"], GOLD)
    p = ease_out(prog(s, 0.3, 0.8))
    place_icon(neon, crisp, IC.pylon(), 470, ICON_Y, 0.85, p)
    q = pop(s, info["vo_start"] + 1.0, 0.4)
    flash = 1.5 if int(T * 3) % 3 == 0 else 0.8
    place_icon(neon, crisp, IC.bolt(GOLD), 820, ICON_Y, 0.8 * q, min(1, q * 2), glow=flash)
    big_stat(crisp, neon, "STAGE 2", s, info["vo_start"] + 2.0, GOLD, y=1110, size=110, label="FROM 18:00 UNTIL FURTHER NOTICE")


def sc_rates(neon, crisp, s, T, info):
    """SARB repo rate decision: dial needle swings, arrow up (hike) / down (cut)."""
    story_header(crisp, neon, s, 3, "INTEREST RATES", ["Reserve Bank", "cuts repo rate"], GREEN)
    p = pop(s)
    q = ease_out(prog(s, info["vo_start"] + 0.8, 1.2))
    ang = int(round(300 - 40 * q))  # 300 -> 260 = needle moves down (a cut)
    place_icon(neon, crisp, IC.rate_gauge(ang), 440, ICON_Y + 20, 1.0 * p, min(1, p * 2))
    r = pop(s, info["vo_start"] + 1.6, 0.4)
    place_icon(neon, crisp, IC.arrow3d(up=False), 860, ICON_Y + 8 * math.sin(T * 5), 0.7 * r, min(1, r * 2))
    big_stat(crisp, neon, "-25 BPS", s, info["vo_start"] + 2.2, GREEN, y=1110, size=110, label="REPO 6.50% · PRIME 10.00%")


def sc_rand(neon, crisp, s, T, info):
    """The rand / markets: floating gold R coin with an arrow."""
    story_header(crisp, neon, s, 4, "THE RAND", ["Rand firms to", "two-week high"], GOLD)
    p = pop(s)
    place_icon(neon, crisp, IC.coin("R"), 460, ICON_Y + 10 * math.sin(T * 2.5), 1.0 * p, min(1, p * 2))
    q = pop(s, info["vo_start"] + 1.2, 0.4)
    place_icon(neon, crisp, IC.arrow3d(up=True, color=GREEN), 820, ICON_Y, 0.8 * q, min(1, q * 2))
    big_stat(crisp, neon, "R16.45/$", s, info["vo_start"] + 2.0, GOLD, y=1110, size=110, label="STRONGEST IN TWO WEEKS")


def sc_health(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 3, "HEALTH", ["Clinics extend", "opening hours"], RED)
    p = pop(s)
    beat = 1 + 0.04 * max(0, math.sin(T * 7)) ** 8
    place_icon(neon, crisp, IC.medical(RED), 540, ICON_Y, 0.95 * p * beat, min(1, p * 2))
    chip(crisp, "DEPARTMENT OF HEALTH", 80, 1100, s, info["vo_start"] + 1.0, RED, size=28)


def sc_housing(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 4, "HOUSING", ["New homes handed", "over in Soweto"], CYAN)
    p = pop(s)
    place_icon(neon, crisp, IC.house(CYAN), 540, ICON_Y, 0.85 * p, min(1, p * 2))
    chip(crisp, "GAUTENG HUMAN SETTLEMENTS", 80, 1100, s, info["vo_start"] + 1.0, CYAN, size=28)


def sc_school(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 4, "EDUCATION", ["Matric exams", "start Monday"], GOLD)
    p = pop(s)
    place_icon(neon, crisp, IC.school(GOLD), 540, ICON_Y, 0.85 * p, min(1, p * 2))
    big_stat(crisp, neon, "900,000+", s, info["vo_start"] + 2.0, GOLD, y=1110, size=110, label="CANDIDATES WRITING")


def sc_trophy(neon, crisp, s, T, info):
    story_header(crisp, neon, s, 5, "SPORT", ["Proteas win", "the series"], GOLD)
    p = pop(s)
    hud_ring(neon, 540, ICON_Y, 300, T, GOLD, 0.4 * p)
    place_icon(neon, crisp, IC.trophy(GOLD), 540, ICON_Y, 0.82 * p, min(1, p * 2))
    chip(crisp, "SERIES WON 2-1", 80, 1100, s, info["vo_start"] + 1.4, GOLD, size=28)


def sc_world(neon, crisp, s, T, info):
    """International / trade: two globes exchanging arrows (use IC.globe() for a single world story)."""
    story_header(crisp, neon, s, 2, "TRADE", ["New export deal", "with the EU"], BLUE)
    p = pop(s)
    place_icon(neon, crisp, IC.handshake_globe(), 540, ICON_Y, 1.25 * p, min(1, p * 2))
    chip(crisp, "DTIC STATEMENT", 80, 1100, s, info["vo_start"] + 1.4, BLUE, size=28)



def sc_outro(neon, crisp, s, T, info):
    polys = map_polys(540, 820, 560)
    f = ease_out(prog(s, 0.0, 1.2))
    for i, pts in enumerate(polys):
        if i == 0:
            poly_partial_flag(neon, pts, f, 4, 0.9)
        else:
            poly_partial(neon, pts, f, col(GOLD, 0.9), 4)
    hud_ring(neon, 540, 820, 360, T, MAG, 0.7)
    title_block(crisp, s, 0.2)
    q = ease_back(prog(s, info["vo_start"] + 1.6, 0.45))
    if q > 0:
        pulse = 1 + 0.04 * math.sin(T * 7)
        btn = pill_sprite(520, 110, 55, (226, 28, 48, 255))
        blit_scaled(crisp, btn, 540, 1150, q * pulse, 1)
        blit_scaled(crisp, text_sprite("FOLLOW  +", F_BLACK, 56, WHITE, track=2), 540, 1150, q * pulse, 1)
    r = prog(s, info["vo_start"] + 2.4, 0.5)
    src = text_sprite("SOURCES: SABC NEWS · NEWS24 · IOL · CNBC AFRICA · ESPN", F_SEMI, 22, (165, 178, 205), track=1)
    blit(crisp, src, 540 - src.shape[1] / 2, 1250, r)


SCENES = dict(intro=sc_intro, vote=sc_vote, agoa=sc_agoa, water=sc_water, film=sc_film, rugby=sc_rugby, outro=sc_outro)


# ---------- compositing ----------
def glow(neon):
    sm = cv2.resize(neon, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    g1 = cv2.GaussianBlur(sm, (0, 0), 2.5)
    g2 = cv2.GaussianBlur(sm, (0, 0), 9)
    g = cv2.addWeighted(g1, 0.9, g2, 0.7, 0)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


def wipe(img, s, accent):
    d = 0.5
    if s >= d:
        return img
    u = s / d
    e = u * u * (3 - 2 * u)
    xr = W + 260 - (W + 700) * e
    band = np.array([[-300, 0], [xr, 0], [xr - 260, H], [-300, H]], np.int32)
    lead = np.array([[xr, 0], [xr + 70, 0], [xr - 190, H], [xr - 260, H]], np.int32)
    cv2.fillPoly(img, [band], NAVY, AA)
    cv2.fillPoly(img, [lead], accent, AA)
    return img


def frame(key, i):
    info = TL[key]
    s = i / FPS
    T = info["start"] + s
    accent = ACCENT[key]
    base = BG_R.copy()
    neon = np.zeros((H, W, 3), np.uint8)
    crisp = np.zeros((H, W, 3), np.uint8)  # drawn with alpha onto final
    crisp_layer = np.zeros((H, W, 4), np.uint8)
    draw_background(base, neon, T, accent)
    if key not in ("intro", "outro"):
        card(base, accent, s)
        card_accent(neon, accent, s)
    # crisp elements are blitted directly onto a holder that we composite after glow
    holder = Holder()
    draw_hud(neon, holder, T, accent, ORDER.index(key), s, info["dur"])
    SCENES[key](neon, holder, s, T, info)
    draw_ticker(holder, neon, T)
    if key != "outro" or s < info["vo_start"] + info["vo_len"]:
        draw_caption(holder, key, info, s)
    out = cv2.add(base, neon)
    out = cv2.add(out, glow(neon))
    holder.apply(out)
    if Y_SHIFT:
        out = np.vstack([BG[:Y_SHIFT], out[:H - Y_SHIFT]])
    out = wipe(out, s, accent)
    if key == "outro":
        fade = prog(s, info["dur"] - 0.5, 0.5)
        if fade > 0:
            out = (out * (1 - fade)).astype(np.uint8)
    return out


class Holder:
    """Collects crisp (non-glowing) draws and replays them after the glow pass."""
    def __init__(self):
        self.ops = []

    @property
    def shape(self):
        return (H, W, 3)

    def apply(self, out):
        for fn, args in self.ops:
            fn(out, *args)


# Route blit/blit_scaled/cv2 calls aimed at a Holder into deferred ops
_blit, _blit_scaled = blit, blit_scaled


def blit(dst, *a, **kw):  # noqa: F811
    if isinstance(dst, Holder):
        dst.ops.append((lambda o, *aa: _blit(o, *aa, **kw), a))
    else:
        _blit(dst, *a, **kw)


def blit_scaled(dst, *a, **kw):  # noqa: F811
    if isinstance(dst, Holder):
        dst.ops.append((lambda o, *aa: _blit_scaled(o, *aa, **kw), a))
    else:
        _blit_scaled(dst, *a, **kw)


_rect = cv2.rectangle


class _CV:
    pass


def crisp_rect(dst, p1, p2, c, th):
    if isinstance(dst, Holder):
        dst.ops.append((lambda o, *aa: _rect(o, *aa), (p1, p2, c, th)))
    else:
        _rect(dst, p1, p2, c, th)


cv2_rectangle_orig = cv2.rectangle


def _rect_dispatch(dst, *a, **kw):
    if isinstance(dst, Holder):
        dst.ops.append((lambda o, *aa: cv2_rectangle_orig(o, *aa, **kw), a))
        return dst
    return cv2_rectangle_orig(dst, *a, **kw)


cv2.rectangle = _rect_dispatch
cv2_circle_orig = cv2.circle


def _circle_dispatch(dst, *a, **kw):
    if isinstance(dst, Holder):
        dst.ops.append((lambda o, *aa: cv2_circle_orig(o, *aa, **kw), a))
        return dst
    return cv2_circle_orig(dst, *a, **kw)


cv2.circle = _circle_dispatch


def render(key, preview_times=None):
    info = TL[key]
    if preview_times is not None:
        for tt in preview_times:
            i = int(tt * FPS)
            img = frame(key, i)
            Image.fromarray(img).save(f"{OUT}/preview_{key}_{tt:.1f}.jpg", quality=88)
        return
    tmp = f"{OUT}/scene_{key}.part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-f", "mp4", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(info["frames"]):
        p.stdin.write(frame(key, i).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, f"{OUT}/scene_{key}.mp4")
        print("done", key)
    else:
        sys.exit(f"ffmpeg failed for {key}")


if __name__ == "__main__":
    key = sys.argv[1]
    if len(sys.argv) > 2:
        render(key, [float(x) for x in sys.argv[2:]])
    else:
        render(key)
