"""Music bed, sound effects and voiceover mix."""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from scipy.io import wavfile

from common import DIR, OUT, SFX, timeline

SR = 48000
TL = timeline()
TOTAL = sum(v["dur"] for v in TL.values())
N = int(TOTAL * SR) + SR
rng = np.random.default_rng(3)


def t_arr(d):
    return np.arange(int(d * SR)) / SR


def env(n, a=0.005, r=0.2, curve=4.0):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-curve * t / max(r, 1e-4))
    return e


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype="high", fs=SR, output="sos"), x)


def place(buf, x, t, g=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    x = x[: len(buf) - i]
    L, R = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[i:i + len(x), 0] += x * g * L * 1.41
    buf[i:i + len(x), 1] += x * g * R * 1.41


# ---------- sfx ----------
def whoosh(d=0.55, up=True):
    n = int(d * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    steps = 24
    for k in range(steps):
        a, b = k * n // steps, (k + 1) * n // steps
        f = 300 + (5200 if up else 2500) * (k / steps if up else 1 - k / steps)
        seg = bp(noise, max(80, f * 0.6), min(20000, f * 1.6))[a:b]
        out[a:b] = seg
    shape = np.sin(np.linspace(0, np.pi, n)) ** 1.6
    return out * shape * 0.9


def impact(d=1.4, f0=90, f1=38):
    t = t_arr(d)
    f = f1 + (f0 - f1) * np.exp(-t * 9)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 3.2)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t * 60) * 0.5
    tail = lp(rng.standard_normal(len(t)), 900) * np.exp(-t * 4) * 0.25
    return np.tanh((body + click + tail) * 1.6) * 0.42


def blip(f=1800, d=0.09, g=0.5):
    t = t_arr(d)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t)
    return x * env(len(t), 0.002, d, 5) * g


def tick(g=0.35):
    t = t_arr(0.03)
    return hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 200) * g + blip(3200, 0.03, g * 0.4)


def thunk():
    t = t_arr(0.35)
    f = 120 + 160 * np.exp(-t * 30)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)
    x += lp(rng.standard_normal(len(t)), 1500) * np.exp(-t * 40) * 0.6
    return x * 0.8


def clap():
    t = t_arr(0.4)
    x = bp(rng.standard_normal(len(t)), 900, 6000) * np.exp(-t * 45)
    x += np.sin(2 * np.pi * 820 * t) * np.exp(-t * 35) * 0.5
    x += np.sin(2 * np.pi * 140 * t) * np.exp(-t * 20) * 0.5
    return np.tanh(x * 2) * 0.75


def bloop():
    t = t_arr(0.25)
    f = 300 + 900 * (t / 0.25) ** 0.6
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.005, 0.12, 3) * 0.45


def alert():
    out = np.zeros(int(0.9 * SR))
    for k, f in enumerate([880, 660, 880, 660]):
        b = blip(f, 0.16, 0.35)
        i = int(k * 0.2 * SR)
        out[i:i + len(b)] += b
    return out


def zap():
    t = t_arr(0.5)
    f = 2400 * np.exp(-t * 8) + 120
    sq = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lp(sq, 5000) * np.exp(-t * 6) * 0.25


def shimmer(d=2.5):
    t = t_arr(d)
    x = np.zeros(len(t))
    for f in [1318.5, 1760, 2093, 2637]:
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 7) * t))
    return x * np.sin(np.linspace(0, np.pi, len(t))) * 0.05


# ---------- music ----------
BPM = 118
BEAT = 60 / BPM
NOTE = lambda m: 440 * 2 ** ((m - 69) / 12)
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
ROOTS = [45, 41, 36, 43]


def music(n):
    out = np.zeros(n)
    bar = 4 * BEAT
    nbars = int(n / SR / bar) + 1
    for b in range(nbars):
        ci = b % 4
        t0 = b * bar
        # pad
        d = bar
        t = t_arr(d)
        pad = np.zeros(len(t))
        for m in CHORDS[ci]:
            for det in (-0.12, 0.12):
                f = NOTE(m + 12) * 2 ** (det / 12)
                pad += 2 * ((t * f) % 1) - 1
        pad = lp(pad, 1400) * 0.025 * np.minimum(1, np.minimum(t / 0.3, (d - t) / 0.3))
        i = int(t0 * SR)
        seg = pad[: max(0, n - i)]
        out[i:i + len(seg)] += seg
        for k in range(8):  # 8ths
            tk = t0 + k * BEAT / 2
            i = int(tk * SR)
            if i >= n:
                break
            # bass
            tb = t_arr(BEAT / 2 * 0.9)
            fb = NOTE(ROOTS[ci])
            bass = (np.sin(2 * np.pi * fb * tb) + 0.4 * np.sin(4 * np.pi * fb * tb)) * env(len(tb), 0.004, 0.18, 3) * 0.12
            # pluck arp
            m = CHORDS[ci][[0, 1, 2, 1][k % 4]] + 24
            tp = t_arr(0.22)
            pl = (2 * ((tp * NOTE(m)) % 1) - 1)
            pl = lp(pl, 3500) * env(len(tp), 0.002, 0.08, 4) * 0.035
            # hat
            th = t_arr(0.05)
            hat = hp(rng.standard_normal(len(th)), 7000) * np.exp(-th * 90) * (0.05 if k % 2 else 0.025)
            for x in (bass, pl, hat):
                x = x[: n - i]
                out[i:i + len(x)] += x
            if k % 2 == 0:  # kick on beats
                tk_ = t_arr(0.3)
                fk = 48 + 90 * np.exp(-tk_ * 35)
                kick = np.sin(2 * np.pi * np.cumsum(fk) / SR) * np.exp(-tk_ * 12) * 0.22
                kick = kick[: n - i]
                out[i:i + len(kick)] += kick
    return out


# ---- story sound presets: (mix, scene start, voice start) ----
def sfx_vote(mix, s, v):
    for t0 in (1.0, 2.2, 3.4):
        place(mix, thunk(), s + t0 + 0.9, 0.7)
    place(mix, blip(1300, 0.1), s + 2.6, 0.5)
    c0 = v + 4.4
    for j in range(16):
        place(mix, tick(0.3), c0 + j * 0.1 * (1 + j / 30), 0.7, rng.uniform(-0.3, 0.3))
    place(mix, blip(2093, 0.35, 0.4), c0 + 1.65, 0.8)
    place(mix, blip(2637, 0.35, 0.3), c0 + 1.68, 0.8)


def sfx_breaking(mix, s, v):
    place(mix, blip(900, 0.12), s + 0.4, 0.5, -0.6)
    place(mix, blip(1100, 0.12), s + 0.45, 0.5, 0.6)
    brk = v + 1.6
    place(mix, impact(1.6, 110, 34), brk, 0.95)
    place(mix, zap(), brk, 0.8)
    place(mix, tick(), brk + 0.5, 0.9)
    place(mix, blip(1500, 0.1), v + 6.4, 0.5)


def sfx_water(mix, s, v):
    place(mix, bloop(), s + 0.35, 0.7)
    place(mix, bloop(), s + 1.1, 0.4, 0.4)
    place(mix, alert(), v + 2.6, 0.55)


def sfx_clap(mix, s, v):
    for j, d in enumerate([0.2, 1.4, 2.9]):
        place(mix, clap(), v + d, 0.85 + 0.1 * j, -0.2 + 0.2 * j)
    place(mix, impact(1.0, 100, 45), v + 2.9, 0.6)


def sfx_sport(mix, s, v):
    place(mix, whoosh(0.8, False), s + 0.3, 0.5, 0.3)
    place(mix, impact(1.2, 95, 40), v + 2.3, 0.7)
    for j in range(3):
        place(mix, blip(1400 + j * 250, 0.09), v + 4.6 + j * 0.45, 0.5, -0.4 + j * 0.4)


def sfx_alert(mix, s, v):
    place(mix, alert(), v + 1.2, 0.55)
    place(mix, impact(1.0, 100, 45), v + 1.2, 0.5)


def sfx_plain(mix, s, v):
    place(mix, blip(1200, 0.1), v + 1.0, 0.5, -0.3)
    place(mix, blip(1500, 0.1), v + 1.2, 0.5, 0.3)


SFX_PRESETS = dict(vote=sfx_vote, breaking=sfx_breaking, water=sfx_water, clap=sfx_clap,
                   sport=sfx_sport, alert=sfx_alert, plain=sfx_plain)


def main():
    mix = np.zeros((N, 2))
    vo = np.zeros(N)
    for k, info in TL.items():
        v = np.load(f"{DIR}/assets/vo_{k}.npy").astype(np.float64)
        v = resample_poly(v, 2, 1)
        v = v / (np.abs(v).max() + 1e-9) * 0.85
        i = int((info["start"] + info["vo_start"]) * SR)
        vo[i:i + len(v)] += v[: N - i]

    # sfx timeline
    S = {k: v["start"] for k, v in TL.items()}
    V = {k: v["start"] + v["vo_start"] for k, v in TL.items()}
    for k in TL:
        if k != "intro":
            place(mix, whoosh(), S[k] - 0.12, 0.55, -0.3)
        if k not in ("intro", "outro"):
            place(mix, blip(1500, 0.08), S[k] + 0.12, 0.6, -0.4)
            place(mix, tick(), S[k] + 0.32, 0.8, 0.2)
            place(mix, tick(), S[k] + 0.55, 0.8, 0.3)
    # intro
    place(mix, whoosh(0.7), 0.0, 0.7)
    place(mix, impact(), 0.28, 0.9)
    place(mix, shimmer(2.6), 0.4, 1.0)
    for j in range(3):
        place(mix, blip(1200 + j * 300, 0.1), 1.5 + j * 0.25, 0.45, -0.5 + j * 0.5)
    # per-story sound presets (see SFX in common.py)
    for k in TL:
        if k in ("intro", "outro"):
            continue
        SFX_PRESETS[SFX.get(k, "plain")](mix, S[k], V[k])
    # outro
    place(mix, impact(), S["outro"] + 0.2, 0.75)
    place(mix, blip(1760, 0.2, 0.45), V["outro"] + 1.6, 0.8)
    place(mix, shimmer(2.5), S["outro"] + TL["outro"]["dur"] - 2.6, 1.2)

    # music with ducking
    m = music(N)
    act = np.abs(vo) > 0.02
    w = int(0.25 * SR)
    act = np.convolve(act.astype(float), np.ones(w) / w, mode="same") > 0.02
    duck = np.where(act, 0.38, 1.0)
    duck = np.convolve(duck, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
    end = TOTAL
    fade = np.clip((end - np.arange(N) / SR) / 1.2, 0, 1)
    fin = np.clip(np.arange(N) / SR / 0.3, 0, 1)
    m = m * duck * fade * fin * 1.1
    mix[:, 0] += m + vo
    mix[:, 1] += np.roll(m, 12) + vo
    mix = mix[: int(TOTAL * SR)]
    mix = mix / np.abs(mix).max() * 0.89
    wavfile.write(f"{OUT}/mix.wav", SR, (mix * 32767).astype(np.int16))
    print("mix", round(len(mix) / SR, 2))


if __name__ == "__main__":
    main()
