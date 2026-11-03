"""Shared timeline for picture and sound."""
import numpy as np

import os
DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "build")
os.makedirs(OUT, exist_ok=True)
FPS = 30

# Everything except the background is drawn this many pixels lower, so the top bar clears
# Instagram's "Reels / Friends" header and the ticker sits just above the username.
Y_SHIFT = 90

# ---- channel branding (same every episode) ----
BRAND = "SA Daily Brief"
# ---- episode settings (change every episode) ----
EPISODE = dict(
    hud_date="TUE 03 NOV 2026",   # under the corner logo
    short_date="3 NOV 2026",      # under the intro title
    file_date="3Nov2026",         # output filename
)
OUTFILE = f"SA_Daily_Brief_{EPISODE['file_date']}.mp4"
VO_SR = 24000

ORDER = ["intro", "vote", "agoa", "water", "film", "rugby", "outro"]
# sound preset per story: vote | breaking | water | clap | sport | alert | plain
SFX = {"vote": "vote", "agoa": "breaking", "water": "water", "film": "clap", "rugby": "sport"}
LEAD = {"intro": 1.5, "outro": 0.45}
TAIL = {"outro": 2.6}
DEFAULT_LEAD, DEFAULT_TAIL = 0.5, 0.65

CAPTIONS = {
    "intro": ["This is SA Daily Brief —", "your 60-second news update", "for Tuesday, 3 November."],
    "vote": ["Special voting wraps up today,", "ahead of tomorrow's", "local government elections.",
             "The IEC says more than", "700,000 votes were cast", "on day one alone."],
    "agoa": ["Washington says it intends", "to remove South Africa", "from AGOA, the duty-free",
             "trade deal, from 1 January.", "Pretoria says it will support", "affected industries."],
    "water": ["In Tshwane,", "Bronkhorstspruit residents", "are told their tap water", "is still not safe to drink."],
    "film": ["Film industry talks", "with the trade department", "have broken down", "for a third time."],
    "rugby": ["In rugby, Rassie Erasmus", "has named six uncapped players", "for the Springboks' tour",
              "of Italy, France and Ireland."],
    "outro": ["That's your 60 seconds.", "Follow for live", "election updates tomorrow."],
}


def timeline():
    t, out = 0.0, {}
    for k in ORDER:
        vo = np.load(f"{DIR}/assets/vo_{k}.npy")
        vlen = len(vo) / VO_SR
        lead = LEAD.get(k, DEFAULT_LEAD)
        dur = lead + vlen + TAIL.get(k, DEFAULT_TAIL)
        nframes = int(round(dur * FPS))
        dur = nframes / FPS
        out[k] = dict(start=t, dur=dur, frames=nframes, vo_start=lead, vo_len=vlen)
        t += dur
    return out


def caption_times(k, info):
    """Spread caption chunks across the voice span by character count."""
    chunks = CAPTIONS[k]
    w = np.array([len(c) + 6 for c in chunks], float)
    edges = np.concatenate([[0], np.cumsum(w) / w.sum()]) * info["vo_len"] + info["vo_start"]
    return [(chunks[i], edges[i], edges[i + 1]) for i in range(len(chunks))]


if __name__ == "__main__":
    tl = timeline()
    for k, v in tl.items():
        print(k, {a: round(b, 2) for a, b in v.items()})
    print("TOTAL", round(sum(v["dur"] for v in tl.values()), 2))
