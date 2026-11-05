# SA Daily Brief: daily reel runbook

Follow this to make a ~55-second Instagram Reel of South African news, plus a ready-to-paste caption. The look, logo, music, sound effects and voice stay the same every time. Only the stories change.

There are **two editions a day**. The scheduled task tells you which one to make:

| | Morning edition (runs ~06:20) | Afternoon edition (runs ~15:45) |
|---|---|---|
| Covers | News since the previous afternoon edition (yesterday ~16:00 to now), plus **what matters for today** | **What happened today** (from this morning to now): the day's wrap-up |
| Must-checks | **Today's weather** (SAWS warnings, heat, storms, big-city forecast), fuel price changes, interest rates, the rand, load-shedding / water outages, major traffic or strike disruptions, and big scheduled events today (court cases, announcements, matches) | Fuel prices and interest rates (if announced today), plus the day's main national, political, economic, local and sport stories |
| Intro line | "…your 60-second morning update for <Weekday>, <D> <Month>." | "…your 60-second news of the day for <Weekday>, <D> <Month>." |
| Outro line | "That's your morning update. Follow for today's news at four." | "That's your 60 seconds. Follow for tomorrow's morning update." |
| Caption first line | 🇿🇦 Your 60-second morning update for <Weekday>, <D> <Month>. | 🇿🇦 Today's SA news in 60 seconds, <Weekday>, <D> <Month>. |
| `file_date` / log file | `<D><Mon><YYYY>_AM` · `episodes/<YYYY-MM-DD>-am.md` | `<D><Mon><YYYY>_PM` · `episodes/<YYYY-MM-DD>-pm.md` |

The morning edition is the "start your day" brief: it should almost always include a weather story, and include a money story (fuel, rand, rates) whenever there's a real change or update. The afternoon edition is "this happened today, that happened today": it focuses on today's events, not things the morning edition already covered (only include those if there's a real new development).

The run is unattended, so don't stop to ask questions. Make sensible calls and keep going.

## 0. Setup

```bash
pip install --break-system-packages numpy scipy opencv-python pillow onnxruntime
which ffmpeg || (apt-get update && apt-get install -y ffmpeg)
mkdir -p models && cd models
curl -sSL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -sSL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..
```
Clone with `git clone --depth 1 https://github.com/Ceptivo/sadailynews` and work in that folder. **Do not commit changes to `render.py`, `audio.py`, `common.py`, `tts.py` or `assets/vo_*.npy`.** The committed versions are the reference template (the 3 Nov 2026 episode). Only `episodes/` gets committed (step 8), plus any new icon you add to `icons.py` (commit that separately as "Add <name> icon", after checking `python3 icon_gallery.py` still runs).

## 1. Find the news (South Africa time, Africa/Johannesburg)

Use WebSearch/WebFetch. Check several of these sources: News24, SABC News, IOL, TimesLIVE, Daily Maverick, EWN, Moneyweb, BusinessTech, Mail & Guardian, SuperSport, plus Reuters/AFP/Bloomberg for SA stories.

Pick **5 stories** for the edition's time window (see the table above), roughly in this mix: the day's biggest national story, then politics/government, economy/business/money (rand, fuel, interest rates, load-shedding, jobs), a local or community story (service delivery, a city or province), and something lighter (sport, entertainment, a good-news story).

**Always check these two, and include them whenever they change:**
- **Petrol and diesel prices.** The Department of Mineral and Petroleum Resources announces changes, usually in the last days of the month, and they take effect on the first Wednesday of the month. When a change is announced or takes effect today or yesterday, include it as a story: increase or decrease, cents per litre for 95 petrol (inland) and diesel, and the date it applies from. Also include it when the mid-month data (Central Energy Fund) points clearly to a big increase or decrease, worded as "early data suggests…".
- **Interest rates.** When the SARB Monetary Policy Committee announces a repo rate decision (cut, hike or hold), include it: the change in basis points, the new repo and prime rates, and one line on what it means for home and car repayments. Also include major-bank prime rate changes that follow.

These count toward the 5 stories (they replace the economy slot, or bump the weakest story). Use the official numbers from the announcement, confirmed by at least 2 outlets. For these stories, use a scene with a neon fuel pump (petrol) or a % sign/rate chart (interest rates), a big up arrow (red) or down arrow (green), and `big_stat` showing the cents per litre or basis points, with the `breaking` sound preset.

Rules:
- Every fact must be confirmed by at least 2 reputable sources, or come from 1 primary source (government, IEC, SARB, Stats SA, police statement). Leave out anything you can't confirm.
- Neutral, factual tone. No opinion, no speculation, no loaded words. Report what each side says.
- Nothing graphic. Leave out victims' names in crime stories. Don't name minors.
- Don't repeat a story that was already covered in `episodes/` (check the last 3 days, both editions, including this morning's) unless there's a real new development.
- Keep the source names for the outro line and the caption.

## 2. Write the script (target 50–58 s total)

- `intro`: the edition's intro line from the table, starting "This is SA Daily Brief — …".
- 5 stories: each 15–28 spoken words, 1–2 short sentences, the most important fact first.
- `outro`: the edition's outro line from the table (or a short, relevant follow line, e.g. "Follow for election results as they come in.").
- Write numbers the way they're spoken in the phonemes ("seven hundred thousand"). Captions can use digits.

## 3. Voiceover (`tts.py`)

Replace `SEGMENTS` with today's keys and **hand-written misaki (US English) phonemes**, in the same style as the existing entries (stress marks ˈ ˌ, A=ay, I=eye, O=oh, W=ow, Y=oy, ɾ flap, ɚ, ʤ, ʧ). Only characters in `assets/kokoro_vocab.json` are allowed (the script exits and lists any it doesn't recognise). Spell SA names phonetically so they sound right (for example ʧwˈɑnɛ for Tshwane, ˈɛs ˈA for SA, ɹˈɑsi for Rassie). Keep the intro phonemes for "This is SA Daily Brief" exactly as they are now: `ðˈɪs ɪz ˈɛs ˈA dˈAli bɹˈif`.

Run `python3 tts.py`. Check each segment's length; if the total from `python3 common.py` is over 58 s, shorten the text.

## 4. Episode settings (`common.py`)

- `EPISODE`: `hud_date` like "WED 04 NOV 2026", `short_date` like "4 NOV 2026", `file_date` like "4Nov2026_AM" or "4Nov2026_PM".
- `ORDER`: `["intro", <5 story keys>, "outro"]` (short lowercase keys).
- `CAPTIONS`: on-screen caption chunks for each key. They must match the spoken words, at most about 30 characters per chunk.
- `SFX`: a sound preset for each story: `vote` (counting/ballot), `breaking` (big impact, good for politics/economy/conflict), `water` (water/utilities), `clap` (clapperboard: film/arts/announcements), `sport`, `alert` (warnings, crime, weather), `plain`.

## 5. Visuals (`render.py`)

- `TICKER`: 5–7 short uppercase headlines separated by `   ■   `.
- `ACCENT`: a colour per key (CYAN, MAG, GOLD, GREEN, RED, BLUE). Vary them between neighbouring stories.
- Sources line in `sc_outro`: "SOURCES: …" listing today's outlets (keep it under about 60 characters).
- One scene function per story, registered in `SCENES`. Every story scene starts with
  `story_header(crisp, neon, s, <n>, "<TAG>", ["<headline line 1>", "<line 2>"], <accent>)`
  (TAG is a short uppercase category, headline lines at most about 18 characters each), then shows **one detailed illustration from `icons.py`** in the card area with `place_icon(neon, crisp, IC.<icon>(...), x, y, scale, alpha, angle=...)` (centre it around `ICON_Y` = 815, keep it within y≈600–1050 and no taller than ~430 px after scaling), plus optional `big_stat(...)` and `chip(...)` lines that appear in time with the voiceover (`info["vo_start"] + seconds`). Animate with `pop(s)` for the entrance, and gentle motion (bob, spin via `angle`, flashing via `glow=`); small neon extras (ripples, speed lines, rings) are fine around the icon.
- **Icon style: detailed and realistic, not flat line art.** Run `python3 icon_gallery.py` and look at `build/icon_gallery.jpg` to see what's available: `ballot_box`, `fuel_pump`, `arrow3d` (red up / green down), `parliament`, `medal`, `trophy`, `storm_cloud`, `sun`, `tap_drop`, `soccer_ball`, `rugby_ball`, `clapperboard`, `bars3d` (results/polls), `map_pin`, `skyline`, `table_mountain`, `gavel`, `police_light`, `pylon`, `bolt`, `house`, `medical`, `rate_gauge`, `coin`, `globe`, `school`, `handshake_globe`, `sphere`. Most take an accent colour or label; pick the closest fit.
- Ready-made scenes in `render.py` (copy, rename the key, change header text, chips and stats): `sc_vote` (elections), `sc_results` (results/polls bars), `sc_agoa` / `sc_world` (trade, international), `sc_water` (water/utilities), `sc_film` (arts/entertainment), `sc_rugby`, `sc_soccer`, `sc_medal`, `sc_trophy` (sport), `sc_weather` (storms/heat), `sc_fuel` (fuel prices), `sc_rates` (SARB repo rate), `sc_rand` (rand/markets), `sc_parliament` (government), `sc_capetown`, `sc_city` (metros), `sc_pin` (a town/municipality), `sc_court` (courts), `sc_crime` (police), `sc_power` (Eskom/load-shedding), `sc_health`, `sc_housing`, `sc_school`.
- **No icon fits?** Add a new one to `icons.py` in the same style (see the docstring at the top): solid gradient body lit from the top-left, a darker side/underside for depth, a white highlight or sheen, real-world details (screws, seams, windows, labels), a soft ground shadow, and an accent rim stroke with `glow=1.0`. Check it in the gallery before using it. New icons may be committed (see step 8); never fall back to plain neon outlines.
- Leave the intro, outro, HUD, logo, ticker style, captions, background and transitions alone. Don't change `Y_SHIFT` in `common.py`: it moves everything down so the top bar clears Instagram's "Reels / Friends" header and the ticker sits just above the username. Keep scene content between y≈230 and y≈1460 (before the shift) so nothing hits those areas.

## 6. Check, then render

1. Preview stills: `python3 render.py <key> 1.0 3.0 6.0` for every scene, then look at `build/preview_*.jpg` with Read. Fix any text that overflows, overlaps or runs off screen, and any icon that's broken or ugly.
2. `rm -rf build && bash run.sh`
3. Verify with ffprobe: 1080x1920, 30 fps, 50–60 s, with an audio track. Look at 4–5 frames taken from across the video.

## 7. Instagram caption (`caption.txt`)

Same format as this example:
```
<edition's caption first line from the table>

<emoji> <story 1 in 1–2 sentences>
<emoji> <story 2>
… (all 5)

Follow @sadailybrief for <the next edition: "today's news at 4pm" or "tomorrow's morning update"> 🔔

Sources: <outlets>

#SADailyBrief #SouthAfrica #SANews #Mzansi #NewsUpdate <4–7 story-specific hashtags>
```

Every caption must include the follow line with the real handle **@sadailybrief**. Never use a placeholder like "@your handle" or "[handle]".

## 8. Deliver

1. Copy the MP4 (`SA_Daily_Brief_<file_date>.mp4`) and `caption.txt` to `/mnt/user-data/outputs/`, then send both with SendUserFile.
2. Send the caption text with SendUserMessage too, so it can be copied straight from the message, along with a one-line list of the 5 stories.
3. Log the episode: write `episodes/<YYYY-MM-DD>-am.md` or `-pm.md` (the stories, sources with URLs, the script and the caption). Commit only that file and push it to the default branch, with message "Episode <YYYY-MM-DD> AM/PM" (pull first if the push is rejected). If the push fails, say so in the final message, but still deliver the video.

If something fails (news search, TTS, render), fix it and carry on. If a video truly can't be made, send a message explaining what went wrong, and still send the caption and story list.
