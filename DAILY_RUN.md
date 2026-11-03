# SA Daily Brief: daily reel runbook

Follow this every day to make one ~55-second Instagram Reel of today's South African news, plus a ready-to-paste caption. The look, logo, music, sound effects and voice stay the same every day. Only the stories change.

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
Clone with `git clone --depth 1 https://github.com/Ceptivo/sadailynews` and work in that folder. **Do not commit changes to `render.py`, `audio.py`, `common.py`, `tts.py` or `assets/vo_*.npy`.** The committed versions are the reference template (the 3 Nov 2026 episode). Only `episodes/` gets committed (step 8).

## 1. Find today's news (South Africa time, Africa/Johannesburg)

Use WebSearch/WebFetch. Check several of these sources: News24, SABC News, IOL, TimesLIVE, Daily Maverick, EWN, Moneyweb, BusinessTech, Mail & Guardian, SuperSport, plus Reuters/AFP/Bloomberg for SA stories.

Pick **5 stories** from the last 24 hours, roughly in this mix: the day's biggest national story, then politics/government, economy/business/money (rand, fuel, interest rates, load-shedding, jobs), a local or community story (service delivery, a city or province), and something lighter (sport, entertainment, a good-news story).

Rules:
- Every fact must be confirmed by at least 2 reputable sources, or come from 1 primary source (government, IEC, SARB, Stats SA, police statement). Leave out anything you can't confirm.
- Neutral, factual tone. No opinion, no speculation, no loaded words. Report what each side says.
- Nothing graphic. Leave out victims' names in crime stories. Don't name minors.
- Don't repeat a story that was already covered in `episodes/` (check the last 3 days) unless there's a real new development.
- Keep the source names for the outro line and the caption.

## 2. Write the script (target 50–58 s total)

- `intro`: "This is SA Daily Brief — your 60-second news update for <Weekday>, <D> <Month>."
- 5 stories: each 15–28 spoken words, 1–2 short sentences, the most important fact first.
- `outro`: "That's your 60 seconds. Follow for more tomorrow." (or a short, relevant follow line).
- Write numbers the way they're spoken in the phonemes ("seven hundred thousand"). Captions can use digits.

## 3. Voiceover (`tts.py`)

Replace `SEGMENTS` with today's keys and **hand-written misaki (US English) phonemes**, in the same style as the existing entries (stress marks ˈ ˌ, A=ay, I=eye, O=oh, W=ow, Y=oy, ɾ flap, ɚ, ʤ, ʧ). Only characters in `assets/kokoro_vocab.json` are allowed (the script exits and lists any it doesn't recognise). Spell SA names phonetically so they sound right (for example ʧwˈɑnɛ for Tshwane, ˈɛs ˈA for SA, ɹˈɑsi for Rassie). Keep the intro phonemes for "This is SA Daily Brief" exactly as they are now: `ðˈɪs ɪz ˈɛs ˈA dˈAli bɹˈif`.

Run `python3 tts.py`. Check each segment's length; if the total from `python3 common.py` is over 58 s, shorten the text.

## 4. Episode settings (`common.py`)

- `EPISODE`: `hud_date` like "WED 04 NOV 2026", `short_date` like "4 NOV 2026", `file_date` like "4Nov2026".
- `ORDER`: `["intro", <5 story keys>, "outro"]` (short lowercase keys).
- `CAPTIONS`: on-screen caption chunks for each key. They must match the spoken words, at most about 30 characters per chunk.
- `SFX`: a sound preset for each story: `vote` (counting/ballot), `breaking` (big impact, good for politics/economy/conflict), `water` (water/utilities), `clap` (clapperboard: film/arts/announcements), `sport`, `alert` (warnings, crime, weather), `plain`.

## 5. Visuals (`render.py`)

- `TICKER`: 5–7 short uppercase headlines separated by `   ■   `.
- `ACCENT`: a colour per key (CYAN, MAG, GOLD, GREEN, RED, BLUE). Vary them between neighbouring stories.
- Sources line in `sc_outro`: "SOURCES: …" listing today's outlets (keep it under about 60 characters).
- One scene function per story, registered in `SCENES`. Every story scene starts with
  `story_header(crisp, neon, s, <n>, "<TAG>", ["<headline line 1>", "<line 2>"], <accent>)`
  (TAG is a short uppercase category, headline lines at most about 18 characters each), then draws one neon line-art icon/illustration in the card area (y about 600–1050) using the existing helpers (`poly_partial`, `iline`, `icircle`, `cv2` shapes on the `neon` layer with `col()`), plus optional `big_stat(...)` and `chip(...)` lines that appear in time with the voiceover (`info["vo_start"] + seconds`).
- Reuse and adapt the existing scenes wherever they fit: `sc_vote` (ballot box: elections/voting), `sc_agoa` (trade/arrows: economy, international), `sc_water` (water drop: utilities/municipal), `sc_film` (clapperboard: arts/entertainment), `sc_rugby` (ball/players: sport). For other topics, write a new scene in the same style (for example: rand/coin chart for money, lightning bolt for electricity, fuel pump, police light, house, school, hospital cross, cloud/rain for weather, gavel for courts, globe for world news). Keep it simple, clean and centred, the same neon-outline look.
- Leave the intro, outro, HUD, logo, ticker style, captions, background and transitions alone.

## 6. Check, then render

1. Preview stills: `python3 render.py <key> 1.0 3.0 6.0` for every scene, then look at `build/preview_*.jpg` with Read. Fix any text that overflows, overlaps or runs off screen, and any icon that's broken or ugly.
2. `rm -rf build && bash run.sh`
3. Verify with ffprobe: 1080x1920, 30 fps, 50–60 s, with an audio track. Look at 4–5 frames taken from across the video.

## 7. Instagram caption (`caption.txt`)

Same format as this example:
```
🇿🇦 Your 60-second SA news update for <Weekday>, <D> <Month>.

<emoji> <story 1 in 1–2 sentences>
<emoji> <story 2>
… (all 5)

Follow for tomorrow's update 🔔

Sources: <outlets>

#SADailyBrief #SouthAfrica #SANews #Mzansi #NewsUpdate <4–7 story-specific hashtags>
```

## 8. Deliver

1. Copy the MP4 (`SA_Daily_Brief_<date>.mp4`) and `caption.txt` to `/mnt/user-data/outputs/`, then send both with SendUserFile.
2. Send the caption text with SendUserMessage too, so it can be copied straight from the message, along with a one-line list of the 5 stories.
3. Log the episode: write `episodes/<YYYY-MM-DD>.md` (the stories, sources with URLs, the script and the caption). Commit only that file and push it to the default branch, with message "Episode <YYYY-MM-DD>". If the push fails, say so in the final message, but still deliver the video.

If something fails (news search, TTS, render), fix it and carry on. If a video truly can't be made, send a message explaining what went wrong, and still send the caption and story list.
