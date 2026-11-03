# SA Daily Brief: reel template

Same every episode: layout, logo, colours, music, sound effects, voice (Kokoro af_heart, speed 1.08).

See DAILY_RUN.md for the full daily process. Per episode, change:
1. `common.py`: `EPISODE` dates, `ORDER`, `CAPTIONS`, `SFX`
2. `tts.py`: `SEGMENTS` phonemes for each story, then `python3 tts.py` (needs `models/` from the kokoro-onnx GitHub release)
3. `render.py`: story scenes (`sc_*`), `TICKER`, `ACCENT`, sources line in `sc_outro`
4. `bash run.sh` gives `SA_Daily_Brief_<date>.mp4`

Logo assets: `assets/logo_source.png` (original), `logo_map.png` + `logo_wordmark.png` (extracted, transparent).
