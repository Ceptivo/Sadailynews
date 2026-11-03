#!/usr/bin/env bash
# Builds the SA Daily Brief reel (1080x1920, 30 fps, AAC 48 kHz). Output name comes from OUTFILE in common.py.
# Needs: python3 with numpy, scipy, opencv-python, pillow; ffmpeg with libx264.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
OUTFILE=$(python3 -c "from common import OUTFILE; print(OUTFILE)")

echo "1/4 rendering scenes (2 at a time)..."
ORDER=$(python3 -c "from common import ORDER; print(' '.join(ORDER))")
printf "%s\n" $ORDER | xargs -P 2 -I{} python3 render.py {}

echo "2/4 building soundtrack..."
python3 audio.py

echo "3/4 joining scenes..."
for k in $ORDER; do echo "file 'scene_$k.mp4'"; done > build/list.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i build/list.txt -c copy build/video.mp4

echo "4/4 adding sound..."
ffmpeg -y -loglevel error -i build/video.mp4 -i build/mix.wav -map 0:v -map 1:a -c:v copy \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" -ar 48000 -c:a aac -b:a 192k -shortest -movflags +faststart \
  "$OUTFILE"

echo "Done: $(pwd)/$OUTFILE"
