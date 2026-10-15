#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

# ===== User settings =====
VIDEO = Path("/mnt/data2/dexmimic/workspace/egoengine-webite/test/inpainted_video.mp4")
OUTPUT = VIDEO.parent / "output.mp4"

BRIGHTNESS = -0.05   # <0 darker, >0 brighter
CONTRAST   = 1.1    # 1.0 = no change, >1 increases contrast

# Optional CLI overrides:
#   python tune_light.py [brightness] [contrast]
#   e.g., python tune_light.py -0.06 1.12
if len(sys.argv) >= 2:
    BRIGHTNESS = float(sys.argv[1])
if len(sys.argv) >= 3:
    CONTRAST = float(sys.argv[2])

cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
    "-i", str(VIDEO),
    "-map_metadata", "0",
    "-movflags", "+faststart",
    "-vf", f"eq=brightness={BRIGHTNESS}:contrast={CONTRAST}",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
    "-c:a", "copy",
    str(OUTPUT)
]

subprocess.run(cmd, check=True)
print(f"Adjusted brightness={BRIGHTNESS}, contrast={CONTRAST} → {OUTPUT}")
