#!/usr/bin/env python3
"""Create short uninterrupted daytime Sui sessions from the proven 1.4.7 master.
No live timers, no browser voices, no extra number synthesis.
"""
from pathlib import Path
import subprocess
import json
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent.parent
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
SRC=ROOT/"sleep-background-full.mp3"
assert SRC.exists() and SRC.stat().st_size>1000000
for minutes in (10,15,20):
    duration=minutes*60
    dst=ROOT/f"sleepless-day-{minutes}.mp3"
    # Preserve all the original Sui introduction and breath guidance; fade
    # down smoothly at the intended time. No sudden alarm or spoken wake-up.
    fade_start=duration-18
    command=[FFMPEG,"-nostdin","-y","-hide_banner","-loglevel","error",
      "-i",str(SRC),"-t",str(duration),
      "-af",f"afade=t=out:st={fade_start}:d=18",
      "-ac","1","-ar","24000","-codec:a","libmp3lame","-b:a","64k",str(dst)]
    subprocess.run(command,check=True)
    assert dst.stat().st_size>300000
    print("PASS",dst.name,dst.stat().st_size)
