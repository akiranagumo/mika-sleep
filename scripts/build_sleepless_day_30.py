#!/usr/bin/env python3
"""Compose 30-minute daytime Sui whisper + real nature field recordings.

No numeric countdown and no text-to-speech substitutions.
All prerecorded Sui whispers; Commons public domain recordings.
Nature: birds in forest, forest creek, rain.
"""
import subprocess, json
from pathlib import Path
import imageio_ffmpeg
R=Path(__file__).resolve().parent.parent
F=imageio_ffmpeg.get_ffmpeg_exe()
LENGTH=1800
# existing Sui audio sources, informal dialogue
voices=[
 "sui-300-intro-informal-v3.mp3",
 "sui-300-breath-informal-v3.mp3",
 "sui-v142-nature.mp3",
 "sui-300-suggestion-informal-v3.mp3",
 "sui-300-cuddle-informal-v3.mp3",
 "sui-300-love-informal-v3.mp3",
 "sui-300-suggestion-short-v3.mp3",
 "sui-300-cuddle-short-v3.mp3",
 "sui-300-love-short-informal-v3.mp3"]
natures=["day-birds.ogg","day-creek.ogg","day-rain.ogg"]
for name in natures+voices:
    file=R/name
    assert file.exists() and file.stat().st_size>15000, f"Missing {name}"
# all starts in seconds, never interrupt existing sentences and never overlap
speech=[
 (0,0),(48,1),(145,2),(250,3),(340,2),(440,4),
 (555,2),(660,6),(755,2),(840,1),(930,4),(1050,2),
 (1135,3),(1250,2),(1350,5),(1510,2),(1640,7),(1730,8)]
assert speech==sorted(speech)
for suffix,levels in [("balanced",(0.055,0.075,0.060)),
                       ("nature-rich",(0.085,0.110,0.090))]:
    args=[F,"-hide_banner","-loglevel","error","-nostdin","-y"]
    for name in natures:
        args += ["-stream_loop","-1","-i",str(R/name)]
    for name in voices: args += ["-i",str(R/name)]
    fcs=[]
    fcs.append(f"[0:a]aresample=24000,aformat=channel_layouts=mono,atrim=duration=1800,volume={levels[0]},afade=t=in:st=0:d=20,afade=t=out:st=490:d=90,atrim=duration=580[b]")
    fcs.append(f"[1:a]aresample=24000,aformat=channel_layouts=mono,atrim=duration=1400,volume={levels[1]},afade=t=in:st=0:d=25,afade=t=out:st=1340:d=60[c]")
    fcs.append(f"[2:a]aresample=24000,aformat=channel_layouts=mono,atrim=duration=1010,volume={levels[2]},afade=t=in:st=0:d=80,afade=t=out:st=930:d=80,adelay=790000:all=1[r]")
    # 18 spoken Sui lines; whisper never competes with nature soundtrack
    labels=["[b]","[c]","[r]"]
    for j,(when,voice_idx) in enumerate(speech):
        input_idx=voice_idx+3
        fcs.append(f"[{input_idx}:a]aresample=24000,aformat=channel_layouts=mono,volume=0.68,adelay={when*1000}:all=1[v{j}]")
        labels.append(f"[v{j}]")
    # The same input voice is used at multiple timestamps. FFmpeg must split
    # shared inputs explicitly, otherwise only one filter can consume them.
    for vi in range(len(voices)):
        occurrences=[j for j,(_,used) in enumerate(speech) if used==vi]
        if len(occurrences)>1:
            lab=f"[{vi+3}:a]"
            outputs="".join(f"[s{vi}_{j}]" for j in occurrences)
            fcs.insert(0,f"{lab}asplit={len(occurrences)}{outputs}")
            for j in occurrences:
                fcs=[x.replace(f"[{vi+3}:a]aresample=24000,aformat=channel_layouts=mono,volume=0.68,adelay={speech[j][0]*1000}:all=1[v{j}]",
                    f"[s{vi}_{j}]aresample=24000,aformat=channel_layouts=mono,volume=0.68,adelay={speech[j][0]*1000}:all=1[v{j}]") for x in fcs]
    fcs.append("".join(labels)+f"amix=inputs={len(labels)}:duration=longest:normalize=0,alimiter=limit=0.85,atrim=duration={LENGTH},afade=t=out:st=1770:d=30[out]")
    out=R/f"sleepless-day-30-{suffix}.mp3"
    args+=["-filter_complex",";".join(fcs),"-map","[out]","-t","1800","-c:a","libmp3lame","-b:a","64k",str(out)]
    subprocess.run(args,check=True,timeout=1200)
    assert 12_000_000<out.stat().st_size<19_000_000,(out,out.stat().st_size)
    print("PASS",out.name,out.stat().st_size,flush=True)
print("30-minute all-Sui narrated natural-sound audio ready")
