#!/usr/bin/env python3
"""SLEEPLESS 300: render one continuous MP3 per mode, only pre-recorded Sui whispers.
All 301 numbers are validated. No browser TTS, no client-side timing."""
from __future__ import annotations
import json, math, subprocess, sys, re
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 24000
INTRO_SECONDS = 170
SLOT_SECONDS = 10
TRACK_SECONDS = INTRO_SECONDS + 301 * SLOT_SECONDS + 18
MODES = ("mix", "lover", "cuddle", "suggestion", "love", "numbers")

def decode(path):
    raw = subprocess.check_output([FFMPEG,"-nostdin","-hide_banner","-loglevel","error",
      "-i",str(path),"-ac","1","-ar",str(SR),"-f","f32le","pipe:1"])
    a = np.frombuffer(raw,dtype="<f4").copy()
    if len(a)<SR//5: raise ValueError("Empty voice: "+str(path))
    return a

@lru_cache(maxsize=240)
def source(filename):
    return decode(ROOT/filename)

manifest=[]
for part in range(4):
    manifest += json.loads((ROOT/f"sui-v4-whisper-manifest-part{part}.json").read_text())
blocks={m["block"]:m for m in manifest if m["lang"]=="ja"}

def numeral(n):
    if n<0 or n>100: raise ValueError(n)
    if 0<n<100 and n%10:
        return source(f"sui-single-v4-ja-{n:03}.mp3")
    block=(100-n)//10
    m=blocks[block]
    info=next((c for c in m["clips"] if c["n"]==n),None)
    if info is None: raise ValueError(f"Missing Sui voiced count {n}")
    audio=source(f"sui-whisper-v4-ja-b{block:02}.mp3")
    a=max(0,int((info["start"]-.02)*SR))
    b=min(len(audio),int((info["end"]+.03)*SR))
    if b<=a:raise ValueError(n)
    return audio[a:b].copy()

def trim(a):
    # Remove only near-silent heads/tails; preserve soft breaths and syllables.
    rms=np.sqrt(np.convolve(np.square(a[::120],dtype=np.float64),np.ones(7)/7,"same"))
    ix=np.flatnonzero(rms>max(.0005,float(np.max(rms))*.065))
    if ix.size<2:return a
    begin=max(0,int(ix[0]*120)-int(.10*SR))
    end=min(len(a),int(ix[-1]*120)+int(.18*SR))
    return a[begin:end]

prefix200=trim(source("sui-300-prefix-200.mp3"))
prefix300=trim(source("sui-300-prefix-300.mp3"))
love_whisper=source("sui-300-love-v2.mp3")
short_love=source("sui-300-love-short-v2.mp3")
assert .35<len(prefix200)/SR<7 and .35<len(prefix300)/SR<7
assert 5.0<len(love_whisper)/SR<12
assert 1.0<len(short_love)/SR<4.5

# Composite numbers 101–299 use the same Sui genuine whispered recordings,
# with the natural pronounced hundred prefix followed by Sui's whispered 1–99.
@lru_cache(maxsize=310)
def count_clip(n):
    if n<=100:return trim(numeral(n))
    if n==200:return prefix200
    if n==300:return prefix300
    if n<200: head=trim(numeral(100)); tail=trim(numeral(n-100))
    else:head=prefix200;tail=trim(numeral(n-200))
    gap=np.zeros(int(.11*SR),dtype=np.float32)
    return np.concatenate((head,gap,tail)).astype(np.float32)

def add(mix, clip, sec, level=.55, max_seconds=None):
    if max_seconds is not None and len(clip)/SR>max_seconds:
        raise ValueError(f"Spoken phrase too long: {len(clip)/SR:.2f}s > {max_seconds}")
    at=round(sec*SR)
    x=clip.copy()
    rms=float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
    if rms<.0004:raise ValueError("Silent Sui voice")
    gain=min(10.0,.065/rms)*level
    x=(np.tanh(x*gain/.72)*.72).astype(np.float32)
    fade=min(int(.03*SR),len(x)//4)
    if fade:
        x[:fade]*=np.linspace(0,1,fade,dtype=np.float32)
        x[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    end=at+len(x)
    if end>len(mix): raise ValueError("Voice beyond audio boundary")
    mix[at:end]+=x

def main():
    # Refuse publication unless all numbers exist as Sui recordings and
    # each entire composite fits into its slot without clipping.
    lengths={}
    for n in range(300,-1,-1):
        a=count_clip(n)
        length=len(a)/SR
        if not .25<length<5.0:raise ValueError(f"Count {n} bad length {length:.2f}")
        lengths[n]=round(length,3)
    assert list(lengths)==list(range(300,-1,-1))
    assert len(lengths)==301
    print("Verified 301 Sui whispered number positions, 300 through 0",flush=True)

    intro=source("sui-300-intro-heartfelt-v2.mp3")
    assert 30<len(intro)/SR<45, "Opening must be complete, never sliced"
    breathing=source("sui-v142-breath-guide.mp3")
    inh=source("sui-v142-inhale.mp3")
    exh=source("sui-v142-exhale.mp3")
    comfort=[source(f"sui-comfort-atomic-{i:02}.mp3") for i in range(14)]
    # Indices 0–13 correspond to verified Japanese hypnosis and comfort phrases.
    groups={
      "mix":[0,6,9,13,3,4,7,11,8,12,2,5],
      "lover":[9,13,6,7,11,8],
      "cuddle":[8,9,13,3,6],
      "suggestion":[0,1,2,3,4,5,7,10,11,12],
      "love":[13,9,6,8],
      "numbers":[]
    }
    for mode in MODES:
        mix=np.zeros(int(TRACK_SECONDS*SR),dtype=np.float32)
        # Entire heartfelt Sui introduction: never slice it at an arbitrary timestamp.
        # Spoken sections get dedicated non-overlapping windows.
        add(mix,intro,0,.78,max_seconds=45)
        add(mix,breathing,48,.65,max_seconds=25)
        for cycle in range(3):
            start=78+19*cycle
            add(mix,inh,start,.55,max_seconds=6)
            add(mix,exh,start+7,.55,max_seconds=9)
        add(mix,love_whisper,151,.52,max_seconds=12)
        log=[]
        for k,n in enumerate(range(300,-1,-1)):
            when=INTRO_SECONDS + k*SLOT_SECONDS
            add(mix,count_clip(n),when,.64*(.95-.18*k/300),max_seconds=5)
            log.append({"n":n,"time_sec":when,"duration_sec":lengths[n]})
            if mode!="numbers" and n>0 and k%4==2:
                if mode in ("mix","love") and k%16==2:
                    clip=short_love
                else:
                    choices=groups[mode]
                    clip=comfort[choices[(k//4)%len(choices)]]
                # No overlaps with next numeral, keep the entire whispered sentence.
                if len(clip)/SR <=4.6:
                    add(mix,clip,when+5.1,.48,max_seconds=4.6)
        # Soft ending, never fade the last number itself.
        peak=np.max(np.abs(mix))
        if peak>0.89: mix*=.89/peak
        out=ROOT/f"sleepless-300-{mode}.mp3"
        p=subprocess.Popen([FFMPEG,"-y","-nostdin","-hide_banner","-loglevel","error",
           "-f","f32le","-ar",str(SR),"-ac","1","-i","pipe:0",
           "-c:a","libmp3lame","-b:a","64k",str(out)],stdin=subprocess.PIPE)
        p.communicate(mix.astype("<f4").tobytes())
        if p.returncode:raise RuntimeError("MP3 encode failed "+mode)
        probe=subprocess.run([FFMPEG,"-hide_banner","-i",str(out)],
            capture_output=True,text=True)
        match=re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)",probe.stderr)
        if not match:raise ValueError("MP3 duration unavailable "+mode)
        hh,mm,ss=match.groups()
        duration=int(hh)*3600+int(mm)*60+float(ss)
        size=out.stat().st_size
        if not TRACK_SECONDS-2<duration<TRACK_SECONDS+2:raise ValueError((mode,duration))
        if size<2000000:raise ValueError("Too small "+mode)
        (ROOT/f"sleepless-300-{mode}.json").write_text(json.dumps(log,ensure_ascii=False),encoding="utf-8")
        print("PASS",mode,"duration",duration,"bytes",size,flush=True)
    print("All six Sui-only modes mastered; all 301 numbers present in every track; complete heartfelt intro and clean love phrases")
if __name__=="__main__":main()
