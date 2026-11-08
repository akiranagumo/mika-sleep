#!/usr/bin/env python3
"""Build a continuous background-playable Sui sleep MP3 from the source assets.

The old v12.html used Web Audio scheduling tied to requestAnimationFrame.
This renders the same material at build time, so iOS only needs one <audio>.
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 32000
# Voice and ambient are mixed into alternative, equal-length files.
# iOS Safari can background-play a single HTMLAudioElement; swapping the
# source while the page is visible allows wind selection without WebAudio.
# Wind gains are audio mixing coefficients, NOT percentage of perceived
# loudness. Earlier values (0.002..0.020) measured ~-75..-55 dBFS and were
# effectively inaudible on phones. This range reaches ~-35 dBFS at the top.
PRESETS = [
    ("sleep-background-full.mp3", 0.0),       # off / Sui only
    ("sleep-background-wind-02.mp3", 0.04),   # gentle breeze
    ("sleep-background-wind-05.mp3", 0.08),   # light breeze
    ("sleep-background-wind-10.mp3", 0.14),   # moderate
    ("sleep-background-wind-20.mp3", 0.22),   # strong
]



def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1",
    ])
    data = np.frombuffer(raw, dtype="<f4").copy()
    if not len(data):
        raise ValueError(f"Empty audio: {path}")
    return data


def extract(pattern: str, src: str, name: str) -> str:
    m = re.search(pattern, src, flags=re.S)
    if not m:
        raise ValueError(f"Missing {name} in index.html")
    return m.group(1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "japanese"
    if mode not in ("japanese", "english"):
        raise ValueError("Expected japanese or english rendering mode")
    print(f"Sleep voice mode: {mode}", flush=True)
    # Each published sound track is ONE fully mastered MP3. Editing of
    # a spoken word or a comfort phrase is forbidden during playback.
    # Source count phrases were recorded in ten-word sections WITH real
    # 1.7-2.5s silent gaps; their safe boundaries are in the silence.
    # Comfort phrases are complete independent recordings.
    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)

        memo = {}
        def asset(filename: str):
            if filename not in memo:
                memo[filename] = decode(ROOT / filename)
            return memo[filename]

        # Read only the language being mastered; the output is a standalone,
        # monolithic MP3 for this language and each chosen wind preset.
        count_lang = "ja" if mode == "japanese" else "en"
        # All count numbers are Sui's original Eleven V4 whispered takes;
        # never fall back to the older multilingual_v2 normal speaking voice.
        v4_blocks = [asset(f"sui-whisper-v4-{count_lang}-b{b:02d}.mp3")
                     for b in range(11)]
        comfort_tracks = [asset(f"sui-comfort-atomic-{i:02d}.mp3")
                          for i in range(14)]
        intro = asset("sui-v142-opening.mp3")
        breath_intro = asset("sui-v142-breath-guide.mp3")
        inhale = asset("sui-v142-inhale.mp3")
        exhale = asset("sui-v142-exhale.mp3")
        repeat = asset("sui-v142-repeat.mp3")
        nature = asset("sui-v142-nature.mp3")
        announce = asset("sui-count-intro-ja.mp3" if mode == "japanese"
                         else "sui-count-intro-en.mp3")

        # Stable master timeline keeps the iPhone seek feature consistent.
        count_start = 308.8
        tail_start = count_start + 1010
        total_seconds = tail_start + 600
        mix = np.zeros(round(total_seconds * SR) + SR, dtype=np.float32)
        print(f"Length: {total_seconds:.1f}s, {len(mix)} samples", flush=True)
        count_checkpoints = []

        def add(track, when, *, gain=1.0, start=0.0, duration=None, target_rms=0.11,
                fade_seconds=0.055, audible_count=False):
            sample_offset = max(0, round(float(start) * SR))
            available = len(track) - sample_offset
            count = available if duration is None else min(available, max(0, round(duration * SR)))
            at = round(when * SR)
            if count < 10 or at < 0 or at >= len(mix):
                raise ValueError("Missing or empty Sui audio segment")
            count = min(count, len(mix) - at)
            part = track[sample_offset:sample_offset + count]
            rms = float(np.sqrt(np.mean(np.square(part, dtype=np.float64))))
            if audible_count:
                # Prior releases used a strict peak-based gain cap: a single
                # transient could force an otherwise soft whispered numeral
                # to be almost inaudible, despite passing metadata checks.
                # Normalize each full spoken number and gently soft-limit its
                # peaks, preserving the breathy quality without hard clipping.
                boost = min(450.0, max(1.0, 0.105 / max(rms, 0.0001)))
                processed = 0.55 * np.tanh((part * boost) / 0.55)
                # Small Japanese syllables like 'に' only last ~0.09s.
                # Their edges need shorter fades to remain intelligible.
                fade_seconds = min(fade_seconds, max(0.009, count / SR * 0.12))
                level = 0.85 * gain
            else:
                peak = float(np.max(np.abs(part)))
                boost = min(6.0, max(1.0, target_rms / max(rms, 0.001)))
                if peak > 0:
                    boost = min(boost, 0.82 / peak)
                processed = part
                level = 0.85 * gain * boost
            fade = min(round(SR * fade_seconds), count // 2)
            levels = np.full(count, level, dtype=np.float32)
            if fade:
                levels[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                levels[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            segment = processed * levels
            if audible_count:
                actual_rms = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
                if actual_rms < 0.012:
                    raise ValueError(f"Sui number at {when:.1f}s was inaudible: output={actual_rms:.5f}, raw={rms:.6f}, boost={boost:.1f}, duration={count/SR:.3f}")
                count_checkpoints.append((int(round((when-count_start)/10)), actual_rms))
            mix[at:at + count] += segment


        # Exactly 101 verified words for each language, transcribed from
        # original Sui Eleven V4 recordings with [whispers] delivery.
        # The clipped word boundaries include a small pre/post-roll. We trim
        # neither the reassuring phrases nor any part of the narration in
        # the browser; each preset is a single already-mastered MP3.
        voice_manifest=[]
        for part_number in range(4):
            voice_manifest.extend(json.loads(
                (ROOT/f"sui-v4-whisper-manifest-part{part_number}.json").read_text("utf-8")))
        blocks=sorted(
            (b for b in voice_manifest if b["lang"]==count_lang),
            key=lambda b:b["block"]
        )
        if len(blocks)!=11 or [b["block"] for b in blocks]!=list(range(11)):
            raise ValueError(f"Missing genuine Sui V4 whispered {count_lang} count blocks")
        count_clips={}
        for block,metadata in enumerate(blocks):
            recording=v4_blocks[block]
            wanted=min(10,101-10*block)
            if len(metadata["clips"])!=wanted:
                raise ValueError(f"Wrong number of Sui V4 whisper clips in {block}")
            for index,word in enumerate(metadata["clips"]):
                number=100-10*block-index
                if word["n"]!=number or number in count_clips:
                    raise ValueError(f"Skipped or repeated Sui V4 count {number}")
                left,right=float(word["start"]),float(word["end"])
                if not 0<=left<right<=len(recording)/SR+.04 or right-left>3.7:
                    raise ValueError(f"Invalid Sui V4 whisper word {number} boundary")
                right=min(right,len(recording)/SR)
                count_clips[number]=(recording,left,right,0.02,max(.07,right-left-.025))
            print(f"Sui V4 whisper {count_lang} block {block}: "
                  f"{wanted} complete spoken numbers, voice source checked",flush=True)
        if sorted(count_clips)!=list(range(101)):
            raise ValueError("Sui V4 whispered countdown must include every number 100 to 0")
        print("101 genuine Sui V4 whispered numbers confirmed: "
              f"{mode}, one completed MP3",flush=True)

        timing=[
            {"n":n,"start":round(count_clips[n][3],3),
             "end":round(count_clips[n][4],3)}
            for n in range(100,-1,-1)
        ]
        (ROOT/f"sui-master-timing-{count_lang}.json").write_text(
            json.dumps(timing,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

        def whisper_count(number: int, when: float, *, intensity=1.0, language=None):
            if number not in count_clips:
                raise ValueError(f"Missing complete Sui spoken count {number}")
            if language not in (None,mode):
                raise ValueError("Do not mix different count languages")
            track,start,end,voice_begin,voice_end=count_clips[number]
            # Sleep-first mix: counts must sit BELOW the reassuring speech,
            # not jump forward like an announcer. -6.6 dB at 100, then
            # another -1.9 dB by zero. Keep comfort sentences unchanged.
            progress=(100-number)/100
            extra_fade=1.0-0.20*progress**1.20
            add(track,when,start=start,duration=end-start,
                gain=.42*intensity*extra_fade,target_rms=.072,
                fade_seconds=.012,audible_count=(language is None))

        def comfort_whisper(phrase_idx: int, when: float, intensity: float):
            """Play a complete standalone reassuring sentence, NEVER trim."""
            idx=phrase_idx%len(comfort_tracks)
            track=comfort_tracks[idx]
            duration=len(track)/SR
            if not 1.6<duration<4.8:
                raise ValueError(f"Bad full Sui sentence {idx}: {duration:.2f}s")
            if 3.4+duration>=9.0:
                raise ValueError(f"Full Sui sentence overlaps next count {idx}")
            add(track,when+3.4,
                gain=.60*intensity,target_rms=.07,
                fade_seconds=.012)

        # One consistently whispered Sui voice for the opening, all cues,
        # relaxation, every count and every gentle phrase.
        add(intro, 0, gain=0.85, target_rms=0.075)
        add(breath_intro, 16, gain=0.85, target_rms=0.075)
        # Three unhurried breathing cycles. No out-of-sync spoken number
        # overlays, no normal-voice cues; 4s inhale, brief hold, 6s exhale.
        for cycle in range(3):
            t = 42 + 17 * cycle
            add(inhale, t, gain=0.8, target_rms=0.065)
            add(exhale, t + 6.2, gain=0.8, target_rms=0.065)
            if cycle < 2:
                add(repeat, t + 12.0, gain=0.72, target_rms=0.06)
        add(nature, 105, gain=0.78, target_rms=0.065)
        # Quiet autogenic training continues until the countdown.
        # Each reassuring line is complete and has room to breathe.
        for j in range(12):
            comfort_whisper(j, 142 + j * 12, 0.88)
        # Countdown-only playback starts at this announcement.
        # Exactly five seconds of silence separate the END of the spoken
        # introduction from the first count (100), in either language.
        announce_duration = len(announce) / SR
        prelude_start = count_start - 5.0 - announce_duration
        if not 2.0 <= announce_duration <= 5.0:
            raise ValueError(f"Bad Sui countdown intro duration: {announce_duration}")
        add(announce, prelude_start, gain=0.80, target_rms=0.07)
        print(f"Countdown prelude: mode={mode}, begin={prelude_start:.3f}, "
              f"voice_end={prelude_start+announce_duration:.3f}, "
              f"number_100={count_start:.3f}, quiet_gap=5.0s",flush=True)

        # A whispered number every ten seconds. A personal whispered phrase
        # follows some numbers, becoming less frequent as sleep deepens.
        # All phrases are COMPLETE separate recordings, never clipped at a
        # fixed 4.6 or 4.9 second cutoff as in the original version.
        phrases_used = 0
        for i in range(101):
            number = 100 - i
            when = count_start + i * 10
            gentle_fade = 1.0 - 0.18 * pow(i / 100, 1.25)
            whisper_count(number, when, intensity=gentle_fade)
            say_phrase = (i < 30 and i % 2 == 1) or (
                30 <= i < 70 and i % 3 == 1) or (
                i >= 70 and i % 4 == 1)
            if say_phrase:
                comfort_whisper(phrases_used, when, gentle_fade)
                phrases_used += 1
        assert phrases_used >= 30, f"Insufficient reassurance whispers: {phrases_used}"
        if len(count_checkpoints) != 101:
            raise RuntimeError(f"Missing audible Sui numbers: {len(count_checkpoints)} / 101")
        if [i for i, level in count_checkpoints] != list(range(101)):
            raise RuntimeError("Count sequence skipped or reordered in the final master")
        print("101 audible Sui spoken count events (quiet mix), min RMS={:.4f}, max RMS={:.4f}".format(
            min(level for _,level in count_checkpoints),
            max(level for _,level in count_checkpoints)), flush=True)
        print(f"Whisper QA: 101 full Sui whisper counts, "
              f"{phrases_used} complete gentle reassurance phrases; "
              f"count language={mode}; all numbers whispered", flush=True)

        # Validate Sui-only audio BEFORE adding wind: ambient must never mask
        # missing or weak speech clips.  The other files are pure remixes.
        for title, at in [("intro", 0), ("count_start", count_start),
                          ("count_middle", count_start + 500),
                          ("count_late", count_start + 900)]:
            # Count numbers are very short (< 1.5 seconds): measuring an
            # eight-second gap punishes a gentle whisper with silence padding.
            # Inspect the actual 2-second vocal window instead.
            probe = 8 if title == "intro" else 2
            seg = mix[round(at * SR): round((at + probe) * SR)]
            level = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
            print(f"Voice checkpoint {title}: {probe}s RMS {level:.5f}", flush=True)
            if level < (0.0045 if title == "intro" else 0.003):
                raise RuntimeError(f"Speech too quiet at {title}: {level:.5f}")

        # Keep the ambient bed identical across all settings to preserve
        # a stable listening experience when switching volume presets.
        rng = np.random.default_rng(2608)
        period = 3 * SR
        noise = rng.uniform(-1, 1, period).astype(np.float32)
        loop = np.empty(period, dtype=np.float32)
        acc = 0.0
        for j in range(period):
            acc = 0.986 * acc + 0.014 * float(noise[j])
            loop[j] = acc * 3.8

        # Render short chunks rather than allocating five full mixes in memory.
        # All outputs share exact sample count and MP3 encoding settings.
        chunk = 6 * SR
        for filename, wind_level in PRESETS:
            outfile = ROOT / (filename if mode == "japanese" else filename.replace("sleep-background", "sleep-english"))
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "80k", "-write_xing", "1",
                str(outfile)
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for begin in range(0, len(mix), chunk):
                    end = min(len(mix), begin + chunk)
                    audio = mix[begin:end].copy()
                    if wind_level > 0:
                        now = np.arange(begin, end, dtype=np.float32) / SR
                        fade_in = np.minimum(1.0, now / 0.8)
                        fade_out = np.minimum(1.0, np.maximum(0, total_seconds - now) / 120.0)
                        indexes = np.arange(begin, end) % period
                        audio += loop[indexes] * (wind_level * 0.45 * fade_in * fade_out)
                    process.stdin.write(np.clip(audio, -0.98, 0.98).astype("<f4").tobytes())
            finally:
                if process.stdin:
                    process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed rendering {filename}")
            size_mb = outfile.stat().st_size / 1048576
            estimated_wind_rms = 0.45 * wind_level * float(np.sqrt(np.mean(np.square(loop, dtype=np.float64))))
            print(f"Created {filename}: {size_mb:.1f} MiB, wind mix={wind_level:.3f}, wind_rms={estimated_wind_rms:.5f}", flush=True)
            if not 1 < size_mb < 90:
                raise RuntimeError(f"Unexpected output size for {filename}")


if __name__ == "__main__":
    main()
