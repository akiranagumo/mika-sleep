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
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    # The countdown uses only Sui's dedicated whisper recording.
    b64_intro = extract(
        r'<audio id="introAudio"[^>]+src="data:audio/[^;]+;base64,([^"]+)"',
        html, "introduction audio"
    )
    whispers = json.loads((ROOT / "sui-whisper-segments.json").read_text("utf-8"))
    new_whispers = json.loads((ROOT / "sui-whisper-japanese-segments.json").read_text("utf-8"))
    old_english_whispers = json.loads((ROOT / "sui-whisper-v2-segments.json").read_text("utf-8"))
    japanese_top = json.loads((ROOT / "sui-japanese-top-segments.json").read_text("utf-8"))
    english_top = json.loads((ROOT / "sui-english-top-segments.json").read_text("utf-8"))
    if any(len(a) != expected for a, expected in [
            (old_english_whispers,90),(japanese_top,11),(english_top,11)]):
        raise ValueError("Missing English/Japanese whisper source segments")
    if any(w["n"] != 100-i for i,w in enumerate(japanese_top)):
        raise ValueError("Japanese top digits are not 100..90")
    if any(w["n"] != 100-i for i,w in enumerate(english_top)):
        raise ValueError("English top digits are not 100..90")
    if any(w["n"] != 89-i for i,w in enumerate(old_english_whispers)):
        raise ValueError("English lower digits are not 89..0")
    comforts = json.loads((ROOT / "sui-comfort-whispers-v2-segments.json").read_text("utf-8"))
    if len(new_whispers) != 90 or any(w["n"] != 89-i for i,w in enumerate(new_whispers)):
        raise ValueError("Expected all 89..0 new Sui whisper numbers in order")
    kana_digits = ["ぜろ","いち","に","さん","よん","ご","ろく","なな","はち","きゅう"]
    def japanese_number(n):
        if n == 0:
            return kana_digits[0]
        if n < 10:
            return kana_digits[n]
        tens, ones = divmod(n, 10)
        return ("" if tens == 1 else kana_digits[tens]) + "じゅう" + (kana_digits[ones] if ones else "")
    for w in new_whispers:
        if w["reading"] != japanese_number(w["n"]):
            raise ValueError(f"English/mismatched count prevented: {w['n']}, {w['reading']}")

    if len(comforts) != 14:
        raise ValueError("Expected 14 Sui whispered relaxation phrases")
    # The final generated clip ends very close to the end of its source.
    # Use only the 13 earlier phrases, whose full spoken endings are intact.
    comforts = comforts[:13]
    if len(whispers) != 101:
        raise ValueError("Expected 101 timed Sui whisper count clips")
    for i, span in enumerate(whispers):
        duration = float(span["end"]) - float(span["start"])
        if not 0.45 <= duration <= 3.5:
            raise ValueError(f"Invalid whisper segment {100-i}: {duration}")
        if i and abs(float(span["start"]) - float(whispers[i-1]["end"])) > 0.3:
            raise ValueError(f"Unexpected whisper segment gap at {100-i}")

    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)

        def from_b64(name: str, encoded: str):
            path = temp / f"{name}.mp3"
            path.write_bytes(base64.b64decode(encoded))
            return decode(path)

        intro = from_b64("intro", b64_intro)
        # Keep the intro intact; countdown and breath numbers use Sui's whisper.

        memo = {}
        def asset(filename: str):
            if filename not in memo:
                memo[filename] = decode(ROOT / filename)
            return memo[filename]

        whisper_audio = asset("sui-count-whisper.mp3")
        new_whisper_audio = asset("sui-count-whisper-japanese.mp3")
        old_english_audio = asset("sui-count-whisper-v2.mp3")
        japanese_top_audio = asset("sui-japanese-top.mp3")
        english_top_audio = asset("sui-english-top.mp3")
        comfort_audio = asset("sui-comfort-whispers-v2.mp3")
        breath_intro = asset("sui-breath-new-intro.mp3")
        inhale = asset("sui-breath-new-inhale.mp3")
        exhale = asset("sui-cue-exhale.mp3")
        repeat = asset("sui-cue-again.mp3")
        nature = asset("sui-nature-night-clean.mp3")

        intro_duration = len(intro) / SR + 0.4
        breath_start = intro_duration
        nature_start = breath_start + 193
        count_start = nature_start + 86
        tail_start = count_start + 1010
        total_seconds = tail_start + 600
        mix = np.zeros(round(total_seconds * SR) + SR, dtype=np.float32)
        print(f"Length: {total_seconds:.1f}s, {len(mix)} samples", flush=True)

        def add(track, when, *, gain=1.0, start=0.0, duration=None, target_rms=0.11, fade_seconds=0.055):
            sample_offset = max(0, round(float(start) * SR))
            available = len(track) - sample_offset
            count = available if duration is None else min(available, max(0, round(duration * SR)))
            at = round(when * SR)
            if count < 10 or at < 0 or at >= len(mix):
                return
            count = min(count, len(mix) - at)
            part = track[sample_offset:sample_offset + count]
            # Sui clips have different recorded levels. Lift quiet speech toward
            # a consistent, audible level before applying the sleep fade.
            # A peak guard prevents clipping or an unexpectedly loud phrase.
            rms = float(np.sqrt(np.mean(np.square(part, dtype=np.float64))))
            peak = float(np.max(np.abs(part)))
            boost = min(6.0, max(1.0, target_rms / max(rms, 0.001)))
            if peak > 0:
                boost = min(boost, 0.82 / peak)
            fade = min(round(SR * fade_seconds), count // 2)
            levels = np.full(count, 0.85 * gain * boost, dtype=np.float32)
            if fade:
                levels[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                levels[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            mix[at:at + count] += part * levels


        def whisper_count(number: int, when: float, *, intensity=1.0, language=None):
            """Use actual Sui whispered numerals in the selected language."""
            if not 0 <= number <= 100:
                raise ValueError(f"Invalid count {number}")
            lang = language or mode
            if number >= 90:
                spans, track = ((japanese_top, japanese_top_audio) if lang == "japanese"
                                else (english_top, english_top_audio))
                span = spans[100-number]
            else:
                spans, track = ((new_whispers, new_whisper_audio) if lang == "japanese"
                                else (old_english_whispers, old_english_audio))
                span = spans[89-number]
            if span["n"] != number:
                raise ValueError(f"Wrong whisper number: expected {number}")
            start = float(span["start"])
            duration = float(span["end"]) - start
            if duration < 0.30 or duration > 3.5:
                raise ValueError(f"Invalid Sui whisper {number} ({duration:.3f}s)")
            add(track, when, start=start, duration=duration,
                gain=0.92 * intensity, target_rms=0.075,
                fade_seconds=0.08)

        def comfort_whisper(phrase_idx: int, when: float, intensity: float):
            """Restore one FULL, short reassuring whisper between counts."""
            segment = comforts[phrase_idx % len(comforts)]
            start = float(segment["start"])
            duration = float(segment["end"]) - start
            if not 1.5 < duration < 3.8:
                raise ValueError(f"Phrase {phrase_idx} too long or invalid ({duration})")
            # 10-second count intervals; the phrase starts at +3.4, and
            # must finish before the next number, never mid-word.
            if 3.4 + duration >= 8.5:
                raise ValueError(f"Comfort phrase overlaps next count: {phrase_idx}")
            add(comfort_audio, when + 3.4, start=start, duration=duration,
                gain=0.57 * intensity, target_rms=0.062,
                fade_seconds=0.09)

        add(intro, 0)
        add(breath_intro, breath_start + 4)
        for cycle in range(3):
            t = breath_start + 24 + 60 * cycle
            add(inhale, t)
            for n in range(1, 5):
                whisper_count(n, t + 5 + 2 * n, intensity=0.9, language="japanese")
            add(exhale, t + 22)
            for n in range(1, 7):
                whisper_count(n, t + 25 + 2 * n, intensity=0.9, language="japanese")
            if cycle < 2:
                add(repeat, t + 47)
        add(nature, nature_start + 4)

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
