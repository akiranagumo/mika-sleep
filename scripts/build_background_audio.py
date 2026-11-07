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
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 32000
OUT = ROOT / "sleep-background-full.mp3"


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
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    b64_digits = extract(r"const AUDIO_B64='([^']+)'", html, "digit audio")
    b64_auto = extract(r'const AUTO_B64="([^"]+)"', html, "autogenic audio")
    b64_intro = extract(
        r'<audio id="introAudio"[^>]+src="data:audio/[^;]+;base64,([^"]+)"',
        html, "introduction audio"
    )
    digit_spans = json.loads(extract(r"const SEGMENTS=(\[[^;]+\]);", html, "digit spans"))
    auto_spans = json.loads(extract(r"const AUTO_SEGMENTS=(\[[^;]+\]);", html, "autogenic spans"))
    whispers = json.loads((ROOT / "sui-whisper-segments.json").read_text("utf-8"))
    if (len(digit_spans), len(auto_spans), len(whispers)) != (101, 10, 101):
        raise ValueError("Expected 101 digit, 10 autogenic and 101 whisper spans")

    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)

        def from_b64(name: str, encoded: str):
            path = temp / f"{name}.mp3"
            path.write_bytes(base64.b64decode(encoded))
            return decode(path)

        intro = from_b64("intro", b64_intro)
        legacy_digits = from_b64("digits", b64_digits)
        autogenic = from_b64("autogenic", b64_auto)

        memo = {}
        def asset(filename: str):
            if filename not in memo:
                memo[filename] = decode(ROOT / filename)
            return memo[filename]

        whisper_audio = asset("sui-count-whisper.mp3")
        breath_intro = asset("sui-breath-new-intro.mp3")
        inhale = asset("sui-breath-new-inhale.mp3")
        exhale = asset("sui-cue-exhale.mp3")
        repeat = asset("sui-cue-again.mp3")
        spoken_digits = [asset(f"sui-digit-{i}.mp3") for i in range(1, 7)]
        nature = asset("sui-nature-night-clean.mp3")
        suggestions = [
            asset(f"sui-suggestion-{word}.mp3")
            for word in ["eyelids", "body", "thoughts", "rest", "nothinking",
                         "restmore", "quiet", "effort", "sink", "flow", "safe"]
        ] + [asset(f"sui-whisper-{word}.mp3") for word in ["rest", "thoughts", "calm"]]

        intro_duration = len(intro) / SR + 0.4
        breath_start = intro_duration
        nature_start = breath_start + 193
        count_start = nature_start + 86
        tail_start = count_start + 1010
        total_seconds = tail_start + 600
        mix = np.zeros(round(total_seconds * SR) + SR, dtype=np.float32)
        print(f"Length: {total_seconds:.1f}s, {len(mix)} samples", flush=True)

        def add(track, when, *, gain=1.0, start=0.0, duration=None):
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
            boost = min(6.0, max(1.0, 0.11 / max(rms, 0.001)))
            if peak > 0:
                boost = min(boost, 0.82 / peak)
            fade = min(round(SR * 0.055), count // 2)
            levels = np.full(count, 0.85 * gain * boost, dtype=np.float32)
            if fade:
                levels[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                levels[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            mix[at:at + count] += part * levels

        add(intro, 0)
        add(breath_intro, breath_start + 4)
        for cycle in range(3):
            t = breath_start + 24 + 60 * cycle
            add(inhale, t)
            for n in range(1, 5):
                add(spoken_digits[n - 1], t + 5 + 2 * n)
            add(exhale, t + 22)
            for n in range(1, 7):
                add(spoken_digits[n - 1], t + 25 + 2 * n)
            if cycle < 2:
                add(repeat, t + 47)
        add(nature, nature_start + 4)

        phrases = {96: 0, 91: 1, 86: 2, 76: 3, 71: 4,
                   66: 5, 56: 6, 46: 7, 36: 8, 26: 9}
        for i in range(101):
            t = count_start + i * 10
            number = 100 - i
            # Keep words gentle but intelligible throughout. The old 18%-at-end
            # curve buried the whisper under wind and phone speaker noise.
            quiet = 1.0 - 0.50 * pow(i / 100, 1.3)
            span = whispers[i]
            take = min(5.4, span["end"] - span["start"])
            if take > 0.1:
                add(whisper_audio, t, gain=quiet * 0.85,
                    start=span["start"], duration=take)
            else:
                span = digit_spans[i]
                add(legacy_digits, t, gain=quiet * 0.65,
                    start=span["start"], duration=min(5.3, span["end"] - span["start"]))
            if number in phrases:
                span = auto_spans[phrases[number]]
                add(autogenic, t + 4.7, gain=0.33 * quiet,
                    start=span["start"], duration=min(4.6, span["end"] - span["start"]))
            else:
                n = 1 if i < 10 else 2 if i < 40 else 3 if i < 60 else 5 if i < 80 else 10 if i < 90 else 0
                if i > 0 and n and (i < 10 or (i - 10) % n == 0):
                    clip = suggestions[(i * 7 + i // 10) % len(suggestions)]
                    add(clip, t + 4.8, gain=0.38 * quiet, duration=4.9)

        # Gentle 4% ambient wind, permanently mixed in so background playback
        # cannot depend on Web Audio, timers or another audio element.
        rng = np.random.default_rng(2608)
        period = 3 * SR
        noise = rng.uniform(-1, 1, period).astype(np.float32)
        loop = np.empty(period, dtype=np.float32)
        acc = 0.0
        for j in range(period):
            acc = 0.986 * acc + 0.014 * float(noise[j])
            loop[j] = acc * 3.8
        for begin in range(0, len(mix), 6 * SR):
            end = min(len(mix), begin + 6 * SR)
            now = np.arange(begin, end, dtype=np.float32) / SR
            fade_in = np.minimum(1.0, now / 4.0)
            fade_out = np.minimum(1.0, np.maximum(0, total_seconds - now) / 120.0)
            indexes = np.arange(begin, end) % period
            mix[begin:end] += loop[indexes] * (0.04 * 0.45 * fade_in * fade_out)

        # These are speech-only checkpoints, before ambient noise is added.
        # Catch a broken/quiet source mix instead of publishing wind alone.
        for title, at in [("intro", 0), ("count_start", count_start),
                          ("count_middle", count_start + 500),
                          ("count_late", count_start + 900)]:
            seg = mix[round(at * SR): round((at + 8) * SR)]
            level = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
            print(f"Voice checkpoint {title}: RMS {level:.5f}", flush=True)
            if level < 0.008:
                raise RuntimeError(f"Speech too quiet at {title}: {level:.5f}")

        raw = temp / "sleep.raw"
        np.clip(mix, -0.98, 0.98).astype("<f4").tofile(raw)
        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw),
            "-c:a", "libmp3lame", "-b:a", "80k", "-write_xing", "1", str(OUT)
        ], check=True)
        size_mb = OUT.stat().st_size / 1048576
        print(f"Created {OUT.name}: {size_mb:.1f} MiB", flush=True)
        if not 1 < size_mb < 90:
            raise RuntimeError("Unexpected size; refusing to commit")


if __name__ == "__main__":
    main()
