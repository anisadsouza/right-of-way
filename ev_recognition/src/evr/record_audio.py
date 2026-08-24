from __future__ import annotations

import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio samples for siren/ambient training.")
    parser.add_argument("label", help="Class name, usually siren or ambient")
    parser.add_argument("--out", type=Path, default=Path("data/audio"))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.out / args.label
    target.mkdir(parents=True, exist_ok=True)
    file_path = target / f"{args.label}_{int(time.time() * 1000)}.wav"

    print(f"Recording {args.seconds:.1f}s of {args.label} audio...")
    audio = sd.rec(int(args.seconds * args.sample_rate), samplerate=args.sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(file_path, audio, args.sample_rate)
    print(f"Saved {file_path}")


if __name__ == "__main__":
    main()
