"""Run saved siren scores through the real EmergencyFusion state machine.

Each ESC-50 test clip (5 s) is scored with the audio model on 1 s windows with a
0.5 s hop, then replayed through a fresh EmergencyFusion. The fusion is updated
at a fixed rate (--fps), as the realtime loop does once per pass, using the
latest completed audio window.

Two vision cases are simulated:
  audio only         visual score is None (no camera information)
  vision 0.10        camera running but seeing no emergency vehicle

Run from ev_recognition/:
    python -m evr.simulate_fusion_audio --test-dir demo/esc50_test
"""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from evr.audio import AudioClassifier
from evr.config import load_config
from evr.fusion import EmergencyFusion, YieldState

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
MFCC_COUNT = 40
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
CLIP_SECONDS = 5.0
START_TIME = 1000.0


def window_scores(classifier, audio):
    size = int(SAMPLE_RATE * WINDOW_SECONDS)
    hop = int(SAMPLE_RATE * HOP_SECONDS)
    scores = []
    for start in range(0, len(audio) - size + 1, hop):
        _, _, score = classifier.predict(audio[start:start + size])
        scores.append(float(score))
    return scores


def replay(fusion_config, scores, fps, visual, extend_seconds=0.0):
    """Return (seconds until first non-NORMAL state, seconds until STOPPED); None if never.

    With extend_seconds > 0 the last audio score is held after the clip ends,
    which assumes the sound keeps going.
    """
    fusion = EmergencyFusion(fusion_config)
    fusion.state_started_at = START_TIME
    first_trigger, stop_time = None, None
    for step in range(int((CLIP_SECONDS + extend_seconds) * fps)):
        t = step / fps
        ready = int((t - WINDOW_SECONDS) // HOP_SECONDS) + 1 if t >= WINDOW_SECONDS else 0
        audio = scores[min(ready, len(scores)) - 1] if ready > 0 else None
        result = fusion.update(visual, audio, now=START_TIME + t)
        if result.state != YieldState.NORMAL and first_trigger is None:
            first_trigger = t
        if result.state == YieldState.STOPPED and stop_time is None:
            stop_time = t
    return first_trigger, stop_time


def median_text(values):
    values = [v for v in values if v is not None]
    return f"{np.median(values):.1f}" if values else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="demo/esc50_test")
    parser.add_argument("--model", default="models/audio_siren_classifier.joblib")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fps", type=float, nargs="+", default=[6.0, 15.0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.60, 0.80])
    parser.add_argument("--extend-seconds", type=float, default=0.0,
                        help="Keep feeding the last audio score this many seconds after each clip (sound continues)")
    args = parser.parse_args()

    test_dir = ROOT / args.test_dir
    wav_files = sorted(test_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files in {test_dir}. Run prepare_esc50_split.py first.")
    if not (test_dir / "siren_external.wav").exists():
        raise SystemExit(f"{test_dir / 'siren_external.wav'} is missing, so siren detection cannot be measured.")

    app = load_config(ROOT / args.config)
    classifier = AudioClassifier(
        ROOT / args.model, ROOT / "models" / "audio_labels.txt", SAMPLE_RATE, MFCC_COUNT
    )

    clips = []  # (category, is_siren, window scores)
    clip_size = int(SAMPLE_RATE * CLIP_SECONDS)
    for wav_path in wav_files:
        audio, rate = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{wav_path.name}: expected {SAMPLE_RATE} Hz, got {rate} Hz")
        if len(audio) % clip_size > SAMPLE_RATE // 10:
            print(f"Warning: {wav_path.name} is not a whole number of {CLIP_SECONDS:g} s clips; the remainder is ignored")
        for start in range(0, len(audio) - clip_size + 1, clip_size):
            scores = window_scores(classifier, audio[start:start + clip_size])
            clips.append((wav_path.stem, wav_path.stem == "siren_external", scores))

    negatives = [c for c in clips if not c[1]]
    sirens = [c for c in clips if c[1]]
    if not negatives or not sirens:
        raise SystemExit("Need both non-siren and siren clips to run the simulation.")
    print(f"Clips: {len(negatives)} non-siren, {len(sirens)} siren ({CLIP_SECONDS:g} s each)")
    print(f"Config: trigger_frames={app.fusion.trigger_frames}, clear_threshold={app.fusion.clear_threshold}, "
          f"configured emergency_threshold={app.fusion.emergency_threshold}")
    print("NORMAL-left = reached SLOW or beyond. Times are seconds from the start of a clip.")
    if args.extend_seconds > 0:
        print(f"Each clip's last audio score is held for {args.extend_seconds:g} s after the clip ends (sound assumed to continue).")
    else:
        print("Each clip is measured within its own 5 s only.")

    scenarios = [("audio only", None), ("vision 0.10", 0.10)]
    print("\nscenario | updates/s | threshold | non-siren left NORMAL | non-siren STOPPED | siren left NORMAL | siren STOPPED | median first reaction | median time to STOPPED")
    details = []
    for label, visual in scenarios:
        for fps in args.fps:
            for threshold in args.thresholds:
                config = replace(app.fusion, emergency_threshold=threshold)
                neg = [replay(config, c[2], fps, visual, args.extend_seconds) for c in negatives]
                pos = [replay(config, c[2], fps, visual, args.extend_seconds) for c in sirens]
                print(
                    f"{label} | {fps:g} | {threshold:.2f} | "
                    f"{sum(r[0] is not None for r in neg)}/{len(neg)} | {sum(r[1] is not None for r in neg)}/{len(neg)} | "
                    f"{sum(r[0] is not None for r in pos)}/{len(pos)} | {sum(r[1] is not None for r in pos)}/{len(pos)} | "
                    f"{median_text([r[0] for r in pos])} | {median_text([r[1] for r in pos])}"
                )
                left = Counter(c[0] for c, r in zip(negatives, neg) if r[0] is not None)
                stopped = Counter(c[0] for c, r in zip(negatives, neg) if r[1] is not None)
                missed = [i + 1 for i, r in enumerate(pos) if r[1] is None]
                details.append((label, fps, threshold, left, stopped, missed))

    print("\nNon-siren categories that caused a reaction, per setting (category: number of clips)")
    for label, fps, threshold, left, stopped, missed in details:
        if not left and not stopped:
            continue
        print(f"{label}, {fps:g} updates/s, threshold {threshold:.2f}")
        print("  reached STOPPED: " + (", ".join(f"{k} ({v})" for k, v in stopped.most_common()) or "none"))
        print("  left NORMAL:     " + (", ".join(f"{k} ({v})" for k, v in left.most_common()) or "none"))
    if all(not left and not stopped for _, _, _, left, stopped, _ in details):
        print("none for any setting")

    print("\nSiren clips that did not reach STOPPED, per setting (clip numbers in file order)")
    for label, fps, threshold, _, _, missed in details:
        print(f"{label}, {fps:g} updates/s, threshold {threshold:.2f}: " + (", ".join(map(str, missed)) or "none"))


if __name__ == "__main__":
    main()