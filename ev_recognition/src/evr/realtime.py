from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from evr.audio import AudioClassifier
from evr.config import load_config
from evr.controller import CsvLogger, make_controller
from evr.fusion import EmergencyFusion
from evr.vision import VisionClassifier, estimate_approach_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime emergency vehicle recognition.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--video", type=Path, help="Optional video file instead of camera.")
    parser.add_argument("--audio", type=Path, help="Optional audio file instead of microphone.")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio classifier.")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision classifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    vision = None
    audio = None
    if not args.no_vision and config.vision_model.exists():
        vision = VisionClassifier(config.vision_model, config.vision_labels, config.vision_input_size)
    elif not args.no_vision:
        print("Vision model not found. Run: python -m evr.train_vision")

    if not args.no_audio and config.audio_model.exists():
        audio = AudioClassifier(
            config.audio_model,
            config.audio_labels,
            config.audio_sample_rate,
            config.audio_mfcc_count,
        )
    elif not args.no_audio:
        print("Audio model not found. Run: python -m evr.train_audio")

    if vision is None and audio is None:
        raise SystemExit("No model available. Train vision/audio models or pass a valid config.")

    fusion = EmergencyFusion(config.fusion)
    controller = make_controller(
        config.controller.mode,
        config.serial.port,
        config.serial.baudrate,
        config.controller.green_pin,
        config.controller.yellow_pin,
        config.controller.red_pin,
        config.controller.buzzer_pin,
    )
    logger = CsvLogger(config.log_csv)
    audio_source = AudioSource(config.audio_sample_rate, config.audio_window_seconds, args.audio)

    cap = cv2.VideoCapture(str(args.video) if args.video else config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video source.")

    if audio is not None:
        audio_source.start()

    frame_count = 0
    last_vision = ("none", 0.0, None)
    last_audio = ("none", 0.0, None)
    last_command = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            side = "unknown"

            if vision is not None and frame_count % config.vision_every_n_frames == 0:
                v_label, v_conf, v_score = vision.predict(frame)
                side = estimate_approach_side(frame)
                last_vision = (v_label, v_conf, v_score)

            if audio is not None:
                chunk = audio_source.latest()
                if chunk is not None:
                    a_label, a_conf, a_score = audio.predict(chunk)
                    last_audio = (a_label, a_conf, a_score)

            result = fusion.update(last_vision[2], last_audio[2])
            if result.command != last_command:
                controller.send(result.command)
                last_command = result.command

            timestamp = time.time()
            logger.write(
                timestamp,
                last_vision[0],
                float(last_vision[2] or 0.0),
                last_audio[0],
                float(last_audio[2] or 0.0),
                result.emergency_confidence,
                result.state.value,
                result.command,
                side,
            )

            if config.show_window:
                draw_overlay(frame, last_vision, last_audio, result.emergency_confidence, result.state.value, side)
                cv2.imshow("Emergency Vehicle Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        audio_source.stop()
        cap.release()
        logger.close()
        controller.close()
        cv2.destroyAllWindows()


class AudioSource:
    def __init__(self, sample_rate: int, window_seconds: float, audio_file: Path | None) -> None:
        self.sample_rate = sample_rate
        self.window = int(sample_rate * window_seconds)
        self.audio_file = audio_file
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.stream: sd.InputStream | None = None
        self.thread: threading.Thread | None = None
        self.stopped = threading.Event()

    def start(self) -> None:
        if self.audio_file:
            self.thread = threading.Thread(target=self._file_loop, daemon=True)
            self.thread.start()
            return

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.window,
            callback=self._mic_callback,
        )
        self.stream.start()

    def latest(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                latest = self.queue.get_nowait()
            except queue.Empty:
                return latest

    def stop(self) -> None:
        self.stopped.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status)
        self._put(indata[:, 0].copy())

    def _file_loop(self) -> None:
        audio, sample_rate = sf.read(self.audio_file, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != self.sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        index = 0
        while not self.stopped.is_set():
            chunk = audio[index : index + self.window]
            if len(chunk) < self.window:
                index = 0
                continue
            self._put(chunk.copy())
            index += self.window
            time.sleep(self.window / self.sample_rate)

    def _put(self, chunk: np.ndarray) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(chunk)


def draw_overlay(frame, vision, audio, fused: float, state: str, side: str) -> None:
    rows = [
        f"Vision: {vision[0]} ev={float(vision[2] or 0):.2f}",
        f"Audio:  {audio[0]} siren={float(audio[2] or 0):.2f}",
        f"Fused:  {fused:.2f}  State: {state}",
        f"Approach side: {side}",
    ]
    color = (0, 0, 255) if state != "NORMAL" else (0, 180, 0)
    cv2.rectangle(frame, (10, 10), (430, 128), (20, 20, 20), -1)
    for i, text in enumerate(rows):
        cv2.putText(frame, text, (22, 38 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


if __name__ == "__main__":
    main()
