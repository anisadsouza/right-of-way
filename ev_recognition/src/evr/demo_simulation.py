from __future__ import annotations

import argparse
import time
from pathlib import Path

from evr.config import load_config
from evr.controller import CsvLogger
from evr.fusion import EmergencyFusion


SCENARIOS = [
    ("normal_car", 0.10, 0.05, "center", 3.0),
    ("distant_siren_audio_only", 0.15, 0.92, "unknown", 4.0),
    ("normal_after_siren_noise", 0.12, 0.15, "unknown", 4.0),
    ("ambulance_visual_only", 0.93, 0.10, "left", 4.0),
    ("combined_ambulance_siren", 0.96, 0.95, "center", 5.0),
    ("emergency_passed", 0.08, 0.08, "unknown", 5.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-camera/no-mic demo of the fusion FSM.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--speed", type=float, default=1.0, help="Higher is faster. Use 4 for quick demo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fusion = EmergencyFusion(config.fusion)
    logger = CsvLogger(config.base_dir / "logs/demo_simulation_log.csv")

    frame_period = 0.25 / max(args.speed, 0.1)
    simulated_time = time.monotonic()
    print("Emergency Vehicle Recognition demo simulation")
    print("scenario, visual, audio, fused, state, command, side")

    try:
        for scenario, visual, audio, side, seconds in SCENARIOS:
            steps = max(1, int(seconds / 0.25))
            for _ in range(steps):
                simulated_time += 0.25
                result = fusion.update(visual, audio, now=simulated_time)
                print(
                    f"{scenario:26s} visual={visual:.2f} audio={audio:.2f} "
                    f"fused={result.emergency_confidence:.2f} "
                    f"state={result.state.value:10s} command={result.command:10s} side={side}"
                )
                logger.write(
                    time.time(),
                    scenario if visual >= audio else "none",
                    visual,
                    "siren" if audio > 0.5 else "ambient",
                    audio,
                    result.emergency_confidence,
                    result.state.value,
                    result.command,
                    side,
                )
                time.sleep(frame_period)
    finally:
        logger.close()

    print("\nSaved logs/demo_simulation_log.csv")


if __name__ == "__main__":
    main()
