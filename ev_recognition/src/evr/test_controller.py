from __future__ import annotations

import argparse
import time
from pathlib import Path

from evr.config import load_config
from evr.controller import make_controller


COMMANDS = ["NORMAL", "SLOW", "PULL_RIGHT", "STOP", "NORMAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test print, ESP32 serial, or Raspberry Pi GPIO controller.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=["print", "serial", "gpio"], help="Override config controller mode.")
    parser.add_argument("--command", choices=COMMANDS, help="Send one command only.")
    parser.add_argument("--delay", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = args.mode or config.controller.mode
    controller = make_controller(
        mode,
        config.serial.port,
        config.serial.baudrate,
        config.controller.green_pin,
        config.controller.yellow_pin,
        config.controller.red_pin,
        config.controller.buzzer_pin,
    )

    try:
        commands = [args.command] if args.command else COMMANDS
        for command in commands:
            print(f"Sending {command}")
            controller.send(command)
            time.sleep(args.delay)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
