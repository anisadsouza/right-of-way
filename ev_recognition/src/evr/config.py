from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass(frozen=True)
class FusionConfig:
    visual_weight: float
    audio_weight: float
    emergency_threshold: float
    clear_threshold: float
    trigger_frames: int
    clear_seconds: float


@dataclass(frozen=True)
class SerialConfig:
    enabled: bool
    port: str
    baudrate: int


@dataclass(frozen=True)
class ControllerConfig:
    mode: str
    green_pin: int
    yellow_pin: int
    red_pin: int
    buzzer_pin: int | None


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    vision_model: Path
    vision_labels: Path
    audio_model: Path
    audio_labels: Path
    camera_index: int
    frame_width: int
    frame_height: int
    vision_input_size: int
    vision_every_n_frames: int
    audio_sample_rate: int
    audio_window_seconds: float
    audio_hop_seconds: float
    audio_mfcc_count: int
    fusion: FusionConfig
    serial: SerialConfig
    controller: ControllerConfig
    show_window: bool
    log_csv: Path


def _path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    base_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any]
        if yaml is None:
            raw = _load_simple_yaml(f.read())
        else:
            raw = yaml.safe_load(f)

    fusion = raw["fusion"]
    serial = raw["serial"]
    controller = raw.get("controller", {})
    gpio_pins = controller.get("gpio_pins", {})
    demo = raw.get("demo", {})
    controller_mode = str(controller.get("mode", "serial" if bool(serial["enabled"]) else "print"))

    return AppConfig(
        base_dir=base_dir,
        vision_model=_path(base_dir, raw["vision_model"]),
        vision_labels=_path(base_dir, raw["vision_labels"]),
        audio_model=_path(base_dir, raw["audio_model"]),
        audio_labels=_path(base_dir, raw["audio_labels"]),
        camera_index=int(raw["camera_index"]),
        frame_width=int(raw["frame_width"]),
        frame_height=int(raw["frame_height"]),
        vision_input_size=int(raw["vision_input_size"]),
        vision_every_n_frames=int(raw["vision_every_n_frames"]),
        audio_sample_rate=int(raw["audio_sample_rate"]),
        audio_window_seconds=float(raw["audio_window_seconds"]),
        audio_hop_seconds=float(raw["audio_hop_seconds"]),
        audio_mfcc_count=int(raw["audio_mfcc_count"]),
        fusion=FusionConfig(
            visual_weight=float(fusion["visual_weight"]),
            audio_weight=float(fusion["audio_weight"]),
            emergency_threshold=float(fusion["emergency_threshold"]),
            clear_threshold=float(fusion["clear_threshold"]),
            trigger_frames=int(fusion["trigger_frames"]),
            clear_seconds=float(fusion["clear_seconds"]),
        ),
        serial=SerialConfig(
            enabled=bool(serial["enabled"]),
            port=str(serial["port"]),
            baudrate=int(serial["baudrate"]),
        ),
        controller=ControllerConfig(
            mode=controller_mode,
            green_pin=int(gpio_pins.get("green", 17)),
            yellow_pin=int(gpio_pins.get("yellow", 27)),
            red_pin=int(gpio_pins.get("red", 22)),
            buzzer_pin=None if gpio_pins.get("buzzer") in {None, ""} else int(gpio_pins.get("buzzer", 23)),
        ),
        show_window=bool(demo.get("show_window", True)),
        log_csv=_path(base_dir, str(demo.get("log_csv", "logs/realtime_log.csv"))),
    )


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for this project's config.yaml when PyYAML is not installed."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = _split_yaml_line(line.strip())

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value is None:
            section: dict[str, Any] = {}
            parent[key] = section
            stack.append((indent, section))
        else:
            parent[key] = _coerce_value(value)

    return result


def _split_yaml_line(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        raise ValueError(f"Invalid config line: {line}")
    key, value = line.split(":", 1)
    value = value.strip()
    return key.strip(), value if value else None


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
