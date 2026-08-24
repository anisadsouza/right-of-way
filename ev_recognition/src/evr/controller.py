from __future__ import annotations

from pathlib import Path

try:
    import serial
except ModuleNotFoundError:
    serial = None

try:
    from gpiozero import Buzzer, LED
except (ImportError, RuntimeError):
    Buzzer = None
    LED = None


class YieldController:
    def send(self, command: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PrintController(YieldController):
    def send(self, command: str) -> None:
        print(f"[controller] {command}")


class SerialController(YieldController):
    def __init__(self, port: str, baudrate: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for ESP32 serial control. Run: pip install pyserial")
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=0.25)
        self.last_command = ""

    def send(self, command: str) -> None:
        if command == self.last_command:
            return
        self.serial.write(f"{command}\n".encode("ascii"))
        self.last_command = command

    def close(self) -> None:
        self.serial.close()


class GpioTrafficLightController(YieldController):
    def __init__(self, green_pin: int, yellow_pin: int, red_pin: int, buzzer_pin: int | None) -> None:
        if LED is None:
            raise RuntimeError("gpiozero is required for GPIO output. Run on Raspberry Pi or install gpiozero.")
        self.green = LED(green_pin)
        self.yellow = LED(yellow_pin)
        self.red = LED(red_pin)
        self.buzzer = Buzzer(buzzer_pin) if buzzer_pin is not None and Buzzer is not None else None
        self.last_command = ""
        self.send("NORMAL")

    def send(self, command: str) -> None:
        if command == self.last_command:
            return
        self.green.off()
        self.yellow.off()
        self.red.off()

        if command == "NORMAL":
            self.green.on()
            self._buzz(False)
        elif command == "SLOW":
            self.yellow.on()
            self._buzz(True)
        elif command == "PULL_RIGHT":
            self.yellow.on()
            self.red.on()
            self._buzz(True)
        elif command == "STOP":
            self.red.on()
            self._buzz(True)

        self.last_command = command

    def close(self) -> None:
        self.green.off()
        self.yellow.off()
        self.red.off()
        self._buzz(False)

    def _buzz(self, enabled: bool) -> None:
        if self.buzzer is None:
            return
        if enabled:
            self.buzzer.on()
        else:
            self.buzzer.off()


def make_controller(
    mode: str,
    port: str,
    baudrate: int,
    green_pin: int = 17,
    yellow_pin: int = 27,
    red_pin: int = 22,
    buzzer_pin: int | None = 23,
) -> YieldController:
    normalized = mode.lower()
    if normalized == "serial":
        return SerialController(port, baudrate)
    if normalized == "gpio":
        return GpioTrafficLightController(green_pin, yellow_pin, red_pin, buzzer_pin)
    if normalized == "print":
        return PrintController()
    raise ValueError(f"Unknown controller mode: {mode}. Use print, serial, or gpio.")


class CsvLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.file.write("timestamp,vision_label,vision_score,audio_label,audio_score,fused,state,command,side\n")

    def write(
        self,
        timestamp: float,
        vision_label: str,
        vision_score: float,
        audio_label: str,
        audio_score: float,
        fused: float,
        state: str,
        command: str,
        side: str,
    ) -> None:
        self.file.write(
            f"{timestamp:.3f},{vision_label},{vision_score:.4f},{audio_label},{audio_score:.4f},"
            f"{fused:.4f},{state},{command},{side}\n"
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()
