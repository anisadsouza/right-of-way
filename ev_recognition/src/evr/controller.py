from __future__ import annotations

import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

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
        # Opening the port can reset the ESP32. Wait for it to boot, or the first command is lost.
        time.sleep(2.0)
        self.serial.reset_input_buffer()
        self.last_command = ""

    def send(self, command: str) -> None:
        if command == self.last_command:
            return
        self.serial.write(f"{command}\n".encode("ascii"))
        self.last_command = command

    def close(self) -> None:
        # Never leave the rover driving when the program ends or is stopped.
        try:
            self.serial.write(b"STOP\n")
            self.serial.flush()
        except Exception:
            pass
        finally:
            self.serial.close()


class WifiController(YieldController):
    """Send Pi recognition commands to the ESP32 over Wi-Fi."""

    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 80,
        timeout: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.last_command = ""

    def send(self, command: str) -> None:
        if command not in {"NORMAL", "SLOW", "PULL_RIGHT", "STOP"}:
            raise ValueError(f"Invalid ESP32 command: {command}")

        params = urlencode({"command": command})
        url = f"http://{self.host}:{self.port}/pi_cmd?{params}"

        try:
            with urlopen(url, timeout=self.timeout) as response:
                response.read()

            if command != self.last_command:
                print(f"[wifi-controller] ESP32 received: {command}")

            self.last_command = command

        except (URLError, TimeoutError, OSError) as exc:
            print(f"[wifi-controller] Could not reach ESP32: {exc}")

    def close(self) -> None:
        # Request a safe stop when the Pi program exits.
        self.send("STOP")


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
    if normalized == "wifi":
        return WifiController()
    if normalized == "gpio":
        return GpioTrafficLightController(green_pin, yellow_pin, red_pin, buzzer_pin)
    if normalized == "print":
        return PrintController()
    raise ValueError(
        f"Unknown controller mode: {mode}. Use print, serial, wifi, or gpio."
    )


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
        