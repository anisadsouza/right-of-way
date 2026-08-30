# Wiring Guide

This guide matches the reduced hardware list:

- Raspberry Pi 4
- Pi Camera or USB webcam
- USB microphone
- ESP32-DevKitC-32E
- LED traffic light module or red/yellow/green LEDs
- Optional buzzer

## System Connection

```text
Camera/Webcam -> Raspberry Pi
USB Mic       -> Raspberry Pi
Raspberry Pi -> USB cable -> ESP32
ESP32 GPIO    -> LED traffic light + buzzer
```

## Camera and Mic

USB webcam:

```text
Webcam USB -> Raspberry Pi USB port
```

Pi Camera:

```text
Pi Camera ribbon cable -> Raspberry Pi camera connector
```

USB mic:

```text
USB mic -> Raspberry Pi USB port
```

## ESP32 Pins

The simple ESP32 traffic-light firmware uses these pins:

| ESP32 Pin | Connects To |
|---|---|
| GPIO 25 | Green LED signal |
| GPIO 26 | Yellow LED signal |
| GPIO 27 | Red LED signal |
| GPIO 14 | Optional buzzer positive |
| GND | LED/buzzer ground |

If you use separate LEDs, put a 220 ohm resistor in series with each LED.

## ESP32 Firmware

Upload this sketch:

```text
firmware/esp32_traffic_light/esp32_traffic_light.ino
```

It accepts these commands from the Raspberry Pi:

```text
NORMAL
SLOW
PULL_RIGHT
STOP
```

## Raspberry Pi Commands

Use `config_esp32.yaml` when the ESP32 is connected:

```bash
python -m evr.test_controller --config config_esp32.yaml --mode serial
python -m evr.realtime --config config_esp32.yaml
```

If serial fails, check the ESP32 device name:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

If it appears as `/dev/ttyACM0`, edit `config_esp32.yaml` and change the serial port.
