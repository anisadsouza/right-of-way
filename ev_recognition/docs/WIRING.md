# Wiring and First-Power Guide

This wiring matches the hardware you have: Raspberry Pi 4 (2 GB), Pi 5 MP Camera,
INMP441 I2S microphone, ESP32 DevKitC/WROVER, L298N, 4WD chassis, buzzer, and a
traffic-light module. The Pi performs AI; the ESP32 controls the rover and indicators.

## Important power rules

- Power the Raspberry Pi only with its official 15 W supply.
- Power the L298N motor input from the two 18650 cells in series (about 7.4 V nominal).
- Do not power the Pi from the L298N 5 V pin and do not run motors from the Pi.
- Join `ESP32 GND` to `L298N GND`. The USB cable between Pi and ESP32 provides their shared reference.
- Switch off motor power while uploading firmware. Raise the chassis before the first motor test.

## Raspberry Pi sensors

| Part | Raspberry Pi connection |
|---|---|
| Pi Camera | Insert the ribbon in the Pi camera connector. Test it with `rpicam-hello`. |
| INMP441 VDD | 3.3 V, physical pin 1 or 17 |
| INMP441 GND | GND, physical pin 6 |
| INMP441 SCK | GPIO 18, physical pin 12 |
| INMP441 WS | GPIO 19, physical pin 35 |
| INMP441 SD | GPIO 20, physical pin 38 |
| INMP441 L/R | GND for the left channel |
| Pi to ESP32 | USB data cable; this is serial control and does not need TX/RX jumper wires |

Enable I2S/audio on Raspberry Pi OS before testing the microphone. Its input name will appear in
`python -m evr.realtime --list-audio-devices`; put that device number or name in `audio_device` in `config_esp32.yaml`.

## ESP32 to L298N and indicators

Remove the `ENA` and `ENB` jumpers on the L298N because the ESP32 sends PWM speed control.
Connect both left motors in parallel to `OUT1/OUT2`, and both right motors in parallel to `OUT3/OUT4`.
If a side moves backward, swap that side's two motor wires.

| ESP32 GPIO | Connect to |
|---|---|
| GPIO 25 | L298N ENA |
| GPIO 26 | L298N IN1 |
| GPIO 27 | L298N IN2 |
| GPIO 14 | L298N ENB |
| GPIO 18 | L298N IN3 |
| GPIO 19 | L298N IN4 |
| GPIO 32 | Traffic module green input |
| GPIO 33 | Traffic module yellow input |
| GPIO 23 | Traffic module red input |
| GPIO 4 | Active buzzer signal pin |
| GND | L298N GND, indicator ground, buzzer ground |

Use the traffic module's `VCC` and `GND` as specified by the module. If it is a 5 V module that does not recognise 3.3 V as HIGH, power it from 3.3 V if supported or use transistor/level-shifter inputs; do not feed 5 V back into an ESP32 GPIO.

## Firmware and staged test

1. Upload `firmware/esp32_yield_controller/esp32_yield_controller.ino` with Arduino IDE, board set to your ESP32.
2. Leave the motor battery disconnected and run `python -m evr.test_controller --config config_esp32.yaml --mode serial`.
3. Check the traffic light and buzzer.
4. Raise the rover, connect the motor battery, then send `NORMAL`, `SLOW`, `PULL_RIGHT`, and `STOP` one at a time.
5. Only then place it on the floor at low speed and tune motor speeds in the firmware if needed.

The ESP32 firmware starts in `STOP` mode. It should remain connected to the Pi via USB during the demo.
