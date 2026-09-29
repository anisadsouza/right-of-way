# Emergency Vehicle Recognition

A multimodal emergency vehicle recognition and autonomous yield rover that combines camera vision, siren audio sensors to detect approaching emergency vehicles and perform safe slow-down, pull-over, stop, and resume maneuvers.

## Project Overview

The system combines:

- Raspberry Pi camera-based emergency vehicle detection
- CNN-based vision classification
- Siren detection using MFCC features
- Audio and vision confidence fusion
- ESP32-based rover control
- Automatic slow, pull-over, stop, and resume behavior

## Hardware

- Raspberry Pi 4 Model B 2GB
- Raspberry Pi 5MP Camera Module
- INMP441 I2S MEMS Microphone
- ESP32-WROVER Development Board
- L298N Dual Motor Driver
- 4WD Smart Robot Car Chassis with DC Gear Motors
- 2 × 3.7V 1200mAh 18650 Batteries
- Raspberry Pi 15W Power Supply
- Active Buzzer Module
- 5V LED Traffic Light Module
- Solderless Breadboard
- Jumper Wires

## Project Structure

The complete implementation is available inside the [`ev_recognition`](./ev_recognition) directory.

## Status

Currently under development and hardware integration/testing is in progress.
