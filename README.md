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

## Dataset Status 

### Vision
- Sources: Roboflow "Indian Emergency Vehicles" (COCO format), Kaggle vehicle classification dataset (normal class)
- Final dataset: 26,763 unique images after deduplication (5,964 duplicates removed)
- Class distribution: normal 16,932, ambulance 6,159, police 2,835, fire_truck 837
- Model: MobileNetV2, alpha 0.35, 160x160 input
- Validation accuracy: 88.25% (target: 93%)
- Status: below target, likely due to fire_truck class imbalance; retraining with alpha 0.5 / 192x192 and additional fire_truck images planned

### Audio
- Source: Kaggle emergency vehicle siren dataset (ambulance, firetruck -> siren; traffic -> ambient)
- Final dataset: 596 unique audio files after deduplication (1,204 duplicates removed), 396 siren / 200 ambient
- Model: MFCC features (40 coefficients) + Random Forest
- Verified accuracy: 99.40% mean across 5-fold grouped cross-validation (SD 1.04%)
- Status: complete

## Project Structure

The complete implementation is available inside the [`ev_recognition`](./ev_recognition) directory.

## Status

Currently under development and hardware integration/testing is in progress.
