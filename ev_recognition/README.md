# Project 27: Emergency Vehicle Recognition and Yield Rover

This is the complete software workspace for a 4WD rover that detects emergency vehicles, yields, and then resumes after the vehicle has passed. The Raspberry Pi runs the camera, microphone, AI models, fusion logic, logging, and USB serial communication. The ESP32 controls the L298N motor driver, traffic-light module, and buzzer.

The detailed report, datasets, measured results, limitations, and presentation-ready evidence are in the [workspace README](../README.md). This file is the practical guide for running the project.

## Faculty Summary

**Problem:** Give right of way to an approaching ambulance, fire truck, or police vehicle by detecting visual and siren evidence, then commanding a rover to slow, pull right, and stop.

**Method:** A MobileNetV2 vision classifier and MFCC plus Random Forest audio classifier run on the Raspberry Pi. Their confidence scores are fused and passed through a multi-frame yield state machine. The Pi sends newline-delimited commands to the ESP32 over USB serial; the ESP32 drives the L298N motor driver, LED traffic module, and buzzer.

| Evidence item | Measured result | Honest interpretation |
| --- | --- | --- |
| Vision model | **92.14%** validation accuracy on **1,667 images** | Below the 93% project target; it must be reported as 92.14%, not rounded up. |
| Vision class performance | Ambulance recall **88.00%**, fire-truck recall **96.41%**, normal recall **95.60%**, police recall **91.40%** | Ambulance-to-normal mistakes remain the most important vision risk. |
| Audio model | **95.01%** held-out-window accuracy on **1,103 windows** | Clean-dataset result, not a motor-noise or live-road result. |
| ESC-50 fusion simulation at threshold 0.80 | **0/196** non-siren clips stopped; **3-5/8** siren clips stopped depending on update rate/vision condition | `0.80` is provisional: it limits simulated false stops but misses sirens. |
| Recorded ambulance demo | Both supplied clips reached `NORMAL -> SLOW -> PULL_OVER -> STOPPED` | Software integration check only; not a live-hardware validation. |

The final hardware results still required for assessment are Pi FPS, INMP441 performance, motor-noise robustness, approach side, ESP32 serial control, false-yield rate, recovery time, and the 20-scenario test.

## Completed Software

| Area | What exists |
| --- | --- |
| Vision | MobileNetV2 transfer-learning pipeline, four classes, TFLite export, labels, validation/evaluation tools |
| Audio | MFCC plus Random Forest siren classifier, grouped recording split, model labels, ESC-50 tests and threshold sweep |
| Fusion | Weighted confidence plus high-confidence single-cue fallback and `NORMAL -> SLOW -> PULL_OVER -> STOPPED -> NORMAL` FSM |
| Runtime | Pi Camera/Picamera2 or OpenCV input, INMP441/ALSA audio input, recorded-file demo mode, CSV logs, serial/GPIO/print controllers |
| Rover firmware | Arduino ESP32 program for the supplied 4WD L298N rover, active buzzer, and three-light module; native ESP-IDF version also present |
| Evaluation | Dataset validation, vision evaluation, audio threshold sweep, fusion simulation, log analysis, and a 20-scenario worksheet |
| Documentation | Wiring, safe power rules, Pi/ESP32 configuration, and this run guide |

## Current Measured Results

- Vision validation accuracy: **92.14%** on 1,667 images. The stated 93% target is not met.
- Audio held-out window accuracy: **95.01%** on 1,103 windows.
- ESC-50 fusion simulation at the provisional `0.80` threshold: zero of 196 non-siren clips reached `STOPPED`, but only 3-5 of 8 siren clips reached `STOPPED`, depending on update rate and simulated vision condition.
- The current V3 audio model was replayed through both ambulance videos. Each reached `NORMAL -> SLOW -> PULL_OVER -> STOPPED` and produced the expected `NORMAL`, `SLOW`, `PULL_RIGHT`, `STOP` command sequence.

These are offline results. Pi speed, microphone quality, motor noise, direction accuracy, serial control, and the 20-scenario test remain hardware validation work.

## Hardware This Supports

- Raspberry Pi 4 Model B 2 GB with official 15 W supply and Pi 5 MP camera
- INMP441 I2S microphone
- ESP32-DevKitC or ESP32-WROVER
- L298N with supplied 4WD chassis and two parallel motor pairs
- Two 18650 cells for motors, active buzzer, 5 V traffic-light module, breadboard, and jumper wires

No steering servo, encoders, database, or extra motor hardware is required. Read [WIRING.md](docs/WIRING.md) before applying power.

## Install and Train

Run from `ev_recognition/`. Use Python 3.10-3.12 for training where possible.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m evr.validate_dataset
python -m evr.train_vision --data data/images --epochs 15 --img-size 192 --alpha 0.5 --finetune-layers 50 --finetune-epochs 10
python -m evr.train_audio --data data/audio
```

The workspace already has model binaries locally. They, datasets, logs, and generated reports are intentionally ignored by Git; a fresh clone needs the source data and training commands above.

## Dataset Layout

```text
data/images/{ambulance,fire_truck,police,normal}/
data/audio/{siren,ambient}/
models/
reports/
```

Prepare downloaded data with:

```bash
python -m evr.prepare_dataset /path/to/images --kind images
python -m evr.prepare_dataset /path/to/audio --kind audio
python -m evr.validate_dataset
```

For the imported Roboflow COCO emergency-vehicle set, use `python -m evr.prepare_roboflow_vision --source "raw_datasets/Indian emergency vehicles"`.

## Test Without Hardware

```bash
python -m evr.sweep_audio_threshold --test-dir demo/esc50_test
python -m evr.simulate_fusion_audio --test-dir demo/esc50_test
python -m evr.realtime --config config.yaml --video demo/traffic.mp4 --audio demo/siren.wav --no-vision --headless --log-csv logs/demo_audio_only.csv
python -m evr.evaluate_logs logs/demo_audio_only.csv
```

Recorded videos run at their saved frame rate by default so paired audio remains synchronised with the runtime clock. Add `--fast-video` only for a quick vision batch check. The default `config.yaml` uses the safe print controller and does not move hardware.

## Run on the Rover

The deployment guide, including exactly what goes on the Pi and ESP32, is [PI_DEPLOYMENT.md](docs/PI_DEPLOYMENT.md). In short:

1. Copy the whole `ev_recognition/` folder to the Pi, including the locally generated `models/` files.
2. Upload `firmware/esp32_yield_controller/esp32_yield_controller.ino` to the ESP32.
3. Follow [WIRING.md](docs/WIRING.md), test LEDs/buzzer with motors disconnected, then test with wheels raised.
4. On the Pi, configure the INMP441 as an I2S capture device and set the microphone device plus ESP32 serial port in `config_esp32.yaml`.
5. Run `python -m evr.test_controller --config config_esp32.yaml --mode serial`, then `python -m evr.realtime --config config_esp32.yaml`.

The ESP32 firmware starts in `STOP`. Keep the rover raised during first motor tests.

## Git Policy

`.gitignore` is ready: it excludes raw datasets, model binaries, reports, logs, local environments, and generated ESP-IDF output while retaining code, configuration, labels, firmware, wiring, and documentation. Do not commit model binaries or downloaded datasets. The current V3 audio model backup is also ignored.
