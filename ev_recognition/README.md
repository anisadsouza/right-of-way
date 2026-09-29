# Project 27: Emergency Vehicle Recognition and Yield Rover

This project detects ambulance, fire-truck, and police vehicles from a Raspberry Pi Camera, detects sirens with an INMP441 I2S microphone, combines both signals, and commands an ESP32 to slow, pull the 4WD rover right, stop, then resume after the emergency vehicle has passed.

The code is prepared for the hardware you bought. No steering servo, encoders, database, or additional motor hardware is required for the planned demo.

## What is implemented

| Area | Implementation |
|---|---|
| Vision AI | MobileNetV2 transfer learning, TensorFlow Lite export, four classes: `ambulance`, `fire_truck`, `police`, `normal` |
| Audio AI | MFCC features plus Random Forest classifier, classes: `siren`, `ambient` |
| Fusion | Weighted visual/audio confidence, high-confidence single-sensor fallback, multi-frame confirmation |
| Behaviour | `NORMAL -> SLOW -> PULL_RIGHT -> STOP -> NORMAL` finite-state machine |
| Pi runtime | Pi Camera Module through Picamera2 with OpenCV fallback, I2S/ALSA microphone input, serial output and CSV logging |
| ESP32 rover | USB serial commands, 4WD L298N differential-drive control, traffic LEDs and buzzer |
| Evaluation | Dataset validator, training metric JSON reports, CSV logging, 20-trial scenario template |
| Safety | ESP32 motor firmware starts in `STOP`; the Pi has separate official power and motors use their own battery pack |

## Honest project status

The full software structure and hardware code are complete and checked syntactically. The workspace currently contains no uploaded image or audio samples beyond placeholder README files, so final trained model binaries and real accuracy numbers cannot yet exist. This is intentional: model results must be produced from your actual dataset, not invented. Once the data folders are populated, the commands below produce the requested `.tflite`, `.joblib`, labels, and metric reports.

## Project layout

```text
data/images/{ambulance,fire_truck,police,normal}/   image dataset (ignored by Git)
data/audio/{siren,ambient}/                         audio dataset (ignored by Git)
models/                                             generated models; binary files ignored by Git
reports/                                            generated validation/training metrics
src/evr/                                            Python application
firmware/esp32_yield_controller/                    ESP32 code for L298N 4WD rover
docs/WIRING.md                                      exact pin, power, and safety guide
docs/scenario_results_template.csv                  20-run real-demo evaluation sheet
```

## Dataset preparation

Place extracted datasets directly in the canonical folders above. The expected minimum for a first training run is 20 images in each image class and 10 recordings in each audio class; aim for 100-200 images per visual class and 30+ independent recordings per audio class for a credible result.

If a downloaded dataset has different folder names, copy it into the project with the importer:

```bash
python -m evr.prepare_dataset /path/to/extracted/image-dataset --kind images
python -m evr.prepare_dataset /path/to/extracted/audio-dataset --kind audio
python -m evr.validate_dataset
```

For the uploaded Roboflow Indian emergency-vehicle COCO dataset, use the dedicated converter instead:

```bash
python -m evr.prepare_roboflow_vision --source "raw_datasets/Indian emergency vehicles"
python -m evr.prepare_dataset raw_datasets/sounds --kind audio
python -m evr.validate_dataset
```

The importer recognises common names such as `fire truck`, `firetruck`, `car`, `traffic`, and `noise`. It copies files and never deletes the source dataset. Read `data/import_manifest.json` after each import to ensure its class mapping is correct.

Do not mix frames from the same video between image classes. For audio, record/download separate clips for each class; the audio trainer splits by original recording, preventing the same clip from appearing in both train and test sets.

## Development setup and training

Use Python 3.10-3.12 for training. TensorFlow does not reliably support every Python 3.13 build, so create the training environment on a supported Python version or use a Raspberry Pi OS Python version supported by your TensorFlow/TFLite package.

```bash
cd ev_recognition
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m evr.validate_dataset
python -m evr.train_vision --data data/images --epochs 15
python -m evr.train_audio --data data/audio
```

Training output:

```text
models/vision_ev_mobilenet.keras
models/vision_ev_mobilenet.tflite
models/vision_labels.txt
models/audio_siren_classifier.joblib
models/audio_labels.txt
reports/vision_training_metrics.json
reports/audio_training_metrics.json
```

The vision model uses a 160x160 MobileNetV2 with width `0.35`, selected to be practical for a Pi 4 with 2 GB RAM. `vision_input_size` in both configuration files already matches it.

The default vision command uses official ImageNet pretrained weights. When a development computer is offline, use `--weights none` to train an offline model from scratch; retain the resulting validation score in the report instead of claiming transfer-learning performance.

## Raspberry Pi setup

Install Raspberry Pi OS 64-bit, enable the camera, and test it before starting this project:

```bash
rpicam-hello
sudo apt install python3-picamera2 portaudio19-dev
```

Configure the INMP441 as an I2S capture device in Raspberry Pi OS, then find its device index:

```bash
python -m evr.realtime --list-audio-devices
```

Put the shown device number or name in `audio_device` in `config_esp32.yaml`. The runtime uses `camera_backend: auto`, which selects Picamera2 when available and falls back to OpenCV for a USB webcam.

## ESP32 and rover

Open `firmware/esp32_yield_controller/esp32_yield_controller.ino` in Arduino IDE and upload it to the ESP32. Connect its USB data cable to the Pi. Use `config_esp32.yaml`; change `/dev/ttyUSB0` to `/dev/ttyACM0` if that is where the ESP32 appears.

Test outputs with the wheels raised first:

```bash
python -m evr.test_controller --config config_esp32.yaml --mode serial --command STOP
python -m evr.test_controller --config config_esp32.yaml --mode serial
```

Read [the wiring guide](docs/WIRING.md) before connecting power. The L298N controls the left and right motor pairs separately; `PULL_RIGHT` is a controlled right arc, not servo steering.

## Run the live demo

```bash
python -m evr.realtime --config config_esp32.yaml
```

For a safe laptop demo without ESP32 hardware, use the default `config.yaml` print controller. For a repeatable clip demo:

```bash
python -m evr.realtime --config config.yaml --video demo/ambulance.mp4 --audio demo/siren.wav
```

Press `q` to stop. Each run writes `logs/realtime_log.csv`; summarize it with:

```bash
python -m evr.evaluate_logs logs/realtime_log.csv
```

Before data and hardware are available, the fusion state machine can still be demonstrated honestly:

```bash
python -m evr.demo_simulation --config config.yaml --speed 4
```

## Evaluation and PPT/report evidence

Run the 20 tests in `docs/scenario_results_template.csv`, record detection/yield/resume outcomes, and calculate the final success rate from those real entries. Do not claim `93%` accuracy or `18/20` yielding until the training reports and test matrix actually show it.

```bash
python -m evr.evaluate_trials docs/scenario_results_template.csv
```

For your PPT include: project objective and road-safety importance; system block diagram (camera and mic to Pi, Pi fusion to ESP32, ESP32 to L298N/LED/buzzer); dataset class counts and source licences; vision and audio model diagrams; fusion formula; yield state diagram; wiring photo; training accuracy/confusion matrix from `reports`; and the 20-trial results table. Explain false negatives as the higher safety risk and show how multi-frame confirmation reduces false alarms.

## Git and .gitignore

`.gitignore` is updated. It keeps code, configurations, wiring documentation, label text files, and the 20-test template in Git, but excludes raw datasets, large model binaries, generated logs, local virtual environments, and machine-specific settings. This is correct even before a GitHub remote is connected. When ready, add a remote and commit the source/documentation; do not commit downloaded data or models.
