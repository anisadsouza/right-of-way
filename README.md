# Right-of-Way: Emergency Vehicle Recognition

A multimodal emergency vehicle recognition system for a 4WD yield rover. A camera model and a siren audio model are fused on a Raspberry Pi to detect an approaching emergency vehicle, and an ESP32 drives the rover through slow-down, pull-over, stop, and resume behaviour.

The full implementation is in [`ev_recognition`](./ev_recognition).

## Faculty Results At A Glance

| Item | Measured result | Meaning |
| --- | --- | --- |
| Vision model | **92.14%** validation accuracy on **1,667 images** | Below the 93% target; the result is reported honestly. |
| Vision safety-relevant result | Ambulance recall **88.00%** | Ambulances were missed as normal in 42 of 500 validation images. |
| Audio model | **95.01%** held-out-window accuracy on **1,103 windows** | Clean-dataset result, not a live motor-noise measurement. |
| Fusion simulation at threshold 0.80 | **0/196** non-siren clips stopped; **3-5/8** siren clips stopped | The current threshold is provisional, not a final safety setting. |
| Recorded ambulance clips | Both reached `NORMAL -> SLOW -> PULL_OVER -> STOPPED` | Confirms offline software integration and command generation. |

For the board-level deployment procedure, code map, and exact Pi command, see [Raspberry Pi Deployment](./ev_recognition/docs/PI_DEPLOYMENT.md). For power, sensor, L298N, LED, and buzzer connections, see [Wiring](./ev_recognition/docs/WIRING.md).

## Current Status

| Area | Status |
| --- | --- |
| Vision dataset sourcing, cleaning, rebalancing | Complete |
| Vision model | Trained, 92.14% validation accuracy (target 93%, not yet met) |
| Audio dataset sourcing and cleaning | Complete |
| Audio model | Trained, 95.01% held-out-window accuracy; external audio checks documented below |
| Fusion and yield state machine (code) | Written; exercised against recorded clips and ESC-50 audio simulation, see Sections 4.2 and 7 |
| ESP32 firmware (code) | Arduino `.ino` and ESP-IDF project present; not flashed or bench-tested on the rover |
| Chassis, soldering, motors | Assembled and connected |
| L298N, ESP32, LED, buzzer wiring and bench tests | Not started |
| Raspberry Pi, camera, microphone integration | Not started |
| Pi to ESP32 serial link | Not tested |
| Live pipeline on hardware | Not started |
| Raspberry Pi frame rate benchmark | Not started |
| Siren detection with motors running | Not started |
| 20-scenario acceptance test, false-yield test, resume timing | Not started |

Hardware integration and physical testing are in progress. No result in this README comes from the assembled rover.

## 1. System Overview

1. The camera frame is classified by a MobileNetV2 model into ambulance, fire_truck, normal, or police.
2. Microphone audio is split into 1 second windows (0.5 second hop), converted to 40 MFCC features, and classified as siren or ambient by a Random Forest.
3. The two confidence scores are fused (visual weight 0.6, audio weight 0.4).
4. If either cue alone is above the emergency threshold, that single score is used instead of the weighted blend. One strong cue can therefore trigger the system without confirmation from the other sensor.
5. A yield state machine moves through NORMAL, SLOW, PULL_OVER, and STOPPED, and returns to NORMAL after the confidence has stayed below the clear threshold for the configured clear time. The controller sends NORMAL, SLOW, PULL_RIGHT, or STOP to the ESP32 over serial.
6. The ESP32 drives the left and right motor pairs through the L298N, plus the LED traffic module and buzzer. It starts in STOP.

## 2. Hardware

- Raspberry Pi 4 Model B 2GB
- Raspberry Pi 5MP Camera Module
- INMP441 I2S MEMS Microphone
- ESP32-WROVER Development Board
- L298N Dual Motor Driver
- 4WD Smart Robot Car Chassis with DC Gear Motors
- 2 x 3.7V 1200mAh 18650 Batteries
- Raspberry Pi 15W Power Supply
- Active Buzzer Module
- 5V LED Traffic Light Module
- Solderless Breadboard
- Jumper Wires

The Raspberry Pi is powered from its own supply and the motors from the battery pack, kept separate. Wiring is documented in `ev_recognition/docs/WIRING.md`.

## 3. Datasets

The models are trained on public datasets, not on images or audio recorded with this project's own camera and microphone.

### 3.1 Vision

Sources: Roboflow Universe "Indian Emergency Vehicles" (COCO format) for ambulance, fire truck, police, and some normal images; a Kaggle vehicle classification dataset for the normal class. Licence terms for each source should be checked and recorded before submission.

| Stage | Count |
| --- | --- |
| Images after initial sorting | 32,727 |
| Exact duplicates removed (MD5 of file contents) | 5,964 |
| Unique images | 26,763 |

Duplicates were all within the same class; no image appeared under two classes. The training loader read 26,761 of the 26,763 files (two files in the normal class were not read as images).

Class counts after deduplication and after rebalancing:

| Class | After deduplication | After rebalancing (used for training) |
| --- | --- | --- |
| normal | 16,932 | 2,500 |
| ambulance | 6,159 | 2,500 |
| police | 2,835 | 2,500 |
| fire_truck | 837 | 837 |
| Total | 26,763 | 8,337 |

Rebalancing capped the three large classes at 2,500 images each. The excess images were moved to `data/images_excluded/` (not deleted).

Per-class stratified 80/20 split (fixed seed 42):

| Class | Train | Validation |
| --- | --- | --- |
| ambulance | 2,000 | 500 |
| fire_truck | 670 | 167 |
| normal | 2,000 | 500 |
| police | 2,000 | 500 |
| Total | 6,670 | 1,667 |

### 3.2 Audio

Source: a Kaggle emergency vehicle siren dataset. Ambulance and firetruck clips form the siren class; traffic clips form the ambient class. Licence terms should be checked and recorded.

| Stage | Count |
| --- | --- |
| Files after initial sorting | 1,800 |
| Exact duplicates removed | 1,204 |
| Unique files | 596 (396 siren, 200 ambient) |

The duplicates came from running the dataset import step more than once.

## 4. Models and Results

### 4.1 Vision

MobileNetV2, ImageNet weights, width multiplier 0.5, 192x192 input. Augmentation: horizontal flip, rotation, zoom, contrast. Dropout 0.25. Two training phases:

1. Frozen base, classifier head only (Adam, learning rate 0.0008, early stopping).
2. Fine-tuning of the upper base layers at learning rate 0.00008, early stopping with best weights restored.

Best model command:

```
python -m evr.train_vision --data data/images --epochs 15 --img-size 192 --alpha 0.5 --finetune-layers 50 --finetune-epochs 10 --report reports/vision_retrain_v6.json
```

Training history:

| Run | Setup | Validation accuracy | Note |
| --- | --- | --- | --- |
| 1 | alpha 0.35, 160x160 | 88.25% | Invalid, see below |
| 2 | alpha 0.5, 192x192 | 89.41% | Invalid, see below |
| 3 | Corrected split, rebalanced data, frozen base | 85.36% | First valid result |
| 4 | Fine-tuning, last 30 layers | 91.06% | |
| 5 | Fine-tuning, last 50 layers | 92.14% | Best, at fine-tuning epoch 9 of 10 |

Runs 1 and 2 are not valid. Their validation set contained no ambulance or fire_truck images, because the dataset loader took a contiguous slice of the class-ordered file list. This was found with a per-class evaluation, and the split was rewritten to shuffle and split each class independently. A model trained from scratch without pretrained weights reached only 27.8% and was discarded.

Final model (run 5), 1,667 validation images:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ambulance | 0.9129 | 0.8800 | 0.8961 | 500 |
| fire_truck | 0.9699 | 0.9641 | 0.9670 | 167 |
| normal | 0.8852 | 0.9560 | 0.9192 | 500 |
| police | 0.9541 | 0.9140 | 0.9336 | 500 |
| Overall accuracy | | | 0.9214 | 1,667 |

Confusion matrix (rows actual, columns predicted):

| | ambulance | fire_truck | normal | police |
| --- | --- | --- | --- | --- |
| ambulance | 440 | 2 | 42 | 16 |
| fire_truck | 0 | 161 | 2 | 4 |
| normal | 18 | 2 | 478 | 2 |
| police | 24 | 1 | 18 | 457 |

The largest remaining errors are ambulance predicted as normal (42), police predicted as ambulance (24), and ambulance predicted as police (16). A missed ambulance is the safety-relevant error. Overfitting began in the last fine-tuning epoch (training accuracy rose while validation accuracy fell), and early stopping restored the best weights. The 93% target is 0.86 points away.

### 4.2 Audio

The deployed V3 model uses 40 MFCC features from 16 kHz, one-second windows and a Random Forest classifier. Its saved held-out evaluation contains 1,103 windows: 600 ambient and 45 ambient windows were classified incorrectly, while 448 of 458 siren windows were classified correctly. Overall held-out window accuracy is **95.01%**. This is a model-window metric, not a vehicle-level safety result.

An independent ESC-50 check uses 980 non-siren windows and 40 external-siren windows. At the current `0.80` score threshold, 0.20% of non-siren windows and 27.50% of siren windows met the threshold. The fusion state-machine simulation below is more useful than window score alone because it includes multi-frame confirmation.

| Scenario | Updates/s | Non-siren clips stopped | Siren clips stopped |
| --- | ---: | ---: | ---: |
| Audio only, threshold 0.80 | 6 | 0 / 196 | 4 / 8 |
| Audio only, threshold 0.80 | 15 | 0 / 196 | 5 / 8 |
| Vision score 0.10 plus audio, threshold 0.80 | 6 | 0 / 196 | 3 / 8 |
| Vision score 0.10 plus audio, threshold 0.80 | 15 | 0 / 196 | 3 / 8 |

`0.80` is a **provisional** threshold, not a validated or final safety threshold. It avoided `STOPPED` outcomes in these simulated non-siren clips but missed several siren clips. Final tuning requires the real INMP441 microphone, Pi inference rate, and motor-noise tests.

## 5. Known Limitations

- Training and validation images come from the same public sources. The Roboflow download lists 19,959 images against 8,227 on the project page, which suggests augmented copies of source photos. Near-duplicate variants of one photo may therefore sit on both sides of the split, and exact-hash deduplication does not remove them. The 92.14% figure may be optimistic.
- There is no held-out test set captured with this project's own camera. The course guide's acceptance test (at least 80 held-out images from the project camera) has not been done.
- fire_truck is the smallest class (837 images).
- The audio set is small (596 files) and comes from a single source dataset. Clips cut from the same longer recording may not be separable by file name, so leakage between folds cannot be fully excluded.
- Audio accuracy was measured on clean audio only, not with rover motor noise.
- The single-strong-cue rule in the fusion can trigger a stop from audio alone.
- The V3 audio model was saved with scikit-learn 1.9.0 and was rechecked in an environment with 1.9.1, which emits a compatibility warning. Recreate the Pi environment with the training version or retrain/export there before final hardware testing.

## 6. Reproducing the Work

Run from `ev_recognition/` with the virtual environment active.

```
pip install -r requirements.txt
pip install -e .
```

Dataset preparation (raw downloads go in `raw_datasets/`):

```
python -m evr.prepare_dataset raw_datasets/sounds --kind audio
python -m evr.prepare_dataset raw_datasets/Vehicles --kind images
python sort_coco_dataset.py
python rebalance_dataset.py
python -m evr.validate_dataset
```

Training and evaluation:

```
python -m evr.train_vision --data data/images --epochs 15 --img-size 192 --alpha 0.5 --finetune-layers 50 --finetune-epochs 10
python -m evr.train_audio --data data/audio
python eval_vision.py
```

Realtime pipeline on recorded files (no hardware):

```
python -m evr.realtime --config config.yaml --video demo/traffic.mp4 --audio demo/siren.wav --headless
```

`config.yaml` and `config_esp32.yaml` set `vision_input_size: 192` to match the trained model.

Raw datasets, sorted images and audio, trained models, logs, reports, and demo clips are excluded by `.gitignore`, so a fresh clone must repeat the dataset steps above and retrain.

## 7. Recorded-Clip Pipeline Check

The V3 audio model was re-run through the realtime pipeline on both supplied ambulance videos in audio-only, headless print-controller mode. The runtime now paces a recorded video at its original frame rate, so that a paired audio file has time to produce its one-second inference windows. Earlier one-row logs were caused by the video finishing before the audio thread could advance.

| Run | Logged frames | Audio-score range | State sequence | Command sequence |
| --- | ---: | --- | --- | --- |
| `traffic.mp4` + `siren.wav` | 592 | 0.4588–0.9695 | NORMAL -> SLOW -> PULL_OVER -> STOPPED | NORMAL -> SLOW -> PULL_RIGHT -> STOP |
| `traffic_second.mp4` + `siren_second.wav` | 574 | 0.7315–0.9652 | NORMAL -> SLOW -> PULL_OVER -> STOPPED | NORMAL -> SLOW -> PULL_RIGHT -> STOP |

These checks confirm the current recorded-file path, audio classifier, fusion FSM, command generation, and CSV logging work together. They do not measure live accuracy, false-yield rate, recovery after a vehicle passes, or hardware behaviour. The audio input loops for the clip demonstration, so a return to `NORMAL` cannot be measured there.

## 8. Remaining Work and Acceptance Targets

Targets are taken from the course build guide.

| Item | Target | Status |
| --- | --- | --- |
| Vision accuracy | 93% or higher | 92.14%, not met |
| Vision accuracy on project-camera test images | At least 80 held-out images | Not done |
| Raspberry Pi inference speed | 6 FPS or higher | Not measured |
| Siren detection with motors running | 90% or higher | Not tested |
| Approach-side detection | Working | Not tested |
| ESP32 and Raspberry Pi serial link | Verified | Not tested |
| Yield, pull-over, stop, resume on the rover | Working | Not tested |
| 20-scenario acceptance test | 18 of 20 successful yields | Not started |
| False-yield test | At most 1 in 10 normal runs | Not started |
| Resume after the emergency vehicle passes | Within 3 seconds | Not started |

The remaining acceptance work is hardware-dependent: flash and bench-test the ESP32; configure the Pi camera and INMP441; measure Pi FPS; test the serial link, motor noise, approach direction, false yields, and recovery; then complete the 20 real scenarios. Do not replace this measured status with estimated success counts in the presentation.

## 9. Repository Layout

```
ev_recognition/
  src/evr/            Python package (vision, audio, fusion, controller, realtime, training)
  firmware/           ESP32 firmware
  docs/               Wiring guide and scenario result template
  scripts/            Demo and test scripts
  config.yaml         Pi-only configuration
  config_esp32.yaml   Pi and ESP32 configuration
  sort_coco_dataset.py, rebalance_dataset.py, eval_vision.py
```
