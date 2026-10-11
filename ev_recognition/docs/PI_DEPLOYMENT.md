# Raspberry Pi Deployment Guide

This guide states exactly which project files run on the Raspberry Pi 4 and which program runs on the ESP32.

## 1. Copy to the Raspberry Pi

Copy the complete `ev_recognition/` project folder to the Pi. The generated models are ignored by Git, so a Git clone alone is not enough. The Pi must receive these local files too:

```text
models/vision_ev_mobilenet.tflite
models/vision_labels.txt
models/audio_siren_classifier.joblib
models/audio_labels.txt
config_esp32.yaml
firmware/esp32_yield_controller/esp32_yield_controller.ino
```

The Python source that runs on the Pi is already in `src/evr/`; do not copy individual Python files separately. Copy the project folder as one unit.

## 2. Install the Pi Runtime

On Raspberry Pi OS 64-bit, run from the copied `ev_recognition/` directory:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 portaudio19-dev libsndfile1
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If the Pi uses a different supported Python version, create the virtual environment with that version. The audio model was saved using scikit-learn 1.9.0, which is pinned in `requirements.txt`.

## 3. Configure Sensors and Serial

1. Enable and test the Pi Camera: `rpicam-hello`.
2. Configure the INMP441 as an ALSA I2S capture device for the installed Raspberry Pi OS image.
3. List microphone devices:

```bash
source .venv/bin/activate
python -m evr.realtime --list-audio-devices
```

4. Edit `config_esp32.yaml` on the Pi:
   - Set `audio_device` to the shown INMP441 device number or name.
   - Set `serial.port` to the ESP32 USB port, normally `/dev/ttyUSB0` or `/dev/ttyACM0`.
   - Leave `controller.mode: serial` and `serial.enabled: true` for the rover.

The Pi Camera, INMP441, and ESP32 wiring are listed in [WIRING.md](WIRING.md).

## 4. Flash the ESP32

Open `firmware/esp32_yield_controller/esp32_yield_controller.ino` in Arduino IDE on a computer, choose the installed ESP32 board and port, then upload it. The ESP32 program starts in `STOP` mode for safety.

The ESP32 does not run TensorFlow, the audio classifier, or Python. Its only job is motor, LED, and buzzer control after receiving Pi commands.

## 5. Test in Safe Order

Keep the motor battery disconnected first. Connect the ESP32 to the Pi with a USB data cable.

```bash
source .venv/bin/activate
python -m evr.test_controller --config config_esp32.yaml --mode serial
```

Check the traffic lights and buzzer. Next raise the rover wheels off the ground, connect motor power, and repeat the controller test. Confirm `NORMAL`, `SLOW`, `PULL_RIGHT`, and `STOP` are correct before placing the rover on the floor.

## 6. Run the Live System

```bash
source .venv/bin/activate
python -m evr.realtime --config config_esp32.yaml
```

This command runs `src/evr/realtime.py` as the `evr.realtime` module. It loads the TFLite vision model and audio joblib model from `models/`, reads the live camera and microphone, logs to `logs/realtime_log.csv`, and sends commands to the ESP32.

## 7. Pi File Map

| File | Purpose on Pi |
|---|---|
| `src/evr/realtime.py` | Main live loop and camera/microphone input |
| `src/evr/vision.py` | TFLite image classification and approach-side estimate |
| `src/evr/audio.py` | MFCC extraction and siren prediction |
| `src/evr/fusion.py` | Confidence fusion and yield state machine |
| `src/evr/controller.py` | USB serial commands to ESP32 and CSV logging |
| `config_esp32.yaml` | Camera, microphone, fusion, serial, and controller settings |
| `models/*` | Trained models and class labels |

Do not change the provisional `emergency_threshold: 0.80` until microphone and rover tests are recorded. It is a starting setting, not a final safety value.
