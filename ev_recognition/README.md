# Project 27: Emergency Vehicle Recognition

This is a complete starter software stack for the ambulance/fire/police recognition project:

- Vision classifier: MobileNetV2 transfer learning, exported to TensorFlow Lite for Raspberry Pi.
- Audio classifier: MFCC features + Random Forest siren detector.
- Fusion: weighted visual/audio confidence with multi-frame confirmation.
- Response: finite state machine sends `NORMAL`, `SLOW`, `PULL_RIGHT`, and `STOP` to an ESP32.
- Demo support: live camera/mic or pre-recorded video/audio files.

## Folder Format

Put images into class folders:

```text
data/images/
  ambulance/
  fire_truck/
  police/
  normal/
```

Put audio into class folders:

```text
data/audio/
  siren/
  ambient/
```

You can split siren into `ambulance_siren`, `police_siren`, etc. if your dataset supports it.

## Setup

```bash
cd ev_recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Raspberry Pi, prefer installing `tensorflow`/`tflite-runtime` according to your OS image if regular TensorFlow is too heavy.

## Train Vision Model

```bash
python -m evr.train_vision --data data/images --epochs 12
```

Output:

- `models/vision_ev_mobilenet.keras`
- `models/vision_ev_mobilenet.tflite`
- `models/vision_labels.txt`

For the report, include training accuracy, validation accuracy, confusion matrix screenshots if you add them, and examples of wrong predictions.

You can collect your own camera samples:

```bash
python -m evr.collect_images ambulance --count 80
python -m evr.collect_images normal --count 80
```

## Train Audio Model

```bash
python -m evr.train_audio --data data/audio
```

Output:

- `models/audio_siren_classifier.joblib`
- `models/audio_labels.txt`

You can record your own short audio clips:

```bash
python -m evr.record_audio siren --seconds 5
python -m evr.record_audio ambient --seconds 5
```

## Run Live Detection

Camera + microphone:

```bash
python -m evr.realtime --config config.yaml
```

Video/audio demo files:

```bash
python -m evr.realtime --config config.yaml --video demo/traffic.mp4 --audio demo/siren.wav
```

Vision only:

```bash
python -m evr.realtime --config config.yaml --no-audio
```

Audio only:

```bash
python -m evr.realtime --config config.yaml --no-vision
```

Press `q` to stop the display window.

## Run Demo Without Dataset or Hardware

This command simulates normal traffic, siren-only detection, visual-only detection, combined detection, and return-to-normal. Use it to show the core project logic before the trained models and Raspberry Pi hardware are ready.

```bash
python -m evr.demo_simulation --config config.yaml --speed 4
```

It writes:

```text
logs/demo_simulation_log.csv
```

Then summarize:

```bash
python -m evr.evaluate_logs logs/demo_simulation_log.csv
```

## ESP32 Connection

For your selected hardware, use the simple traffic-light firmware:

```text
firmware/esp32_traffic_light/esp32_traffic_light.ino
```

1. Open that file in Arduino IDE.
2. Select your ESP32 board.
3. Upload the sketch.
4. Connect ESP32 USB to Raspberry Pi.
5. Use `config_esp32.yaml` when running the Python code.

```yaml
serial:
  enabled: true
  port: /dev/ttyUSB0
  baudrate: 115200
```

If your ESP32 appears as `/dev/ttyACM0`, update the port.

Test the controller before running AI:

```bash
python -m evr.test_controller --config config.yaml --mode print
python -m evr.test_controller --config config_esp32.yaml --mode serial
```

If you skip ESP32 and connect LEDs directly to Raspberry Pi GPIO later, set `controller.mode` to `gpio`.

Detailed wiring is in `docs/WIRING.md`.

## Fusion Logic

The live loop starts with:

```text
fused_confidence = 0.6 * visual_emergency_score + 0.4 * siren_score
```

If either single sensor is very confident, that score can override the weighted score. This supports visual-only and audio-only scenarios while still making combined detection the strongest case. The system only triggers after several confident frames. It clears after the emergency signal is gone for 3 seconds. Tune these in `config.yaml`.

## Demo Day Script

1. Start with normal traffic clip: state should remain `NORMAL`.
2. Play siren audio only: state should move to `SLOW`, then `PULL_RIGHT`/`STOP`.
3. Show ambulance/fire/police image or video: visual classifier should detect emergency class.
4. Run combined ambulance video + siren: strongest confidence and cleanest response.
5. Stop the clip/audio: after 3 seconds, state returns to `NORMAL`.

The CSV log is saved at `logs/realtime_log.csv`; summarize it with:

```bash
python -m evr.evaluate_logs logs/realtime_log.csv
```

## Suggested Dataset Sources

Use these search terms on Kaggle or public dataset portals:

- `emergency vehicle image dataset ambulance fire truck police`
- `siren sound dataset ambulance police fire truck`
- `UrbanSound8K siren`

Keep a note of dataset source, class counts, and license/usage terms for your report.
