from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect image samples from webcam/Pi camera.")
    parser.add_argument("label", help="Class name, e.g. ambulance, fire_truck, police, normal")
    parser.add_argument("--out", type=Path, default=Path("data/images"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between saved frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.out / args.label
    target.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try --camera 1 for an external USB webcam.")

    saved = 0
    last_save = 0.0
    print("Press q to stop early. Saved images go to:", target)

    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_save >= args.delay:
                file_path = target / f"{args.label}_{int(now * 1000)}.jpg"
                cv2.imwrite(str(file_path), frame)
                saved += 1
                last_save = now

            cv2.putText(
                frame,
                f"{args.label}: {saved}/{args.count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Collect Images", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} images in {target}")


if __name__ == "__main__":
    main()
