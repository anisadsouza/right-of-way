"""Send commands to the ESP32 over serial and check each reply.

Use this instead of the idf.py serial monitor when the monitor will not accept typing.
Close any serial monitor first: only one program can hold the port.

Run from ev_recognition/ with the venv active (needs pyserial):

    python serial_check.py /dev/cu.usbserial-0001                 sends NORMAL, SLOW, PULL_RIGHT, STOP
    python serial_check.py /dev/cu.usbserial-0001 --command SLOW  sends one command
    python serial_check.py /dev/cu.usbserial-0001 --interactive   type commands yourself, empty line to quit

Keep the motor battery disconnected, or the wheels raised, for the first run.
"""
import argparse
import sys
import time

import serial

COMMANDS = ["NORMAL", "SLOW", "PULL_RIGHT", "STOP"]


def read_lines(port, seconds):
    """Collect lines that arrive within the given time."""
    lines = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        raw = port.readline()
        if raw:
            lines.append(raw.decode("ascii", errors="replace").strip())
    return [line for line in lines if line]


def main():
    parser = argparse.ArgumentParser(description="Check ESP32 serial commands.")
    parser.add_argument("port", help="For example /dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--command", choices=COMMANDS, help="Send only this command")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--boot-wait", type=float, default=2.5, help="Seconds to wait after opening the port, because opening it can reset the board")
    parser.add_argument("--reply-wait", type=float, default=1.0, help="Seconds to wait for each reply")
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds between commands so you can watch the outputs")
    args = parser.parse_args()

    try:
        port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as error:
        sys.exit(f"Could not open {args.port}: {error}. Close the serial monitor and any other program using the port, then check the name with: ls /dev/cu.*")

    with port:
        boot = read_lines(port, args.boot_wait)
        print("Boot output: " + (" | ".join(boot) if boot else "(none)"))
        port.reset_input_buffer()

        if args.interactive:
            print("Type a command and press Enter. Empty line to quit.")
            while True:
                text = input("> ").strip()
                if not text:
                    break
                port.write(f"{text}\n".encode("ascii"))
                print("Reply: " + (" | ".join(read_lines(port, args.reply_wait)) or "(none)"))
            port.write(b"STOP\n")
            return

        failed = 0
        for command in ([args.command] if args.command else COMMANDS):
            port.write(f"{command}\n".encode("ascii"))
            replies = read_lines(port, args.reply_wait)
            ok = f"OK {command}" in replies
            failed += not ok
            print(f"{command}: {' | '.join(replies) if replies else '(no reply)'} -> {'pass' if ok else 'FAIL'}")
            time.sleep(args.pause)
        port.write(b"STOP\n")
        print(f"{len(COMMANDS) if not args.command else 1} command(s) sent, {failed} failed. STOP sent at the end.")


if __name__ == "__main__":
    main()