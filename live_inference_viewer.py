"""
Live inference viewer for the EmotionDetector ESP32 application.

Wire protocol (matches esp32/main/main.cpp):
    "\n===FRAME:<W>:<H>===\n"    ASCII preamble
    <W * H * 2 bytes>            raw big-endian RGB565
    "PRED:c0:p0,c1:p1,...\n"     ASCII prediction line

Usage:
    python live_inference_viewer.py [--port COM3] [--scale 4]

Controls:
    Q  - quit
"""

import argparse
import re
import sys
import time

import cv2
import numpy as np
import serial
import serial.tools.list_ports


BAUD_RATE = 921600          # ignored by USB-CDC but the API needs a value
SERIAL_TIMEOUT = 3.0
PREAMBLE_TAIL = b"===\n"    # the preamble always ends with this

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BAR_COLORS = [(86, 194, 86), (86, 150, 230), (80, 80, 230)]


def find_esp32_port():
    for p in serial.tools.list_ports.comports():
        desc = p.description or ""
        if any(k in desc for k in ("USB JTAG", "CH340", "CP210", "FTDI")):
            return p.device
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None


def decode_rgb565(raw: bytes, w: int, h: int):
    if len(raw) != w * h * 2:
        return None
    pixels = np.frombuffer(raw, dtype=">u2").astype(np.uint16)
    r = ((pixels >> 11) & 0x1F).astype(np.uint8) << 3
    g = ((pixels >>  5) & 0x3F).astype(np.uint8) << 2
    b = ( pixels        & 0x1F).astype(np.uint8) << 3
    rgb = np.stack([r, g, b], axis=-1).reshape(h, w, 3)
    return rgb[..., ::-1]   # RGB -> BGR


def parse_predictions(line: bytes):
    s = line.decode("utf-8", errors="ignore").strip()
    if s.startswith("PRED:"):
        s = s[len("PRED:"):]
    out = []
    for tok in s.split(","):
        if ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        try:
            out.append((k.strip(), float(v)))
        except ValueError:
            pass
    return out


def draw_overlay(frame, class_probs):
    if not class_probs:
        return
    h, w = frame.shape[:2]
    best_cls, best_prob = max(class_probs, key=lambda x: x[1])

    sidebar_w = max(180, w // 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 24
    for idx, (cls, prob) in enumerate(class_probs):
        color = BAR_COLORS[idx % len(BAR_COLORS)]
        bar_w = int((sidebar_w - 10) * max(0.0, min(1.0, prob)))
        cv2.rectangle(frame, (5, y), (5 + bar_w, y + 16), color, -1)
        cv2.putText(frame, f"{cls}: {prob * 100:.1f}%",
                    (8, y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    label = f"{best_cls.upper()}  {best_prob * 100:.0f}%"
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    cv2.putText(frame, label, ((w - tw) // 2, h - 12),
                FONT, 0.7, (0, 255, 180), 2, cv2.LINE_AA)


def read_preamble(ser: serial.Serial):
    """Read up to the next preamble and return (W, H) or None on timeout."""
    chunk = ser.read_until(PREAMBLE_TAIL)
    if not chunk.endswith(PREAMBLE_TAIL):
        return None
    # The chunk ends with "...===FRAME:<W>:<H>===\n". Find that fragment.
    m = re.search(rb"===FRAME:(\d+):(\d+)===\n$", chunk)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser(description="EmotionDetector live viewer")
    parser.add_argument("--port",  default=None, help="Serial port (auto if omitted)")
    parser.add_argument("--baud",  default=BAUD_RATE, type=int)
    parser.add_argument("--scale", default=4, type=int,
                        help="Display scale factor (default 4)")
    args = parser.parse_args()

    port = args.port or find_esp32_port()
    if not port:
        print("ERROR: ESP32 not found. Plug in the device or pass --port.")
        sys.exit(1)

    print(f"Opening {port}...")
    try:
        ser = serial.Serial(port, args.baud, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print("Close idf.py monitor (Ctrl+]) before running this script.")
        sys.exit(1)

    win = "Emotion Detector - Live View"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    scale = max(1, args.scale)

    try:
        ser.reset_input_buffer()
        ser.write(b"S")
        ser.flush()
        print("Sent start byte. Waiting for frames (Q in window to quit)...")

        frame_count = 0
        t0 = time.time()
        while True:
            dims = read_preamble(ser)
            if dims is None:
                print("  Preamble timeout, retrying...")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            w, h = dims

            raw = ser.read(w * h * 2)
            if len(raw) != w * h * 2:
                print(f"  Incomplete frame: got {len(raw)} bytes, expected {w * h * 2}")
                continue

            pred_line = ser.readline()
            probs = parse_predictions(pred_line)

            bgr = decode_rgb565(raw, w, h)
            if bgr is None:
                continue

            disp = cv2.resize(bgr, (w * scale, h * scale),
                              interpolation=cv2.INTER_NEAREST)
            draw_overlay(disp, probs)
            cv2.imshow(win, disp)

            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            if probs:
                best = max(probs, key=lambda x: x[1])
                print(f"[{frame_count:4d}] {best[0]:10s} {best[1] * 100:5.1f}%  "
                      f"|  {fps:.2f} fps  ({w}x{h})")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
