import cv2
import os
from pathlib import Path

INPUT_PATH  = "data/input.mp4"
OUTPUT_PATH = "outputs/output.mp4"

def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"Place your padel video at {INPUT_PATH}")

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {INPUT_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] {width}x{height} @ {fps:.2f} fps, {total} frames")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        label = f"frame={frame_idx}  t={t_sec:6.2f}s"
        cv2.putText(frame, label, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] wrote {frame_idx} frames -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()