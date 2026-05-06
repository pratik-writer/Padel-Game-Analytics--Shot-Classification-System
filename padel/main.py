### Initial I/O code to read a video, annotate it with frame number and timestamp, and write it back out.

# import cv2
# import os
# from pathlib import Path

# INPUT_PATH  = "data/input.mp4"
# OUTPUT_PATH = "outputs/output.mp4"

# def main():
#     if not Path(INPUT_PATH).exists():
#         raise FileNotFoundError(f"Place your padel video at {INPUT_PATH}")

#     cap = cv2.VideoCapture(INPUT_PATH)
#     if not cap.isOpened():
#         raise RuntimeError(f"OpenCV could not open {INPUT_PATH}")

#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     print(f"[INFO] {width}x{height} @ {fps:.2f} fps, {total} frames")

#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v") #what format does video needs to be compressed in
#     out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#         label = f"frame={frame_idx}  t={t_sec:6.2f}s"
#         cv2.putText(frame, label, (15, 35),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         out.write(frame)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     print(f"[DONE] wrote {frame_idx} frames -> {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()





#integrating YOLOv8

# import cv2
# import os
# from pathlib import Path
# from tracker import Tracker

# INPUT_PATH  = "data/input.mp4"
# OUTPUT_PATH = "outputs/output.mp4"

# COLOR_BY_NAME = {
#     "person": (0, 255, 0),
#     "ball":   (0, 165, 255),
#     "racket": (255, 0, 255),
# }

# def draw_tracks(frame, tracks):
#     for t in tracks:
#         x1, y1, x2, y2 = map(int, t["bbox"])
#         color = COLOR_BY_NAME.get(t["name"], (200, 200, 200))
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         label = f'{t["name"]} #{t["id"]} {t["conf"]:.2f}'
#         cv2.putText(frame, label, (x1, max(0, y1 - 8)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# def main():
#     if not Path(INPUT_PATH).exists():
#         raise FileNotFoundError(f"Place your padel video at {INPUT_PATH}")

#     cap = cv2.VideoCapture(INPUT_PATH)
#     if not cap.isOpened():
#         raise RuntimeError(f"OpenCV could not open {INPUT_PATH}")

#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"[INFO] {width}x{height} @ {fps:.2f} fps, {total} frames")

#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     out = cv2.VideoWriter(OUTPUT_PATH,
#                           cv2.VideoWriter_fourcc(*"mp4v"),
#                           fps, (width, height))

#     # tracker = Tracker(weights="yolov8n.pt", conf=0.30)
#     tracker = Tracker(weights="yolov8s.pt", base_conf=0.10, imgsz=1280)

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#         tracks = tracker.update(frame)
#         draw_tracks(frame, tracks)

#         cv2.putText(frame, f"frame={frame_idx}  t={t_sec:6.2f}s  dets={len(tracks)}",
#                     (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         out.write(frame)
#         frame_idx += 1
#         if frame_idx % 30 == 0:
#             print(f"[..] frame {frame_idx}/{total}  tracks={len(tracks)}")

#     cap.release()
#     out.release()
#     print(f"[DONE] wrote {frame_idx} frames -> {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()

#stage 3/4
#adding pose estimation with mediapipe

import cv2
import os
from pathlib import Path
from tracker import Tracker
from pose import PoseEstimator, POSE_CONNECTIONS

INPUT_PATH  = "data/input.mp4"
OUTPUT_PATH = "outputs/output.mp4"

COLOR_BY_NAME = {
    "person": (0, 255, 0),
    "ball":   (0, 165, 255),
    "racket": (255, 0, 255),
}
SKELETON_COLOR = (255, 255, 0)
KEYPOINT_COLOR = (0, 255, 255)
VIS_THRESH = 0.3

def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = COLOR_BY_NAME.get(t["name"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{t["name"]} #{t["id"]} {t["conf"]:.2f}',
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_poses(frame, poses):
    for pid, kps in poses.items():
        for a, b in POSE_CONNECTIONS:
            if a in kps and b in kps and kps[a][2] > VIS_THRESH and kps[b][2] > VIS_THRESH:
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, pa, pb, SKELETON_COLOR, 2)
        for name, (x, y, v) in kps.items():
            if v > VIS_THRESH:
                cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)

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
    out = cv2.VideoWriter(OUTPUT_PATH,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))

    tracker  = Tracker(weights="yolov8s.pt", base_conf=0.10, imgsz=1280)
    poser    = PoseEstimator(model_complexity=1)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_sec  = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            tracks = tracker.update(frame)
            poses  = poser.estimate_for_tracks(frame, tracks)

            draw_tracks(frame, tracks)
            draw_poses(frame, poses)

            cv2.putText(frame,
                        f"frame={frame_idx} t={t_sec:6.2f}s dets={len(tracks)} poses={len(poses)}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[..] frame {frame_idx}/{total}  tracks={len(tracks)}  poses={len(poses)}")
    finally:
        cap.release()
        out.release()
        poser.close()
    print(f"[DONE] wrote {frame_idx} frames -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()