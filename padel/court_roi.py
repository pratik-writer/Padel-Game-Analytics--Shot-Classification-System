import json
import os
from pathlib import Path
import cv2
import numpy as np

ROI_FILE = "court_roi.json"

def calibrate(video_path: str, roi_file: str = ROI_FILE) -> list:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    pts = []
    win = "Click 4 court corners (ENTER=save, r=reset, ESC=cancel)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([int(x), int(y)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, tuple(p), 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(pts) >= 2:
            cv2.polylines(disp, [np.array(pts, dtype=np.int32)], len(pts) == 4,
                          (0, 255, 0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Calibration aborted.")
        if key == ord('r'):
            pts.clear()
        if key in (13, 10) and len(pts) == 4:   # ENTER
            break
    cv2.destroyAllWindows()

    with open(roi_file, "w") as f:
        json.dump({"polygon": pts}, f, indent=2)
    print(f"[ROI] saved -> {roi_file}: {pts}")
    return pts


def load_or_calibrate(video_path: str, roi_file: str = ROI_FILE) -> np.ndarray:
    env_roi = os.environ.get("PADEL_ROI")        # "x1,y1;x2,y2;x3,y3;x4,y4"
    if env_roi:
        pts = [list(map(int, p.split(","))) for p in env_roi.split(";")]
        if len(pts) >= 3:
            print(f"[ROI] loaded from PADEL_ROI env var: {pts}")
            return np.array(pts, dtype=np.int32)

    if not Path(roi_file).exists():
        if os.environ.get("PADEL_HEADLESS"):
            raise RuntimeError(
                f"No ROI file ({roi_file}) and PADEL_HEADLESS set. "
                "Provide PADEL_ROI=\"x1,y1;x2,y2;x3,y3;x4,y4\" or upload court_roi.json."
            )
        print("[ROI] no saved polygon, launching calibration window...")
        calibrate(video_path, roi_file)
    with open(roi_file) as f:
        pts = json.load(f)["polygon"]
    return np.array(pts, dtype=np.int32)


def foot_point(bbox):
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), y2)


def filter_tracks_in_court(tracks, polygon: np.ndarray, max_persons: int = 4):
    persons, others = [], []
    for t in tracks:
        if t["name"] == "person":
            fx, fy = foot_point(t["bbox"])
            inside = cv2.pointPolygonTest(polygon, (float(fx), float(fy)), False) >= 0
            if inside:
                persons.append(t)
        else:
            others.append(t)
    persons.sort(key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]),
                 reverse=True)
    return others + persons[:max_persons]


def draw_court(frame, polygon: np.ndarray, color=(0, 200, 255)):
    cv2.polylines(frame, [polygon], True, color, 2)