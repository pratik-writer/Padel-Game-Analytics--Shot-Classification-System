import os
import cv2
import numpy as np
from collections import deque

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "tracknet_weights.pt")
TRACKNET_W   = 640
TRACKNET_H   = 360

KALMAN_PROCESS_NOISE     = 5.0
KALMAN_MEASUREMENT_NOISE = 2.0
MAX_JUMP_PX        = 250
MAX_PREDICT_FRAMES = 3
MAX_LOST_FRAMES    = 8
TRAIL_LEN          = 25
PERSON_SHRINK      = 4

SOURCE_TRACKNET  = "yolo"        # treated as 'measured' by contact.py
SOURCE_PREDICTED = "predicted"
SOURCE_LOST      = "lost"

_SRC_COLOR = {
    SOURCE_TRACKNET:  (0, 255, 0),
    SOURCE_PREDICTED: (180, 180, 180),
    SOURCE_LOST:      (0, 0, 200),
}


class _TrackNetDetector:
    def __init__(self, weights_path: str):
        import torch
        from tracknet import BallTrackerNet
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BallTrackerNet()
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()
        self.frame_buffer: list = []
        self.video_w = None
        self.video_h = None

    def _preprocess(self, frame):
        resized = cv2.resize(frame, (TRACKNET_W, TRACKNET_H))
        return resized.astype(np.float32) / 255.0

    def _postprocess(self, output):
        out = output.argmax(dim=1).detach().cpu().numpy()
        heatmap = (out.reshape(TRACKNET_H, TRACKNET_W) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        if thresh.max() == 0:
            return None
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                                   param1=50, param2=2, minRadius=2, maxRadius=7)
        if circles is None:
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            largest = max(cnts, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = float(circles[0][0][0]), float(circles[0][0][1])
        sx = self.video_w / TRACKNET_W
        sy = self.video_h / TRACKNET_H
        return (cx * sx, cy * sy)

    def detect(self, frame):
        if self.video_w is None:
            self.video_h, self.video_w = frame.shape[:2]
        self.frame_buffer.append(self._preprocess(frame))
        if len(self.frame_buffer) < 3:
            return None
        if len(self.frame_buffer) > 3:
            self.frame_buffer = self.frame_buffer[-3:]
        stacked = np.concatenate(self.frame_buffer, axis=2)
        x = self._torch.from_numpy(stacked.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            out = self.model(x, testing=True)
        return self._postprocess(out)


def _build_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
    return kf


class BallTracker:
    """
    TrackNet detector + Kalman filter. Falls back to legacy motion+white tracker
    if weights or torch are unavailable.

    update() returns: {xy, vxy, source, age, trail}
        xy:     (x, y) in pixels or None
        vxy:    (vx, vy) in pixels per second
        source: 'yolo' (= TrackNet measurement), 'predicted' (Kalman gap-fill), 'lost'
        age:    consecutive frames without a fresh measurement
        trail:  list of (x, y, source) for drawing
    """

    def __init__(self):
        self._legacy = None
        self.detector = None
        try:
            self.detector = _TrackNetDetector(WEIGHTS_PATH)
            print(f"[BALL] TrackNet loaded from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[BALL] TrackNet unavailable ({e}); using legacy motion+white tracker")
            from ball_legacy import BallTracker as _LegacyTracker
            self._legacy = _LegacyTracker()

        self.kf = _build_kalman()
        self.kf_init = False
        self._xy: tuple = None
        self._vxy = (0.0, 0.0)
        self._t   = None
        self._src = SOURCE_LOST
        self._age = 0
        self._trail: deque = deque(maxlen=TRAIL_LEN)

    def update(self, frame: np.ndarray, t_sec: float,
               tracks: list, polygon: np.ndarray) -> dict:
        if self._legacy is not None:
            return self._legacy.update(frame, t_sec, tracks, polygon)

        H, W = frame.shape[:2]
        dt = max(1e-3, (t_sec - self._t)) if self._t is not None else 1 / 30.0

        raw = self.detector.detect(frame)

        if raw is not None and not _inside_polygon(raw, polygon):
            raw = None

        if raw is not None and _hits_person(raw, tracks):
            raw = None

        if raw is not None and self.kf_init:
            pred = self._kf_predict()
            if _dist(raw, pred) > MAX_JUMP_PX and self._xy is not None and \
               _dist(raw, self._xy) > MAX_JUMP_PX:
                raw = None

        if raw is not None:
            if not self.kf_init:
                self._kf_seed(raw)
            else:
                self._kf_predict()
                m = np.array([[np.float32(raw[0])], [np.float32(raw[1])]])
                corr = self.kf.correct(m)
                smooth = (float(corr[0][0]), float(corr[1][0]))
                vx_pf, vy_pf = float(corr[2][0]), float(corr[3][0])
                fps_eq = 1.0 / dt
                self._xy  = smooth
                self._vxy = (vx_pf * fps_eq, vy_pf * fps_eq)
            self._src = SOURCE_TRACKNET
            self._age = 0
            self._trail.append((self._xy[0], self._xy[1], self._src))
        elif self.kf_init and self._age < MAX_PREDICT_FRAMES:
            pred = self._kf_predict()
            self._xy  = (float(np.clip(pred[0], 0, W - 1)),
                         float(np.clip(pred[1], 0, H - 1)))
            self._src = SOURCE_PREDICTED
            self._age += 1
            self._trail.append((self._xy[0], self._xy[1], self._src))
        else:
            self._age += 1
            if self._age >= MAX_LOST_FRAMES:
                self._reset()
            else:
                self._src = SOURCE_LOST

        self._t = t_sec
        return self._result()

    def _kf_seed(self, xy):
        s = np.array([[np.float32(xy[0])], [np.float32(xy[1])], [0.0], [0.0]], np.float32)
        self.kf.statePre  = s.copy()
        self.kf.statePost = s.copy()
        self.kf_init = True
        self._xy = (float(xy[0]), float(xy[1]))
        self._vxy = (0.0, 0.0)

    def _kf_predict(self):
        p = self.kf.predict()
        return (float(p[0][0]), float(p[1][0]))

    def _reset(self):
        self.kf = _build_kalman()
        self.kf_init = False
        self._xy = None
        self._vxy = (0.0, 0.0)
        self._src = SOURCE_LOST

    def _result(self) -> dict:
        return {
            "xy":     self._xy,
            "vxy":    self._vxy,
            "source": self._src,
            "age":    self._age,
            "trail":  list(self._trail),
        }


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _inside_polygon(xy, polygon) -> bool:
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (float(xy[0]), float(xy[1])), False) >= 0


def _hits_person(xy, tracks) -> bool:
    px, py = xy
    for t in tracks:
        if t.get("name") != "person":
            continue
        x1, y1, x2, y2 = t["bbox"]
        x1 += PERSON_SHRINK; y1 += PERSON_SHRINK
        x2 -= PERSON_SHRINK; y2 -= PERSON_SHRINK
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def draw_ball(frame: np.ndarray, ball_state: dict) -> None:
    trail = ball_state.get("trail", [])
    for i in range(1, len(trail)):
        x0, y0, _  = trail[i - 1]
        x1, y1, s1 = trail[i]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                 _SRC_COLOR.get(s1, (255, 255, 255)), 1)
    xy  = ball_state.get("xy")
    src = ball_state.get("source", SOURCE_LOST)
    if xy is not None and src != SOURCE_LOST:
        col = _SRC_COLOR.get(src, (255, 255, 255))
        cv2.circle(frame, (int(xy[0]), int(xy[1])), 5, col, -1)
        cv2.putText(frame, src[0].upper(),
                    (int(xy[0]) + 7, int(xy[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
