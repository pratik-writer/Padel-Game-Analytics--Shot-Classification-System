import cv2
import numpy as np
from collections import deque

BOOTSTRAP_FRAMES   = 45     # frames used to build the static-white mask
HSV_S_MAX          = 70
HSV_V_MIN          = 160
MOTION_THRESH      = 14
AREA_MIN           = 1
AREA_MAX           = 40     # px^2 for a compact ball
STREAK_AREA_MAX    = 250    # px^2 for a motion-blurred streak
ASPECT_MIN         = 0.3
ASPECT_MAX         = 3.0
STREAK_ASPECT_MAX  = 10.0
GATE_RADIUS_MIN    = 35
GATE_RADIUS_K      = 3.5    # dynamic gate = K * speed * dt
VEL_EMA_ALPHA      = 0.6
MAX_PREDICTED      = 15     # consecutive predicted frames before declaring 'lost'
PERSON_SHRINK      = 4      # px shrunk inward off person bbox before masking
TRAIL_LEN          = 25
ALIGN_DOT_MIN      = 0.4    # streak must point along velocity (cos angle)

SOURCE_YOLO      = "yolo"
SOURCE_WHITE     = "white"
SOURCE_PREDICTED = "predicted"
SOURCE_LOST      = "lost"

_SRC_COLOR = {
    SOURCE_YOLO:      (0, 255, 0),
    SOURCE_WHITE:     (0, 215, 255),
    SOURCE_PREDICTED: (180, 180, 180),
    SOURCE_LOST:      (0, 0, 200),
}


class BallTracker:
    def __init__(self):
        self._boot_buf: list = []
        self._static_mask    = None
        self._gray_buf       = deque(maxlen=3)

        self._xy   = None
        self._vxy  = (0.0, 0.0)        # px/s
        self._t    = None
        self._src  = SOURCE_LOST
        self._age  = 0
        self._trail: deque = deque(maxlen=TRAIL_LEN)

    def update(self, frame: np.ndarray, t_sec: float,
               tracks: list, polygon: np.ndarray) -> dict:
        # returns: {xy, vxy, source, age, trail}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._static_mask is None:
            self._boot_buf.append(gray.astype(np.float32))
            self._gray_buf.append(gray)
            self._t = t_sec
            if len(self._boot_buf) >= BOOTSTRAP_FRAMES:
                self._build_static_mask(frame, polygon)
            return self._result()

        H, W = frame.shape[:2]
        dt = max(1e-3, t_sec - self._t) if self._t is not None else 1 / 30.0

        x_pred, y_pred = None, None
        if self._xy is not None:
            vx, vy = self._vxy
            x_pred = float(np.clip(self._xy[0] + vx * dt, 0, W - 1))
            y_pred = float(np.clip(self._xy[1] + vy * dt, 0, H - 1))

        yolo_xy = self._best_yolo_hit(tracks, x_pred, y_pred)
        if yolo_xy is not None:
            self._update_state(yolo_xy, t_sec, dt, SOURCE_YOLO)
            self._gray_buf.append(gray)
            self._t = t_sec
            return self._result()

        self._gray_buf.append(gray)
        candidate = None
        if len(self._gray_buf) >= 2:
            combined = self._candidate_mask(frame, gray, tracks, polygon, H, W)
            candidate = self._best_candidate(combined, x_pred, y_pred, dt)

        if candidate is not None:
            self._update_state(candidate, t_sec, dt, SOURCE_WHITE)
        elif self._xy is not None and self._age < MAX_PREDICTED and x_pred is not None:
            self._xy  = (x_pred, y_pred)
            self._src = SOURCE_PREDICTED
            self._age += 1
            self._trail.append((x_pred, y_pred, SOURCE_PREDICTED))
        else:
            self._src = SOURCE_LOST
            self._age += 1

        self._t = t_sec
        return self._result()

    def _build_static_mask(self, frame, polygon):
        H, W = frame.shape[:2]
        stack  = np.stack(self._boot_buf, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)
        _, white_static = cv2.threshold(median, 220, 255, cv2.THRESH_BINARY)

        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)

        self._static_mask = cv2.bitwise_and(white_static, roi_mask)
        n_px = int(np.sum(self._static_mask > 0))
        print(f"[BALL] bootstrap done. static mask has {n_px} always-white px")
        self._boot_buf.clear()

    def _candidate_mask(self, frame, gray, tracks, polygon, H, W):
        prev1 = self._gray_buf[-2]
        motion = cv2.absdiff(gray, prev1)
        if len(self._gray_buf) == 3:
            motion = cv2.max(motion, cv2.absdiff(gray, self._gray_buf[0]))
        _, motion_mask = cv2.threshold(motion, MOTION_THRESH, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, HSV_V_MIN), (180, HSV_S_MAX, 255))

        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)

        person_mask = np.zeros((H, W), dtype=np.uint8)
        for t in tracks:
            if t["name"] == "person":
                x1 = max(0,   int(t["bbox"][0]) + PERSON_SHRINK)
                y1 = max(0,   int(t["bbox"][1]) + PERSON_SHRINK)
                x2 = min(W-1, int(t["bbox"][2]) - PERSON_SHRINK)
                y2 = min(H-1, int(t["bbox"][3]) - PERSON_SHRINK)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)

        combined = motion_mask & white_mask & roi_mask
        combined = combined & ~self._static_mask
        combined = combined & ~person_mask

        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined

    def _best_candidate(self, mask, x_pred, y_pred, dt):
        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None

        speed = (self._vxy[0]**2 + self._vxy[1]**2) ** 0.5
        gate  = max(GATE_RADIUS_MIN, GATE_RADIUS_K * speed * dt)

        if speed > 1.0:
            ex, ey = self._vxy[0] / speed, self._vxy[1] / speed
        else:
            ex, ey = 0.0, 0.0

        best_xy, best_score = None, float("inf")
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w    = int(stats[i, cv2.CC_STAT_WIDTH])
            h    = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx   = float(centroids[i][0])
            cy   = float(centroids[i][1])
            aspect_long = max(w, h) / max(1, min(w, h))

            is_compact = (AREA_MIN <= area <= AREA_MAX
                          and (1.0 / aspect_long) >= ASPECT_MIN)

            is_streak = False
            streak_endpoint = None
            if speed > 50.0 and not is_compact:
                if (AREA_MIN <= area <= STREAK_AREA_MAX
                        and aspect_long <= STREAK_ASPECT_MAX):
                    if w >= h:
                        sdx, sdy = 1.0, 0.0
                        ep_left  = (stats[i, cv2.CC_STAT_LEFT], cy)
                        ep_right = (stats[i, cv2.CC_STAT_LEFT] + w - 1, cy)
                    else:
                        sdx, sdy = 0.0, 1.0
                        ep_left  = (cx, stats[i, cv2.CC_STAT_TOP])
                        ep_right = (cx, stats[i, cv2.CC_STAT_TOP] + h - 1)
                    align = abs(sdx * ex + sdy * ey)
                    if align >= ALIGN_DOT_MIN:
                        is_streak = True
                        d_left  = (ep_left[0]  - (x_pred or cx)) * ex + (ep_left[1]  - (y_pred or cy)) * ey
                        d_right = (ep_right[0] - (x_pred or cx)) * ex + (ep_right[1] - (y_pred or cy)) * ey
                        streak_endpoint = ep_right if d_right > d_left else ep_left

            if not (is_compact or is_streak):
                continue

            cand_xy = streak_endpoint if is_streak else (cx, cy)

            if x_pred is not None:
                dist = ((cand_xy[0] - x_pred)**2 + (cand_xy[1] - y_pred)**2) ** 0.5
                if dist > gate:
                    continue
                if speed > 1.0:
                    ax = cand_xy[0] - x_pred; ay = cand_xy[1] - y_pred
                    an = max(1e-6, (ax**2 + ay**2) ** 0.5)
                    alignment = -(ex * ax/an + ey * ay/an)
                    score = dist + 8.0 * alignment
                else:
                    score = dist
                if is_streak:
                    score -= 5.0
            else:
                score = -area

            if score < best_score:
                best_score, best_xy = score, cand_xy

        return best_xy

    def _best_yolo_hit(self, tracks, x_pred, y_pred):
        balls = [t for t in tracks if t["name"] == "ball"]
        if not balls:
            return None
        to_center = lambda t: (
            0.5 * (t["bbox"][0] + t["bbox"][2]),
            0.5 * (t["bbox"][1] + t["bbox"][3])
        )
        if len(balls) == 1:
            return to_center(balls[0])
        if x_pred is None:
            return to_center(balls[0])
        return min(
            (to_center(b) for b in balls),
            key=lambda p: (p[0]-x_pred)**2 + (p[1]-y_pred)**2
        )

    def _update_state(self, xy, t_sec, dt, source):
        if self._xy is not None:
            raw_vx = (xy[0] - self._xy[0]) / dt
            raw_vy = (xy[1] - self._xy[1]) / dt
            ovx, ovy = self._vxy
            self._vxy = (
                VEL_EMA_ALPHA * raw_vx + (1 - VEL_EMA_ALPHA) * ovx,
                VEL_EMA_ALPHA * raw_vy + (1 - VEL_EMA_ALPHA) * ovy,
            )
        self._xy  = xy
        self._src = source
        self._age = 0
        self._trail.append((xy[0], xy[1], source))

    def _result(self) -> dict:
        return {
            "xy":     self._xy,
            "vxy":    self._vxy,
            "source": self._src,
            "age":    self._age,
            "trail":  list(self._trail),
        }


def draw_ball(frame: np.ndarray, ball_state: dict) -> None:
    trail = ball_state.get("trail", [])
    for i in range(1, len(trail)):
        x0, y0, _  = trail[i - 1]
        x1, y1, s1 = trail[i]
        cv2.line(frame,
                 (int(x0), int(y0)),
                 (int(x1), int(y1)),
                 _SRC_COLOR.get(s1, (255, 255, 255)), 1)

    xy  = ball_state.get("xy")
    src = ball_state.get("source", SOURCE_LOST)
    if xy is not None and src != SOURCE_LOST:
        col = _SRC_COLOR.get(src, (255, 255, 255))
        cv2.circle(frame, (int(xy[0]), int(xy[1])), 5, col, -1)
        cv2.putText(frame, src[0].upper(),
                    (int(xy[0]) + 7, int(xy[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
