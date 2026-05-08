import cv2
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Tunable hyperparameters — adjust here, not inside the class
# ---------------------------------------------------------------------------
BOOTSTRAP_FRAMES   = 45     # frames to build static mask (~1.5 s @ 30 fps)
HSV_S_MAX          = 55     # max saturation to be considered "white"
HSV_V_MIN          = 190    # min brightness  to be considered "white"
MOTION_THRESH      = 18     # absdiff threshold to count as "moving"
AREA_MIN           = 1      # min blob area in px²
AREA_MAX           = 40     # max blob area in px² (5 px ball ≈ 20 px²)
ASPECT_MIN         = 0.3    # min w/h ratio
ASPECT_MAX         = 3.0    # max w/h ratio
GATE_RADIUS_MIN    = 30     # minimum search radius around predicted position (px)
GATE_RADIUS_K      = 3.0    # dynamic gate = K * speed (px/s) * dt
VEL_EMA_ALPHA      = 0.55   # velocity smoothing (higher = tracks fast changes better)
MAX_PREDICTED      = 8      # max consecutive predicted frames before declaring "lost"
PERSON_SHRINK      = 4      # px to shrink person bbox inward before masking
TRAIL_LEN          = 20     # number of past positions to draw as trail

SOURCE_YOLO      = "yolo"
SOURCE_WHITE     = "white"
SOURCE_PREDICTED = "predicted"
SOURCE_LOST      = "lost"

_SRC_COLOR = {
    SOURCE_YOLO:      (0, 255, 0),       # green   — high confidence
    SOURCE_WHITE:     (0, 215, 255),     # yellow  — motion+white candidate
    SOURCE_PREDICTED: (180, 180, 180),   # gray    — extrapolated
    SOURCE_LOST:      (0, 0, 200),       # dark red — for reference only
}

# ---------------------------------------------------------------------------

class BallTracker:
    """
    Hybrid motion∩white ball tracker.

    update() must be called every frame in order.
    Returns a state dict; use draw_ball() for visualization.
    """

    def __init__(self):
        self._boot_buf: list     = []           # bootstrap grayscale frames
        self._static_mask        = None         # always-white pixels (lines/net)
        self._gray_buf           = deque(maxlen=3)  # rolling grey frames for motion

        # Tracker state
        self._xy   = None           # (float, float) | None
        self._vxy  = (0.0, 0.0)     # velocity in px / second
        self._t    = None           # timestamp of last observation (seconds)
        self._src  = SOURCE_LOST
        self._age  = 0              # consecutive predicted/lost frames
        self._trail: deque = deque(maxlen=TRAIL_LEN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, t_sec: float,
               tracks: list, polygon: np.ndarray) -> dict:
        """
        frame   : BGR numpy array
        t_sec   : current timestamp in seconds
        tracks  : list of track dicts from tracker.update()
        polygon : court ROI polygon (numpy int32 array of shape (N,2))

        Returns:
            {
              "xy"    : (x, y) | None,
              "vxy"   : (vx, vy) in px/s,
              "source": one of SOURCE_* constants,
              "age"   : int (consecutive non-observed frames),
              "trail" : list of (x, y, source) tuples
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Bootstrap: collect frames to build static mask ----
        if self._static_mask is None:
            self._boot_buf.append(gray.astype(np.float32))
            self._gray_buf.append(gray)
            self._t = t_sec
            if len(self._boot_buf) >= BOOTSTRAP_FRAMES:
                self._build_static_mask(frame, polygon)
            return self._result()

        H, W = frame.shape[:2]
        dt = max(1e-3, t_sec - self._t) if self._t is not None else 1 / 30.0

        # ---- Step 1: Predict next position ----
        x_pred, y_pred = None, None
        if self._xy is not None:
            vx, vy = self._vxy
            x_pred = float(np.clip(self._xy[0] + vx * dt, 0, W - 1))
            y_pred = float(np.clip(self._xy[1] + vy * dt, 0, H - 1))

        # ---- Step 2: YOLO anchor ----
        yolo_xy = self._best_yolo_hit(tracks, x_pred, y_pred)
        if yolo_xy is not None:
            self._update_state(yolo_xy, t_sec, dt, SOURCE_YOLO)
            self._gray_buf.append(gray)
            self._t = t_sec
            return self._result()

        # ---- Step 3: Motion ∩ White candidate search ----
        self._gray_buf.append(gray)
        candidate = None
        if len(self._gray_buf) >= 2:
            combined = self._candidate_mask(frame, gray, tracks, polygon, H, W)
            candidate = self._best_candidate(combined, x_pred, y_pred, dt)

        if candidate is not None:
            self._update_state(candidate, t_sec, dt, SOURCE_WHITE)

        # ---- Step 4: Fallback to prediction ----
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_static_mask(self, frame, polygon):
        H, W = frame.shape[:2]
        stack  = np.stack(self._boot_buf, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)

        # Pixels that are always bright white = court lines, net top, boards
        _, white_static = cv2.threshold(median, 220, 255, cv2.THRESH_BINARY)

        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)

        self._static_mask = cv2.bitwise_and(white_static, roi_mask)
        n_px = int(np.sum(self._static_mask > 0))
        print(f"[BALL] bootstrap done — static mask has {n_px} always-white px")
        self._boot_buf.clear()

    def _candidate_mask(self, frame, gray, tracks, polygon, H, W):
        # --- Motion mask (moving pixels vs last 1–2 frames) ---
        prev1 = self._gray_buf[-2]
        motion = cv2.absdiff(gray, prev1)
        if len(self._gray_buf) == 3:
            motion = cv2.max(motion, cv2.absdiff(gray, self._gray_buf[0]))
        _, motion_mask = cv2.threshold(motion, MOTION_THRESH, 255, cv2.THRESH_BINARY)

        # --- White mask (HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, HSV_V_MIN), (180, HSV_S_MAX, 255))

        # --- Court ROI mask ---
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)

        # --- Person exclusion mask ---
        person_mask = np.zeros((H, W), dtype=np.uint8)
        for t in tracks:
            if t["name"] == "person":
                x1 = max(0,   int(t["bbox"][0]) + PERSON_SHRINK)
                y1 = max(0,   int(t["bbox"][1]) + PERSON_SHRINK)
                x2 = min(W-1, int(t["bbox"][2]) - PERSON_SHRINK)
                y2 = min(H-1, int(t["bbox"][3]) - PERSON_SHRINK)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)

        # --- Combine ---
        combined = motion_mask & white_mask & roi_mask
        combined = combined & ~self._static_mask
        combined = combined & ~person_mask

        # Small dilation to merge sub-pixel ball fragments
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined

    def _best_candidate(self, mask, x_pred, y_pred, dt):
        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None

        speed = (self._vxy[0]**2 + self._vxy[1]**2) ** 0.5   # px/s
        gate  = max(GATE_RADIUS_MIN, GATE_RADIUS_K * speed * dt)

        # Normalised expected direction
        if speed > 1.0 and self._vxy is not None:
            ex, ey = self._vxy[0] / speed, self._vxy[1] / speed
        else:
            ex, ey = 0.0, 0.0

        best_xy, best_score = None, float("inf")
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w    = int(stats[i, cv2.CC_STAT_WIDTH])
            h    = int(stats[i, cv2.CC_STAT_HEIGHT])
            if not (AREA_MIN <= area <= AREA_MAX):
                continue
            aspect = w / max(1, h)
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            if x_pred is not None:
                dist = ((cx - x_pred)**2 + (cy - y_pred)**2) ** 0.5
                if dist > gate:
                    continue
                # Alignment bonus: prefer blobs in the expected direction
                if speed > 1.0:
                    ax = cx - x_pred; ay = cy - y_pred
                    an = max(1e-6, (ax**2 + ay**2) ** 0.5)
                    alignment = -(ex * ax/an + ey * ay/an)   # -1=perfect align
                    score = dist + 8.0 * alignment
                else:
                    score = dist
            else:
                # No prediction: prefer larger blobs (more reliable)
                score = -area

            if score < best_score:
                best_score, best_xy = score, (cx, cy)

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
        # Multiple YOLO balls: pick nearest to prediction
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


# ---------------------------------------------------------------------------
# Visualization helper (called from main.py)
# ---------------------------------------------------------------------------

def draw_ball(frame: np.ndarray, ball_state: dict) -> None:
    trail = ball_state.get("trail", [])

    # Draw trail polyline
    for i in range(1, len(trail)):
        x0, y0, _  = trail[i - 1]
        x1, y1, s1 = trail[i]
        cv2.line(frame,
                 (int(x0), int(y0)),
                 (int(x1), int(y1)),
                 _SRC_COLOR.get(s1, (255, 255, 255)), 1)

    # Draw current ball dot
    xy  = ball_state.get("xy")
    src = ball_state.get("source", SOURCE_LOST)
    if xy is not None and src != SOURCE_LOST:
        col = _SRC_COLOR.get(src, (255, 255, 255))
        cv2.circle(frame, (int(xy[0]), int(xy[1])), 5, col, -1)
        # Single-letter source tag next to dot
        cv2.putText(frame, src[0].upper(),
                    (int(xy[0]) + 7, int(xy[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
