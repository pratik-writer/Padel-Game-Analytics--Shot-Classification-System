"""
Contact detection: find frames where the ball trajectory bends sharply
near a player's wrist. These are the only valid 'shot taken' moments.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List

# ----------------------------- Tunables -----------------------------
WINDOW           = 7      # frames buffered for in/out velocity estimation (must be odd)
HALF             = WINDOW // 2
MIN_SPEED        = 60.0   # px/s — both incoming and outgoing must exceed this
TURN_ANGLE_MIN   = 50.0   # degrees — angular change required to count as contact
WRIST_DIST_MAX   = 75.0   # px — contact point must be within this of a wrist
PLAYER_COOLDOWN  = 0.5    # s — minimum gap between two contacts on same player
# ---------------------------------------------------------------------

@dataclass
class ContactEvent:
    frame: int          # frame index where contact occurred (mid-window)
    timestamp: float    # seconds
    player_id: int      # nearest player at contact moment
    contact_xy: tuple   # (x, y) ball position at contact
    in_dir:   tuple     # unit vector — incoming direction
    out_dir:  tuple     # unit vector — outgoing direction
    in_speed:  float    # px/s
    out_speed: float    # px/s
    turn_angle: float   # degrees
    wrist_dist: float   # px to chosen wrist
    wrist_side: str     # 'r' or 'l'
    confidence: str     # 'high' / 'med' / 'low'


def _angle_deg(v1, v2):
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def _unit(v):
    n = math.hypot(*v)
    return (v[0]/n, v[1]/n) if n > 1e-6 else (0.0, 0.0)


class ContactDetector:
    """
    Buffers (frame_idx, t, ball_xy, ball_source, poses) for the last WINDOW frames
    and emits ContactEvents at the moment the ball trajectory bends near a wrist.
    """

    def __init__(self):
        self.buf = deque(maxlen=WINDOW)
        self.last_event_t: Dict[int, float] = {}

    def update(self, frame_idx: int, t_sec: float,
               ball_state: dict, poses: Dict[int, dict]) -> Optional[ContactEvent]:
        xy  = ball_state.get("xy")
        src = ball_state.get("source", "lost")
        self.buf.append((frame_idx, t_sec, xy, src, poses))
        if len(self.buf) < WINDOW:
            return None

        # Need ball xy in BOTH halves of the window
        first_half = list(self.buf)[:HALF + 1]   # includes mid
        last_half  = list(self.buf)[HALF:]       # includes mid
        if any(b[2] is None for b in first_half) or any(b[2] is None for b in last_half):
            return None

        mid_frame, mid_t, mid_xy, mid_src, mid_poses = self.buf[HALF]

        # Incoming velocity = (mid - start) / dt
        x0, y0 = first_half[0][2]
        x_m, y_m = mid_xy
        x_e, y_e = last_half[-1][2]
        dt_in  = mid_t - first_half[0][1]
        dt_out = last_half[-1][1] - mid_t
        if dt_in < 1e-3 or dt_out < 1e-3:
            return None

        v_in  = ((x_m - x0) / dt_in,  (y_m - y0) / dt_in)
        v_out = ((x_e - x_m) / dt_out,(y_e - y_m) / dt_out)
        s_in  = math.hypot(*v_in)
        s_out = math.hypot(*v_out)
        if s_in < MIN_SPEED or s_out < MIN_SPEED:
            return None

        turn = _angle_deg(v_in, v_out)
        if turn < TURN_ANGLE_MIN:
            return None

        # --- Find nearest wrist across all players & both arms at mid frame ---
        best = None  # (dist, pid, side)
        for pid, kps in mid_poses.items():
            for side in ("r", "l"):
                wk = f"{side}_wrist"
                if wk not in kps or kps[wk][2] < 0.3:
                    continue
                wx, wy, _ = kps[wk]
                d = math.hypot(wx - x_m, wy - y_m)
                if best is None or d < best[0]:
                    best = (d, pid, side)
        if best is None or best[0] > WRIST_DIST_MAX:
            return None

        wrist_dist, pid, wside = best

        # Per-player cooldown
        if (mid_t - self.last_event_t.get(pid, -1e9)) < PLAYER_COOLDOWN:
            return None
        self.last_event_t[pid] = mid_t

        # Confidence: depends on ball source quality at contact + wrist distance
        if mid_src in ("yolo", "white") and wrist_dist < 40:
            conf = "high"
        elif mid_src == "predicted" or wrist_dist < 60:
            conf = "med"
        else:
            conf = "low"

        return ContactEvent(
            frame=mid_frame,
            timestamp=mid_t,
            player_id=pid,
            contact_xy=(float(x_m), float(y_m)),
            in_dir=_unit(v_in),
            out_dir=_unit(v_out),
            in_speed=s_in,
            out_speed=s_out,
            turn_angle=turn,
            wrist_dist=wrist_dist,
            wrist_side=wside,
            confidence=conf,
        )
