# from collections import deque
# from dataclasses import dataclass
# from typing import Optional, Dict

# VIS_THRESH         = 0.3      
# SMASH_Y_RATIO      = 0.20     
# SIDE_X_RATIO       = 0.10     
# WRIST_SPEED_TRIG   = 0.6      
# COOLDOWN_SEC       = 0.7
# HISTORY_LEN        = 6        
# ASSUME_RIGHT_HANDED = True

# @dataclass
# class ShotEvent:
#     frame: int
#     timestamp: float
#     player_id: int
#     shot_type: str          
#     side: str               
#     confidence: str         

# class _PlayerState:
#     __slots__ = ("wrist_hist", "last_event_t", "armed")
#     def __init__(self):
#         self.wrist_hist = deque(maxlen=HISTORY_LEN)  # (t, x, y, shoulder_w)
#         self.last_event_t = -1e9
#         self.armed = True   # becomes False after trigger, re-arms after cooldown

# class ShotClassifier:
#     def __init__(self):
#         self.players: Dict[int, _PlayerState] = {}

#     def _state(self, pid: int) -> _PlayerState:
#         if pid not in self.players:
#             self.players[pid] = _PlayerState()
#         return self.players[pid]

#     def update(self, frame_idx: int, t_sec: float,
#                poses: Dict[int, dict]) -> list[ShotEvent]:
#         events = []
#         for pid, kps in poses.items():
#             ev = self._update_one(frame_idx, t_sec, pid, kps)
#             if ev is not None:
#                 events.append(ev)
#         return events

#     def _update_one(self, frame_idx, t_sec, pid, kps) -> Optional[ShotEvent]:
#         # Pick dominant arm. Right-handed assumption => right wrist/shoulder/elbow.
#         side_kp = "r" if ASSUME_RIGHT_HANDED else "l"
#         wrist_k    = f"{side_kp}_wrist"
#         shoulder_k = f"{side_kp}_shoulder"
#         elbow_k    = f"{side_kp}_elbow"
#         other_sh   = "l_shoulder" if side_kp == "r" else "r_shoulder"

#         for k in (wrist_k, shoulder_k, elbow_k, other_sh):
#             if k not in kps or kps[k][2] < VIS_THRESH:
#                 return None

#         wx, wy, _ = kps[wrist_k]
#         sx, sy, _ = kps[shoulder_k]
#         ex, ey, _ = kps[elbow_k]
#         osx, osy, _ = kps[other_sh]

#         body_cx     = 0.5 * (sx + osx)
#         shoulder_w  = max(1.0, abs(sx - osx))
#         torso_h_est = shoulder_w * 1.5  # rough proxy if hips missing

#         st = self._state(pid)
#         st.wrist_hist.append((t_sec, wx, wy))

#         # Compute wrist speed in shoulder-widths per second (scale-invariant)
#         speed = 0.0
#         if len(st.wrist_hist) >= 2:
#             t0, x0, y0 = st.wrist_hist[0]
#             t1, x1, y1 = st.wrist_hist[-1]
#             dt = max(1e-3, t1 - t0)
#             dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
#             speed = (dist / shoulder_w) / dt

#         # Cooldown: re-arm trigger after COOLDOWN_SEC has passed
#         if not st.armed and (t_sec - st.last_event_t) >= COOLDOWN_SEC:
#             st.armed = True
#         if not st.armed:
#             return None

#         # Trigger: wrist must be moving fast
#         if speed < WRIST_SPEED_TRIG:
#             return None

#         # --- Classify ---
#         # 1. Serve/Smash: wrist clearly above shoulder (y is smaller in image coords)
#         if (sy - wy) > SMASH_Y_RATIO * torso_h_est:
#             shot_type, side = "Serve/Smash", "overhead"
#         else:
#             # 2. Forehand vs Backhand based on wrist-x vs body center
#             dx = wx - body_cx                     
#             min_dx = SIDE_X_RATIO * shoulder_w
#             if abs(dx) < min_dx:
#                 return None  # too central -> ambiguous, skip
#             on_dominant_side = (dx > 0) if side_kp == "r" else (dx < 0)
#             if on_dominant_side:
#                 shot_type, side = "Forehand", "dominant"
#             else:
#                 shot_type, side = "Backhand", "non-dominant"

#         st.last_event_t = t_sec
#         st.armed = False

#         conf = "high" if speed > 1.5 * WRIST_SPEED_TRIG else "med"
#         return ShotEvent(frame_idx, t_sec, pid, shot_type, side, conf)



import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List

VIS_THRESH         = 0.3
SMASH_Y_RATIO      = 0.20
SIDE_X_RATIO       = 0.10
WRIST_SPEED_TRIG   = 0.8      
ELBOW_ANGLE_MIN    = 130.0    
COOLDOWN_SEC       = 0.7
HISTORY_LEN        = 8
SPEED_BUFFER_LEN   = 5        
ASSUME_RIGHT_HANDED = True

@dataclass
class ShotEvent:
    frame: int
    timestamp: float
    player_id: int
    shot_type: str
    side: str
    confidence: str

def _angle_deg(a, b, c):
    """Angle ABC at vertex B, in degrees."""
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na  = math.hypot(bax, bay)
    nc  = math.hypot(bcx, bcy)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))

class _PlayerState:
    __slots__ = ("wrist_hist", "speed_buf", "last_event_t", "armed")
    def __init__(self):
        self.wrist_hist  = deque(maxlen=HISTORY_LEN)   # (t, x, y)
        self.speed_buf   = deque(maxlen=SPEED_BUFFER_LEN)  # recent speeds
        self.last_event_t = -1e9
        self.armed = True

class ShotClassifier:
    def __init__(self):
        self.players: Dict[int, _PlayerState] = {}

    def _state(self, pid: int) -> _PlayerState:
        if pid not in self.players:
            self.players[pid] = _PlayerState()
        return self.players[pid]

    def update(self, frame_idx: int, t_sec: float,
               poses: Dict[int, dict],
               racket_centers: Optional[List[tuple]] = None) -> List[ShotEvent]:
        events = []
        for pid, kps in poses.items():
            ev = self._update_one(frame_idx, t_sec, pid, kps, racket_centers or [])
            if ev is not None:
                events.append(ev)
        return events

    def _update_one(self, frame_idx, t_sec, pid, kps, racket_centers) -> Optional[ShotEvent]:
        side_kp    = "r" if ASSUME_RIGHT_HANDED else "l"
        wrist_k    = f"{side_kp}_wrist"
        shoulder_k = f"{side_kp}_shoulder"
        elbow_k    = f"{side_kp}_elbow"
        other_sh   = "l_shoulder" if side_kp == "r" else "r_shoulder"

        for k in (wrist_k, shoulder_k, elbow_k, other_sh):
            if k not in kps or kps[k][2] < VIS_THRESH:
                return None

        wx, wy, _   = kps[wrist_k]
        sx, sy, _   = kps[shoulder_k]
        ex, ey, _   = kps[elbow_k]
        osx, _, _   = kps[other_sh]

        body_cx     = 0.5 * (sx + osx)
        shoulder_w  = max(1.0, abs(sx - osx))
        torso_h_est = shoulder_w * 1.5

        st = self._state(pid)
        st.wrist_hist.append((t_sec, wx, wy))

        speed = 0.0
        if len(st.wrist_hist) >= 2:
            t0, x0, y0 = st.wrist_hist[0]
            t1, x1, y1 = st.wrist_hist[-1]
            dt   = max(1e-3, t1 - t0)
            dist = math.hypot(x1 - x0, y1 - y0)
            speed = (dist / shoulder_w) / dt
        st.speed_buf.append(speed)

        # Cooldown re-arm
        if not st.armed and (t_sec - st.last_event_t) >= COOLDOWN_SEC:
            st.armed = True
        if not st.armed:
            return None

        # --- Fix A: Peak detection on speed buffer ---
        if len(st.speed_buf) < SPEED_BUFFER_LEN:
            return None
        buf = list(st.speed_buf)
        peak_idx = SPEED_BUFFER_LEN // 2          # middle position
        peak_val = buf[peak_idx]
        if peak_val < WRIST_SPEED_TRIG:
            return None
        is_local_max = all(peak_val >= b for i, b in enumerate(buf) if i != peak_idx)
        if not is_local_max:
            return None

        # --- Fix B: Arm extension gate ---
        elbow_angle = _angle_deg((sx, sy), (ex, ey), (wx, wy))
        if elbow_angle < ELBOW_ANGLE_MIN:
            return None

        # --- Classification ---
        if (sy - wy) > SMASH_Y_RATIO * torso_h_est:
            shot_type, side = "Serve/Smash", "overhead"
        else:
            dx = wx - body_cx
            min_dx = SIDE_X_RATIO * shoulder_w
            if abs(dx) < min_dx:
                return None
            on_dominant_side = (dx > 0) if side_kp == "r" else (dx < 0)
            shot_type = "Forehand" if on_dominant_side else "Backhand"
            side      = "dominant" if on_dominant_side else "non-dominant"

        # --- Fix C: racket proximity for confidence flag (does NOT block) ---
        conf = "med"
        if peak_val > 1.5 * WRIST_SPEED_TRIG and elbow_angle > 150:
            conf = "high"
        for rcx, rcy in racket_centers:
            if math.hypot(rcx - wx, rcy - wy) < shoulder_w:
                conf = "high"
                break

        st.last_event_t = t_sec
        st.armed = False
        return ShotEvent(frame_idx, t_sec, pid, shot_type, side, conf)