"""
Simple rule-based bounce detector.

Heuristic: a bounce is the frame where the ball's *y-velocity* changes sign
from + (going down, since image-y increases downward) to - (going up), and the
speed magnitude before the change exceeds a threshold (filters noise).

Inputs : per-frame ball position (x, y) — None if missing.
Outputs: BounceEvent(frame, timestamp, position) when detected.

Limitations:
- Requires the ball to be tracked across the bounce frames. With our
  motion-gated tracker the ball can be briefly invisible right at the bounce
  (the contact moment is near-stationary), so recall is moderate.
- 2D only — we cannot distinguish a true floor bounce from a wall bounce or a
  shot. Shot contacts are also y-sign changes; we exclude them by ignoring
  bounces within 0.25s of a registered shot in main.py if needed.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional, List


# Tunables ------------------------------------------------------------
WINDOW          = 5      # frames buffered (must be odd)
MIN_DOWN_SPEED  = 6.0    # px/frame downward speed before bounce
MIN_UP_SPEED    = 3.0    # px/frame upward speed after bounce
COOLDOWN_FRAMES = 8      # min gap between two bounces


@dataclass
class BounceEvent:
    frame: int
    timestamp: float
    x: float
    y: float


class BounceDetector:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.buf: deque = deque(maxlen=WINDOW)   # (frame, x, y)
        self.last_bounce_frame: int = -10**9
        self.events: List[BounceEvent] = []

    def update(self, frame_idx: int, ball_xy: Optional[tuple]) -> Optional[BounceEvent]:
        if ball_xy is None:
            self.buf.clear()
            return None

        self.buf.append((frame_idx, float(ball_xy[0]), float(ball_xy[1])))
        if len(self.buf) < WINDOW:
            return None

        mid = WINDOW // 2
        # average vy in first half (pre) and second half (post), in px/frame
        pre  = [self.buf[i+1][2] - self.buf[i][2] for i in range(mid)]
        post = [self.buf[i+1][2] - self.buf[i][2] for i in range(mid, WINDOW - 1)]
        if not pre or not post:
            return None
        vy_pre  = sum(pre)  / len(pre)
        vy_post = sum(post) / len(post)

        is_bounce = (
            vy_pre  >  MIN_DOWN_SPEED and       # was going down fast
            vy_post < -MIN_UP_SPEED   and       # now going up
            (self.buf[mid][0] - self.last_bounce_frame) >= COOLDOWN_FRAMES
        )
        if not is_bounce:
            return None

        f, x, y = self.buf[mid]
        ev = BounceEvent(frame=f, timestamp=f / self.fps, x=x, y=y)
        self.events.append(ev)
        self.last_bounce_frame = f
        return ev
