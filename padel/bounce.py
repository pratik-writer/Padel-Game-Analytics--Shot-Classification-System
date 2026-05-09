from collections import deque
from dataclasses import dataclass
from typing import Optional, List

WINDOW          = 5        # must be odd
MIN_DOWN_SPEED  = 6.0      # px/frame downward speed before bounce
MIN_UP_SPEED    = 3.0      # px/frame upward speed after bounce
COOLDOWN_FRAMES = 8


@dataclass
class BounceEvent:
    frame: int
    timestamp: float
    x: float
    y: float


class BounceDetector:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.buf: deque = deque(maxlen=WINDOW)
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
        pre  = [self.buf[i+1][2] - self.buf[i][2] for i in range(mid)]
        post = [self.buf[i+1][2] - self.buf[i][2] for i in range(mid, WINDOW - 1)]
        if not pre or not post:
            return None
        vy_pre  = sum(pre)  / len(pre)
        vy_post = sum(post) / len(post)

        is_bounce = (
            vy_pre  >  MIN_DOWN_SPEED and
            vy_post < -MIN_UP_SPEED   and
            (self.buf[mid][0] - self.last_bounce_frame) >= COOLDOWN_FRAMES
        )
        if not is_bounce:
            return None

        f, x, y = self.buf[mid]
        ev = BounceEvent(frame=f, timestamp=f / self.fps, x=x, y=y)
        self.events.append(ev)
        self.last_bounce_frame = f
        return ev
