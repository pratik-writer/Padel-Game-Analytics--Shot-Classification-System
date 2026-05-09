from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

MERGE_WINDOW = 0.4      # s; two signals within this gap are treated as the same shot
HOLD_BUFFER  = 0.5      # s; emit events after this delay so the partner signal can arrive


@dataclass
class MergedShot:
    frame: int
    timestamp: float
    player_id: int
    shot_type: str        # 'Forehand' | 'Backhand' | 'Serve/Smash'
    side: str
    confidence: str       # 'high' | 'med' | 'low'
    source: str           # 'contact+pose' | 'contact' | 'pose'
    contact_xy: Optional[Tuple[float, float]] = None
    out_dir:    Optional[Tuple[float, float]] = None


class EventMerger:
    def __init__(self):
        # pending item: (deadline_t, kind, shot_obj, player_id)
        self._pending: deque = deque()

    def push(self, t_sec: float,
             contact_shot=None,
             pose_shot=None) -> List[MergedShot]:
        if contact_shot is not None:
            self._pending.append((t_sec + HOLD_BUFFER, "contact",
                                  contact_shot, contact_shot.player_id))
        if pose_shot is not None:
            self._pending.append((t_sec + HOLD_BUFFER, "pose",
                                  pose_shot, pose_shot.player_id))
        return self._flush_ready(t_sec)

    def flush(self) -> List[MergedShot]:
        return self._flush_ready(t_sec=float("inf"))

    def _flush_ready(self, t_sec: float) -> List[MergedShot]:
        emitted: List[MergedShot] = []
        ready, kept = [], deque()
        for item in self._pending:
            (ready if item[0] <= t_sec else kept).append(item)
        self._pending = kept

        by_pid: dict = {}
        for it in ready:
            by_pid.setdefault(it[3], []).append(it)

        for pid, items in by_pid.items():
            items.sort(key=lambda x: x[2].timestamp)
            used = [False] * len(items)
            for i, (_, kind_i, shot_i, _) in enumerate(items):
                if used[i]:
                    continue
                partner_idx = None
                for j in range(i + 1, len(items)):
                    if used[j]:
                        continue
                    _, kind_j, shot_j, _ = items[j]
                    if kind_j == kind_i:
                        continue
                    if abs(shot_j.timestamp - shot_i.timestamp) <= MERGE_WINDOW:
                        partner_idx = j
                        break
                if partner_idx is not None:
                    used[i] = True
                    used[partner_idx] = True
                    contact_shot = shot_i if kind_i == "contact" else items[partner_idx][2]
                    pose_shot    = shot_i if kind_i == "pose"    else items[partner_idx][2]
                    emitted.append(self._merge_pair(contact_shot, pose_shot))
                else:
                    used[i] = True
                    emitted.append(self._wrap_solo(kind_i, shot_i))
        emitted.sort(key=lambda e: e.timestamp)
        return emitted

    def _merge_pair(self, contact_shot, pose_shot) -> MergedShot:
        shot_type = contact_shot.shot_type or pose_shot.shot_type
        side      = contact_shot.side      or pose_shot.side
        return MergedShot(
            frame=contact_shot.frame,
            timestamp=contact_shot.timestamp,
            player_id=contact_shot.player_id,
            shot_type=shot_type,
            side=side,
            confidence="high",
            source="contact+pose",
            contact_xy=getattr(contact_shot, "contact_xy", None),
            out_dir=getattr(contact_shot, "out_dir", None),
        )

    def _wrap_solo(self, kind: str, shot) -> MergedShot:
        if kind == "contact":
            confidence = "med" if getattr(shot, "confidence", "med") != "low" else "low"
            return MergedShot(
                frame=shot.frame,
                timestamp=shot.timestamp,
                player_id=shot.player_id,
                shot_type=shot.shot_type,
                side=shot.side,
                confidence=confidence,
                source="contact",
                contact_xy=getattr(shot, "contact_xy", None),
                out_dir=getattr(shot, "out_dir", None),
            )
        # pose-only events are demoted: high -> med, med -> low
        pose_conf = getattr(shot, "confidence", "med")
        confidence = "med" if pose_conf == "high" else "low"
        return MergedShot(
            frame=shot.frame,
            timestamp=shot.timestamp,
            player_id=shot.player_id,
            shot_type=shot.shot_type,
            side=shot.side,
            confidence=confidence,
            source="pose",
        )
