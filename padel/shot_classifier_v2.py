from dataclasses import dataclass
from typing import Optional

ASSUME_RIGHT_HANDED = True
SMASH_Y_RATIO       = 0.20    # wrist must be this fraction of torso above shoulder
SIDE_X_MIN_RATIO    = 0.10    # min |contact_x - body_cx| in shoulder-widths
VIS_THRESH          = 0.3


@dataclass
class ShotEvent:
    frame: int
    timestamp: float
    player_id: int
    shot_type: str        # 'Forehand' | 'Backhand' | 'Serve/Smash'
    side: str             # 'dominant' | 'non-dominant' | 'overhead'
    confidence: str
    contact_xy: tuple
    out_dir: tuple        # outgoing ball direction


def classify(contact_event, poses) -> Optional[ShotEvent]:
    pid = contact_event.player_id
    kps = poses.get(pid)
    if kps is None:
        return None

    side_kp    = contact_event.wrist_side
    wrist_k    = f"{side_kp}_wrist"
    shoulder_k = f"{side_kp}_shoulder"
    other_sh   = "l_shoulder" if side_kp == "r" else "r_shoulder"

    for k in (wrist_k, shoulder_k, other_sh):
        if k not in kps or kps[k][2] < VIS_THRESH:
            return None

    wx, wy, _   = kps[wrist_k]
    sx, sy, _   = kps[shoulder_k]
    osx, osy, _ = kps[other_sh]

    body_cx     = 0.5 * (sx + osx)
    shoulder_w  = max(1.0, abs(sx - osx))
    torso_h_est = shoulder_w * 1.5
    cx, cy      = contact_event.contact_xy

    # image y grows downward, so smaller wy means higher position
    if (sy - wy) > SMASH_Y_RATIO * torso_h_est:
        return ShotEvent(
            frame=contact_event.frame,
            timestamp=contact_event.timestamp,
            player_id=pid,
            shot_type="Serve/Smash",
            side="overhead",
            confidence=contact_event.confidence,
            contact_xy=contact_event.contact_xy,
            out_dir=contact_event.out_dir,
        )

    dx = cx - body_cx
    if abs(dx) < SIDE_X_MIN_RATIO * shoulder_w:
        return None

    on_dominant_side = (dx > 0) if ASSUME_RIGHT_HANDED else (dx < 0)
    shot_type = "Forehand" if on_dominant_side else "Backhand"
    side      = "dominant" if on_dominant_side else "non-dominant"

    return ShotEvent(
        frame=contact_event.frame,
        timestamp=contact_event.timestamp,
        player_id=pid,
        shot_type=shot_type,
        side=side,
        confidence=contact_event.confidence,
        contact_xy=contact_event.contact_xy,
        out_dir=contact_event.out_dir,
    )
