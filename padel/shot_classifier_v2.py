"""
Shot classifier v2 — runs ONLY on contact events from contact.py.
Uses ball contact position relative to player body + wrist height to classify.
"""
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
    confidence: str       # inherited from contact event
    contact_xy: tuple
    out_dir: tuple        # outgoing ball direction (for direction analytics later)


def classify(contact_event, poses) -> Optional[ShotEvent]:
    """
    contact_event : ContactEvent (from contact.py)
    poses         : {player_id: {kp_name: (x,y,vis)}} at contact frame

    Returns ShotEvent or None (if pose missing for the contacted player).
    """
    pid = contact_event.player_id
    kps = poses.get(pid)
    if kps is None:
        return None

    # Need both shoulders + wrist on the side where contact happened
    side_kp    = contact_event.wrist_side    # 'r' or 'l'
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

    # --- Serve/Smash: wrist clearly above shoulder at contact ---
    # (image y grows downward, so smaller wy = higher position)
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

    # --- Forehand vs Backhand: contact x relative to body center ---
    dx = cx - body_cx
    min_dx = SIDE_X_MIN_RATIO * shoulder_w
    if abs(dx) < min_dx:
        # Contact too close to body center — ambiguous; skip (rare)
        return None

    # For right-handed player: contact RIGHT of body center → Forehand
    # (For left-handed it would be inverted; documented assumption)
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
