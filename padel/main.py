import cv2
import os
from pathlib import Path
from tracker import Tracker
from pose import PoseEstimator, POSE_CONNECTIONS
from classifier import ShotClassifier
from court_roi import load_or_calibrate, filter_tracks_in_court, draw_court
from ball import BallTracker, draw_ball
from contact import ContactDetector
from shot_classifier_v2 import classify as classify_shot
from event_merger import EventMerger
from logger import EventLogger, player_counts

INPUT_PATH  = "data/input.mp4"
OUTPUT_PATH = "outputs/output.mp4"

COLOR_BY_NAME = {
    "person": (0, 255, 0),
    "ball":   (0, 165, 255),
    "racket": (255, 0, 255),
}
SKELETON_COLOR = (255, 255, 0)
KEYPOINT_COLOR = (0, 255, 255)
VIS_THRESH = 0.3

def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = COLOR_BY_NAME.get(t["name"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{t["name"]} #{t["id"]} {t["conf"]:.2f}',
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_poses(frame, poses):
    for pid, kps in poses.items():
        for a, b in POSE_CONNECTIONS:
            if a in kps and b in kps and kps[a][2] > VIS_THRESH and kps[b][2] > VIS_THRESH:
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, pa, pb, SKELETON_COLOR, 2)
        for name, (x, y, v) in kps.items():
            if v > VIS_THRESH:
                cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1)

def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"Place your padel video at {INPUT_PATH}")

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {INPUT_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} @ {fps:.2f} fps, {total} frames")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))

    tracker  = Tracker(weights="yolov8s.pt", base_conf=0.10, imgsz=1280)
    poser    = PoseEstimator(model_complexity=1)
    pose_classifier = ShotClassifier()
    court_polygon = load_or_calibrate(INPUT_PATH)
    print(f"[ROI] using polygon: {court_polygon.tolist()}")
    ball_tracker = BallTracker()
    contact_det  = ContactDetector()
    merger       = EventMerger()
    event_logger = EventLogger(out_dir="outputs")
    events_log = []
    last_event_text = ""
    last_event_until = -1.0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_sec  = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            tracks = tracker.update(frame)
            tracks = filter_tracks_in_court(tracks, court_polygon, max_persons=4)
            ball_state = ball_tracker.update(frame, t_sec, tracks, court_polygon)
            poses  = poser.estimate_for_tracks(frame, tracks)

            new_events = []
            contact = contact_det.update(frame_idx, t_sec, ball_state, poses)
            contact_shot = classify_shot(contact, poses) if contact is not None else None

            pose_shots = pose_classifier.update(
                frame_idx, t_sec, poses,
                racket_centers=[(0.5*(t["bbox"][0]+t["bbox"][2]),
                                 0.5*(t["bbox"][1]+t["bbox"][3]))
                                for t in tracks if t["name"] == "racket"],
            )
            pose_shot = pose_shots[0] if pose_shots else None

            new_events = merger.push(t_sec, contact_shot=contact_shot, pose_shot=pose_shot)

            for ev in new_events:
                events_log.append(ev)
                event_logger.add(ev)
                last_event_text = f"P{ev.player_id}: {ev.shot_type} [{ev.confidence}]"
                last_event_until = t_sec + 1.5
                print(f"[SHOT] t={ev.timestamp:6.2f}s  P{ev.player_id}  {ev.shot_type}  conf={ev.confidence}  src={ev.source}")

            draw_tracks(frame, tracks)
            draw_poses(frame, poses)
            draw_ball(frame, ball_state)
            draw_court(frame, court_polygon)

            cv2.putText(frame,
                        f"frame={frame_idx} t={t_sec:6.2f}s dets={len(tracks)} poses={len(poses)} ball={ball_state['source']}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if t_sec <= last_event_until and last_event_text:
                cv2.putText(frame, last_event_text, (15, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

            # ---- Per-player counters (top-right) ----
            counts = player_counts(events_log)
            y = 35
            cv2.putText(frame, "Shot counts (FH / BH / SS)",
                        (width - 380, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            y += 25
            for pid in sorted(counts.keys()):
                c = counts[pid]
                line = (f"P{pid}: {c.get('Forehand',0)} / "
                        f"{c.get('Backhand',0)} / {c.get('Serve/Smash',0)}")
                cv2.putText(frame, line, (width - 380, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 22
            total = len(events_log)
            cv2.putText(frame, f"Total shots: {total}",
                        (width - 380, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 255), 2)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[..] frame {frame_idx}/{total}  tracks={len(tracks)}  poses={len(poses)}")
    finally:
        cap.release()
        out.release()
        poser.close()
    # Drain any pending merger events that arrived in the last 0.5s
    for ev in merger.flush():
        events_log.append(ev)
        event_logger.add(ev)
        print(f"[SHOT] t={ev.timestamp:6.2f}s  P{ev.player_id}  {ev.shot_type}  conf={ev.confidence}  src={ev.source}")

    summary = event_logger.export()
    print(f"[SUMMARY] {summary['total_events']} total events "
          f"({summary['trusted_events']} trusted)")
    print(f"  by type:        {summary['by_shot_type']}")
    print(f"  by confidence:  {summary['by_confidence']}")
    print(f"  by source:      {summary['by_source']}")
    print(f"  per player:     {summary['per_player']}")
    print(f"[DONE] wrote {frame_idx} frames -> {OUTPUT_PATH}")
    print(f"[DONE] events.csv / events.json / summary.json -> outputs/")

if __name__ == "__main__":
    main()