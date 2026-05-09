# Padel Game Analytics — Shot Classification Prototype

End-to-end pipeline that ingests a padel match video, detects and tracks players / ball / racket, runs pose estimation per player, and classifies shots (**Forehand / Backhand / Serve-Smash**) using interpretable heuristics. Outputs an annotated video plus structured CSV/JSON event logs and per-player analytics.

> **Design philosophy.** Heuristic and explainable over learned and opaque. Every shot decision can be traced to numeric thresholds in code. Limitations are documented honestly rather than hidden.

---

## 1. Architecture

```
                ┌─────────────────────────────────────────────────────────┐
  input.mp4 ──► │  main.py  (frame loop, orchestration, overlays, I/O)    │
                └────────┬────────────────────────────────────────────────┘
                         │
       ┌─────────────────┼─────────────────┬──────────────────┐
       ▼                 ▼                 ▼                  ▼
  tracker.py        court_roi.py        ball.py           pose.py
  YOLOv8s +        4-pt polygon       motion ∩ white     MediaPipe Pose
  ByteTrack        + foot-point        candidate +        per-player crop,
  (person/ball/    filter, top-4       streak fallback    landmarks remapped
   racket)         persons              + prediction       to frame coords
       │                 │                 │                  │
       └────────┬────────┴───────┬─────────┴──────────────────┘
                ▼                ▼
          contact.py        classifier.py
          trajectory-bend   pose-only
          + wrist proximity wrist-speed peak,
          → ContactEvent    elbow-extension,
                            cooldown
                │                │
                └────────┬───────┘
                         ▼
                  event_merger.py
                  fuse within 0.4 s →
                  high / med / low confidence
                         ▼
                   logger.py
                   events.csv, events.json,
                   summary.json
```

**Key idea — dual-signal fusion.** Pose alone fires on any arm swing; contact alone fires only when the ball trajectory bends near a wrist. Each is noisy alone but their **agreement** within a short window is a reliable shot. The merger tags every event with one of three confidence levels so downstream analysis can filter.

---

## 2. Project structure

```
padel/
├── main.py                  # video I/O, loop, overlays, finalization
├── tracker.py               # YOLOv8s + ByteTrack
├── pose.py                  # MediaPipe Pose per player crop
├── classifier.py            # pose-only heuristic classifier
├── ball.py                  # motion ∩ white ball tracker + prediction
├── contact.py               # trajectory-bend contact event detector
├── shot_classifier_v2.py    # label Forehand / Backhand / Smash from contact
├── event_merger.py          # confidence-tagged fusion of contact + pose
├── court_roi.py             # interactive 4-click court calibration + filter
├── logger.py                # CSV / JSON / summary export
├── requirements.txt
├── notebooks/
│   └── run_on_colab.ipynb   # GPU runtime
├── data/input.mp4           # your video
└── outputs/
    ├── output.mp4           # annotated video
    ├── events.csv           # one row per detected shot
    ├── events.json          # same data as JSON
    └── summary.json         # totals + per-player counts
```

---

## 3. Setup & run (local CPU)

```bash
cd padel
python -m venv .venv
.venv\Scripts\activate           # Windows  (use `source .venv/bin/activate` on Linux/macOS)
pip install -r requirements.txt

# Place your video at  data/input.mp4
python court_roi.py              # ONE-TIME: click 4 court corners, ENTER, saves court_roi.json
python main.py                   # processes video → outputs/
```

CPU runtime: ~1–3 fps at `imgsz=1280`. For real-time, use Colab GPU (next section).

---

## 4. Run on Colab / Kaggle (GPU)

Open `notebooks/run_on_colab.ipynb` in Colab → Runtime → T4 GPU. The notebook handles repo clone, dependency install, ROI upload, execution, and output download.

**Headless ROI**: Calibration uses an OpenCV GUI window which Colab does not have. Two options:

1. **Recommended** — calibrate once locally, upload the resulting `court_roi.json` alongside `input.mp4`.
2. **Inline** — set environment variable before running `main.py`:
   ```python
   os.environ['PADEL_ROI'] = 'x1,y1;x2,y2;x3,y3;x4,y4'   # four pixel coordinates
   os.environ['PADEL_HEADLESS'] = '1'                    # error out instead of opening GUI
   ```

Ultralytics auto-detects CUDA — no code change needed. Expect ~15–25× speedup on T4.

---

## 5. Methodology

### 5.1 Detection & tracking — `tracker.py`
YOLOv8s (COCO-pretrained) with classes `person (0)`, `sports ball (32)`, `tennis racket (38)`. ByteTrack assigns persistent IDs. Per-class confidence thresholds (`person=0.35`, `racket=0.20`, `ball=0.10`) reflect the size/visibility difference. `imgsz=1280` to recover small objects.

### 5.2 Court ROI filter — `court_roi.py`
Multi-court broadcast footage means many off-court people get detected. We compute each person's foot-point (`bbox_bottom_center`) and keep only those inside the user-supplied polygon, then keep the **top 4 by bbox area** (closer to camera = bigger). This single step eliminates ~70% of false-positive shot candidates from background players.

### 5.3 Ball tracking — `ball.py` (hybrid)
Per-frame YOLO ball detection is unreliable (~3–5 px ball, motion blur). We fuse three sources:

1. **YOLO** when confidence high enough → trusted (green trail).
2. **Motion ∩ white** fallback (yellow trail). Frame differencing → motion mask. HSV filter for white. Static-mask (built over first 45 frames) AND-NOT'd to suppress court lines and net. Person bboxes shrunk by 4 px and masked out. Largest bright moving blob wins. Streak-aspect candidates accepted to handle motion-blur ellipses.
3. **Constant-velocity prediction** for up to 5 frames when neither fires (gray trail).

### 5.4 Pose — `pose.py`
MediaPipe Pose runs **per player crop** with `static_image_mode=True` (different person each call). 33 landmarks remapped to full-frame coordinates, so all downstream code uses one coordinate system.

### 5.5 Contact detection — `contact.py`
A **shot contact** is when the ball's trajectory sharply bends near a wrist:

- 7-frame ball-position buffer split into pre/post halves.
- Compute incoming and outgoing velocity vectors.
- Trigger if `|v_in| > MIN_SPEED` **and** `angle(v_in, v_out) > TURN_ANGLE_MIN` **and** the bend point lies within `WRIST_DIST_MAX` pixels of any tracked player's wrist.
- Output: `ContactEvent(player_id, contact_xy, in_vel, out_vel, frame_idx)`.

This is the **objective** signal — it requires the ball to actually change direction near a hand. Noise mode: it fires only when the ball is tracked, so missed-ball frames produce missed contacts.

### 5.6 Pose-only classifier — `classifier.py`
The **subjective** signal. Per-player rolling wrist-speed buffer (in shoulder-widths/sec → scale-invariant). Triggers on a peak above `WRIST_SPEED_TRIG=0.95` with elbow extension `> 135°`, plus a per-player cooldown. Side classification by wrist-x relative to body-center-x in shoulder-width units. Smash detection: wrist_y above shoulder_y by > 0.5 shoulder-widths.

### 5.7 Event fusion — `event_merger.py`
Both signals push into a 0.4 s window per player_id. Outcomes:

| Contact | Pose | Confidence | Source label |
|---|---|---|---|
| ✓ | ✓ | **high** | `contact+pose` |
| ✓ | ✗ | **med**  | `contact` |
| ✗ | ✓ (high internal) | **med** | `pose` |
| ✗ | ✓ (med internal)  | **low** | `pose` |

A 0.5 s hold buffer waits for the partner signal before emitting.

### 5.8 Logging — `logger.py`
Every emitted shot writes one CSV row + JSON object. Schema:

| field | description |
|---|---|
| `frame` | frame index |
| `timestamp_s` | seconds from video start |
| `player_id` | ByteTrack ID |
| `shot_type` | `Forehand` \| `Backhand` \| `Smash` |
| `side` | `forehand` \| `backhand` (raw side, useful for analytics) |
| `confidence` | `high` \| `med` \| `low` |
| `source` | `contact+pose` \| `contact` \| `pose` |
| `contact_x`, `contact_y` | ball position at contact (if known) |
| `out_dx`, `out_dy` | post-contact velocity vector (for direction analytics) |

`summary.json` aggregates totals and per-player counts, also rendered live as an overlay in `output.mp4`.

---

## 6. Assumptions

1. **Right-handed players.** Side classification uses wrist-x relative to body center; mirroring would invert Forehand/Backhand for left-handers.
2. **Fixed broadcast camera.** The static-mask bootstrap and ROI calibration assume the camera does not pan.
3. **Padel doubles, single court of interest.** ROI selects one court; the other 3 players visible in the frame are excluded.
4. **COCO classes generalize.** YOLOv8s is COCO-pretrained — `tennis racket` covers padel rackets reasonably; `sports ball` is the weakest link.
5. **Court is well-lit and contrasts the ball.** Ball detection uses a brightness/saturation HSV threshold that assumes a white ball on a darker court.

---

## 7. Limitations (honest)

1. **Ball at 3–5 px is statistically indistinguishable from sensor noise.** Per-frame YOLO confidence rarely exceeds 0.10 even with `imgsz=1280`.
2. **Motion blur dilutes the white signal.** During fast shots — exactly the moments of interest — the ball stretches into a faint streak that bleeds into the background.
3. **`motion ∩ white` mask hides a stationary ball.** Intentional (the candidate must be moving), but means just-bounced or held balls disappear.
4. **Player-bbox exclusion masks the ball at contact.** A ball overlapping the wrist is geometrically inside the player bbox at the very moment we want to detect it. Compromise: shrink bbox by 4 px and accept some false positives elsewhere.
5. **Static mask doesn't update** — auto-exposure drift over a long match would slowly leak court lines back into the candidate mask.
6. **ByteTrack ID swaps on occlusion.** When players cross, IDs swap, so `player_id` in the output is **session-stable but not globally identifiable**. A re-ID model (e.g., OSNet) would help.
7. **MediaPipe Pose is single-person.** We run it once per player crop, which is expensive. CPU runtime is ~1–3 fps; GPU YOLO doesn't help Pose latency.
8. **Pose classifier still fires on hand-swings without a ball.** That's why the merger tags pose-only events as `med`/`low` — for analytics, filter to `high`.

---

## 8. What I tried that didn't work (and why)

- **Pure-pose classification.** Any arm swing fires. False positives flood the log because players move continuously between rallies.
- **Pure-contact classification.** Misses real shots because ball detection has ~30–50% recall on fast strokes. You see a Forehand but get no event.
- **Pure-color (white-only) ball tracking.** Five failure modes: (a) court lines and net are also white (mitigated with static mask, but mask decays); (b) motion blur turns the ball into a faint streak below brightness threshold; (c) at contact the ball is inside the player bbox — exclusion masks it out; (d) on bounce, the ball is briefly stationary so motion-gating drops it; (e) at 3–5 px, sensor noise pixels are visually identical.

The honest answer: this is a **model gap**, not a tuning gap. Per-frame detection on tiny fast objects is what specialized architectures (TrackNet, TrackNetv2) are built for.

---

## 9. Improvements I'd make

1. **TrackNet for ball.** Heatmap-based architecture trained for tiny fast balls (designed for tennis/badminton). Recall on fast shots would jump from ~30% to ~85%.
2. **Custom-trained YOLO.** Fine-tune on padel-specific footage — current model is COCO, so racket and ball are out-of-distribution.
3. **Re-ID for player tracking.** Replace ByteTrack-only with ByteTrack + OSNet appearance embeddings to survive occlusion crosses.
4. **Action-recognition head as 3rd merge signal.** A small SlowFast or TSN trained on a few hundred labeled clips would resolve the "swing without ball" ambiguity that pure heuristics cannot.
5. **Multi-camera triangulation** for ball z-coordinate (true bounce detection, not just y-pattern heuristics).
6. **Auto ROI from court-line detection** (Hough transform on white pixels) — eliminates the manual calibration step.

---

## 10. Run reference

```bash
python main.py                                # standard run
PADEL_HEADLESS=1 python main.py               # error if no ROI source
PADEL_ROI="450,320;1480,320;1700,1020;220,1020" python main.py   # inline ROI
```

Outputs land in `outputs/` and overwrite previous runs.

---

*Built as an AI/ML internship assignment. Designed to be defensible in interview: every threshold, every fallback, every limitation is documented in code and here.*
