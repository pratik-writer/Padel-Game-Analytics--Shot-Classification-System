# Padel Game Analytics — Shot Classification Prototype

A simple, end-to-end pipeline that watches a padel match video and tries to figure out:
- Who the players are
- Where the ball is
- When a shot is taken
- What kind of shot it was (Forehand / Backhand / Serve-Smash)
- Roughly which direction the ball went and when it bounced

It then spits out an annotated video, a CSV/JSON log of every shot, a small summary file with per-player counts, and a one-page dashboard image.

The whole thing runs on rules and pre-trained models. No custom training, no fancy action-recognition network. Everything that decides "this is a shot" can be traced back to a number you can change in code.

---

## What the assignment asked for vs what's in this repo

| Asked for | Where it lives |
|---|---|
| Detect & track players, ball, racket | `tracker.py` (YOLOv8s + ByteTrack) |
| Pose estimation per player | `pose.py` (MediaPipe Pose, run on each player crop) |
| Shot classification (FH / BH / Serve-Smash) | `classifier.py` + `contact.py` + `event_merger.py` |
| Structured CSV / JSON output | `logger.py` → `outputs/events.csv`, `events.json` |
| Shot count analytics | `logger.py` → `outputs/summary.json` + live overlay in video |
| Visualization on video | `main.py` (skeletons, boxes, ball trail, bounce rings, counters) |
| Simple post-run dashboard | `dashboard.py` → `outputs/dashboard.png` |
| Shot direction (rule-based) | `logger.py::_direction_label` |
| Bounce detection (rule-based) | `bounce.py` |

---

## How to read this project in 30 seconds

The video is processed one frame at a time. For each frame:

```
read frame
  └── YOLOv8 finds people, balls, rackets
        └── keep only the 4 players inside the court polygon
              ├── MediaPipe draws a skeleton on each player
              ├── Ball tracker tries: YOLO → motion+white blob → predicted position
              └── Bounce detector watches the ball's y-velocity for a sign-flip

  Two independent shot detectors run side-by-side:
    A) Pose-only:    "did this player's wrist suddenly speed up with the elbow extending?"
    B) Contact-only: "did the ball trajectory bend sharply right next to a wrist?"

  Event merger:
    - both fired together → high-confidence shot
    - only one fired      → medium / low confidence

  Write everything to the output video + CSV + JSON
```

That's the whole idea. The rest of this README explains why each piece exists and what its weaknesses are.

---

## Pseudocode of `main.py`

```text
load YOLO model, MediaPipe Pose, court ROI
open input video, create output video writer

for each frame in video:

    detections = YOLO(frame)                       # persons, balls, rackets
    detections = filter_to_court(detections)       # keep top-4 players inside ROI
    ball       = ball_tracker.update(frame)        # YOLO → motion+white → predicted
    poses      = MediaPipe(crop) for each player   # 33 landmarks, mapped back to frame
    bounce     = bounce_detector.update(ball)      # y-velocity sign change?

    contact_event = contact_detector.update(ball, poses)   # trajectory bend near wrist?
    pose_event    = pose_classifier.update(poses)          # wrist speed peak?

    shot = event_merger.combine(contact_event, pose_event) # tag: high / med / low

    if shot:
        log it (frame, time, player, shot_type, direction, confidence)

    draw boxes, skeletons, ball trail, bounce rings, counters on frame
    write frame to output video

at the end:
    write events.csv, events.json, summary.json
    render dashboard.png from summary.json
```

If you only read one section of this README, this is it.

---

## File-by-file

```
padel/
├── main.py                  # the loop above; wires every module together
├── tracker.py               # YOLOv8s + ByteTrack wrapper (person / ball / racket)
├── pose.py                  # runs MediaPipe Pose per player crop, remaps to full frame
├── classifier.py            # pose-only shot detector (wrist speed + elbow angle)
├── ball.py                  # hybrid ball tracker: YOLO + motion∩white + prediction
├── contact.py               # ball-trajectory-bend contact detector
├── shot_classifier_v2.py    # labels Forehand/Backhand/Smash from a contact event
├── event_merger.py          # fuses pose + contact into a confidence-tagged shot
├── bounce.py                # rule-based bounce detector (ball y-velocity sign flip)
├── court_roi.py             # 4-click court calibration + foot-point filter
├── logger.py                # writes CSV / JSON / summary, computes direction label
├── dashboard.py             # post-run matplotlib summary PNG
├── requirements.txt
├── notebooks/
│   └── run_on_colab.ipynb   # GPU runtime (Colab / Kaggle)
├── data/input.mp4           # your video goes here
└── outputs/
    ├── output.mp4
    ├── events.csv
    ├── events.json
    ├── summary.json
    └── dashboard.png
```

---

## How to run it

### Locally (CPU)

```bash
cd padel
python -m venv .venv
.venv\Scripts\activate          # on Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt

# put your video at data/input.mp4
python court_roi.py             # ONE-TIME: click 4 court corners, press ENTER
python main.py                  # processes the video
```

CPU is slow — expect 1–3 fps at `imgsz=1280`. Fine for a one-off run, painful for tuning.

### On Colab / Kaggle (GPU)

Open `notebooks/run_on_colab.ipynb`, switch the runtime to T4 GPU, then run cells top to bottom. The notebook handles installing dependencies, uploading the video and the calibrated `court_roi.json`, running the pipeline, and zipping the outputs back.

If you can't run the GUI calibration (Colab has no display), you have two options:
1. Calibrate locally once, upload the resulting `court_roi.json`.
2. Set `PADEL_ROI="x1,y1;x2,y2;x3,y3;x4,y4"` as an environment variable before running.

---

## Output schema

Every detected shot becomes one row in `outputs/events.csv`:

| field | meaning |
|---|---|
| `frame` | frame index |
| `timestamp` | seconds from start of video |
| `player_id` | ByteTrack ID (stable within the run, not across runs) |
| `shot_type` | `Forehand` / `Backhand` / `Serve/Smash` |
| `side` | raw side classification before naming |
| `confidence` | `high` / `med` / `low` (from event merger) |
| `source` | which detector(s) fired: `contact+pose`, `contact`, or `pose` |
| `contact_x`, `contact_y` | ball position at contact, if known |
| `out_dx`, `out_dy` | post-contact velocity vector, if known |
| `direction` | `cross-left` / `cross-right` / `down-the-line` / `lob/up-court` / `deep-left` / etc. |

`summary.json` adds totals, per-player counts, breakdown by direction, and bounce count. `dashboard.png` visualises all of it.

---

## Assumptions

These are not guesses, they are choices baked into the code:

1. **The camera doesn't move.** Static-mask bootstrap and the fixed court ROI both assume this.
2. **One court of interest.** The video has multiple courts in view; we pick one.
3. **Players are right-handed.** Side classification (`wrist_x` vs body center) flips for left-handers.
4. **The ball is white-ish.** The motion+white fallback uses an HSV brightness/saturation filter.
5. **COCO classes generalise to padel.** YOLOv8s is COCO-pretrained — its `tennis racket` covers padel rackets, its `sports ball` is the weakest link.

---

## Challenges I ran into

This is the honest version. The polished version went into the architecture diagrams above; this section is the journey.

**Racket detection was bad with `yolov8n`.** The nano model just couldn't see the rackets reliably — they ended up inside the player boxes most of the time and never came out as separate detections. Switching to `yolov8s` and bumping `imgsz` to 1280 fixed it. Cost was speed, but accuracy mattered more here.

**Off-court people kept getting tracked.** The video is wide enough to show neighbouring courts and bystanders. YOLO happily detected all of them, MediaPipe drew skeletons on all of them, and the shot classifier counted their arm waves as shots. The fix was a manual court calibration step — `court_roi.py` opens the first frame, you click the four corners of the court that matters, and from then on we filter by foot-point inside that polygon. Then we keep only the top 4 by bbox area (closer to camera = bigger). This single change cleaned up most of the noise.

**Ball tracking was the hardest part of this whole project, and it's still the weakest link.** YOLO's `sports ball` class barely sees a 3–5 pixel padel ball — confidence is below 0.10 most of the time, and motion blur during fast shots makes it almost invisible. I tried to treat it as a pure colour-tracking problem: the ball is white, the court is mostly green/blue, so look for white blobs. That works when the ball is sitting still on the floor, but during actual play the motion blur and the small size mean the white pixels get diluted into the background. Court lines, the top of the net, and even players' white pants would also light up the white mask. I added a static-mask bootstrap to subtract the always-white pixels (lines, net), a motion mask to ignore stationary stuff, and a streak-detection branch to catch motion-blurred ellipses. It's better than YOLO alone, but ball recall on fast shots is still maybe 30–50%. Honest verdict: per-frame detection of a tiny fast object is what specialised architectures like TrackNet are built for. A pre-trained generic detector is the wrong tool, but I worked within the constraint.

**Shot classification went through three painful iterations.**

- **First attempt:** pure pose-based wrist-speed peak. Easy to write, easy to over-fire. Players walking, swinging arms between rallies, adjusting their grip — all classified as shots. The log was flooded.
- **Second attempt:** add a ball-contact detector — only count a shot if the ball trajectory bent near a wrist. This is the *correct* signal, but it depends on the ball being tracked, which we already established is unreliable. So now real shots were being missed because the ball wasn't seen at the contact frame.
- **Third attempt (current):** run both detectors in parallel and merge them with a confidence tag. If both fire within 0.4s for the same player → `high` confidence shot. Only one fires → `med` or `low`. This way nothing gets thrown away and downstream code (or you, when reading the CSV) can pick the precision/recall tradeoff. The pose thresholds were also tightened (wrist-speed in shoulder-widths/sec to be scale-invariant, plus an elbow-extension gate of 135°) to cut down the worst pose false positives without going so strict that it missed real shots.

**Tuning the classifier felt like whack-a-mole.** Every time I made it stricter to kill false positives, it started missing real shots. Every time I loosened it to catch more shots, the false positives came back. I eventually accepted that pure heuristics have a ceiling here, documented it, and moved on. The right next step is a small action-recognition model trained on a few hundred labeled clips — that's the only thing that resolves "swing without ball".

**Player IDs sometimes swap on occlusion.** ByteTrack is good but not perfect — when two players cross, IDs can switch. So `player_id=2` in the first half might be a different person in the second half. A re-ID model on top would fix it; I noted it as a limitation.

**MediaPipe runs per crop, which is slow.** MediaPipe Pose is single-person, so I run it once per player bounding box (with `static_image_mode=True`, otherwise the temporal smoothing leaks landmarks between players). On CPU this dominates the per-frame cost. GPU helps YOLO but not MediaPipe.

**Calibration UX doesn't work in Colab.** OpenCV GUI windows need a display. I added a headless mode (`PADEL_ROI` env var, or just upload the JSON) so the same script works on a laptop and on Colab.

**No custom training was done.** Everything uses pre-trained weights. Training a small YOLO on padel-specific footage (ball + racket especially) would almost certainly close most of the remaining accuracy gap. I planned for this but didn't have the data or time within scope.

---

## Limitations (things you should know before trusting the output)

1. **Ball recall on fast shots is moderate.** Tiny + blurry = hard. Expect to miss some.
2. **Static white mask doesn't update.** Long matches with auto-exposure drift would slowly leak court lines back into the ball candidate mask.
3. **Player-bbox masking can hide the ball at contact.** A ball overlapping a wrist is geometrically inside the bbox at the exact moment we want to detect it. I shrink the bbox by 4 pixels as a compromise.
4. **Pose-only events still over-fire.** That's why they're tagged `med`/`low` — for clean analytics, filter to `confidence == "high"`.
5. **Right-handedness assumption.** Side labels invert for left-handed players.
6. **2D bounce detection.** A wall bounce and a floor bounce both look like a y-velocity flip; I can't tell them apart without depth.
7. **One court only.** The ROI is fixed; multi-court analysis would need one ROI per court.
8. **Player IDs can swap on occlusion.** Per-run stable, not globally identifiable.

---

## What I'd build next, given more time

1. **TrackNet for ball.** Heatmap-based, designed exactly for tiny fast balls in racket sports. This is the single biggest accuracy win available.
2. **Fine-tune YOLO on padel footage.** Even a few hundred annotated frames would massively improve racket and ball detection.
3. **Re-ID for stable player tracking** (e.g., OSNet on top of ByteTrack).
4. **Small action-recognition head** (TSN or SlowFast) trained on a few hundred labeled swing clips. Use it as a third merge signal alongside pose and contact. This is the cleanest fix for "swing without ball".
5. **Auto court detection** via Hough-line transform on white pixels — kills the manual calibration step.
6. **Multi-camera triangulation** for true 3D ball tracking and proper bounce-vs-wall disambiguation.

---

## Demo videos & artifacts

The annotated output video and input sample are too large for git. Drive links:

- **Input video:** `<paste-drive-link-here>`
- **Annotated output video:** `<paste-drive-link-here>`
- **`outputs/` zip (events.csv, events.json, summary.json, dashboard.png):** `<paste-drive-link-here>`

To reproduce locally: drop `input.mp4` into `padel/data/`, run `python court_roi.py`, then `python main.py`.

---

## Run reference

```bash
python main.py                                                # standard run (also writes dashboard.png)
python dashboard.py                                           # re-render dashboard from existing summary.json
PADEL_HEADLESS=1 python main.py                               # error out if no ROI source available
PADEL_ROI="450,320;1480,320;1700,1020;220,1020" python main.py  # provide ROI inline
```

Outputs land in `outputs/` and are overwritten each run.

---

*Built as an AI/ML internship assignment. Designed to be defensible in interview — every threshold, every fallback, every limitation is documented in the code and here. Nothing is hidden behind a learned black box.*
