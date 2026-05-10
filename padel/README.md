# Padel Shot Classifier

A small pipeline that watches a padel match video and figures out who the players are, where the ball is, when a shot is taken, and what kind of shot it was (Forehand, Backhand or Serve/Smash). It writes an annotated video, a CSV and JSON log of every shot, a summary file with per player counts, and a one page dashboard image.

Everything runs on rules and pre trained models. No custom training was done.

## What it can do

- Detect and track the four players, the ball and the rackets across frames.
- Filter out off court people and neighbouring courts using a manual court polygon.
- Run pose estimation (33 landmarks) on each tracked player.
- Track the ball with a small CNN (TrackNet) plus a Kalman filter for smoothing.
- Classify each detected shot as Forehand, Backhand or Serve/Smash.
- Tag every shot with a confidence level (high / med / low) based on how many detectors agreed.
- Estimate a coarse shot direction (cross-left, cross-right, down-the-line, lob, etc.) from the post contact ball trajectory.
- Detect ball bounces using a sign change in the ball y velocity.
- Produce a CSV and JSON event log with one row per shot.
- Produce a summary JSON with totals, per player counts, direction breakdown and bounce count.
- Render a post run dashboard PNG with four charts.
- Overlay live counters, skeletons, ball trail and bounce markers on the output video.

## What it covers

| Task | File |
|---|---|
| Detect and track players, ball, racket | `tracker.py` (YOLOv8s + ByteTrack) |
| Pose estimation per player | `pose.py` (MediaPipe Pose, run on each player crop) |
| Ball detection and tracking | `ball.py` (TrackNet + Kalman), `tracknet.py` (model) |
| Shot classification | `classifier.py`, `contact.py`, `event_merger.py` |
| CSV / JSON output | `logger.py` |
| Shot count analytics | `summary.json` plus a live overlay on the video |
| Visual overlays | `main.py` (skeletons, boxes, ball trail, bounce rings, counters) |
| Post run dashboard | `dashboard.py` |
| Shot direction (rule based) | `logger.py::_direction_label` |
| Bounce detection (rule based) | `bounce.py` |

## How it works

The video is processed one frame at a time. For each frame:

```
read frame
  YOLOv8 finds people, rackets
  keep only the 4 players inside the court polygon
  MediaPipe draws a skeleton on each player
  TrackNet looks at the last 3 frames and produces a heatmap of the ball
  Kalman filter smooths the ball position and fills 1 to 3 frame gaps
  bounce detector watches the ball y velocity for a sign flip

  two independent shot detectors run side by side:
    A) pose only:    "did this player's wrist suddenly speed up with the elbow extending?"
    B) contact only: "did the ball trajectory bend sharply right next to a wrist?"

  event merger:
    both fired together  -> high confidence
    only one fired       -> medium or low confidence

  write everything to the output video, CSV and JSON
```

## Pseudocode

```text
load YOLO model, MediaPipe Pose, TrackNet, court ROI
open input video, create output video writer

for each frame in video:

    detections = YOLO(frame)                       # persons, rackets
    detections = filter_to_court(detections)       # keep top 4 players inside ROI
    ball       = ball_tracker.update(frame)        # TrackNet detector + Kalman smoother
    poses      = MediaPipe(crop) for each player   # 33 landmarks, mapped back to frame
    bounce     = bounce_detector.update(ball)      # y velocity sign change?

    contact_event = contact_detector.update(ball, poses)
    pose_event    = pose_classifier.update(poses)

    shot = event_merger.combine(contact_event, pose_event)

    if shot:
        log it (frame, time, player, shot_type, direction, confidence)

    draw boxes, skeletons, ball trail, bounce rings, counters on frame
    write frame to output video

at the end:
    write events.csv, events.json, summary.json
    render dashboard.png from summary.json
```

## Files

```
padel/
  main.py                  # the loop above, wires every module together
  tracker.py               # YOLOv8s + ByteTrack wrapper (person, racket)
  pose.py                  # runs MediaPipe Pose per player crop, remaps to full frame
  classifier.py            # pose only shot detector (wrist speed + elbow angle)
  ball.py                  # ball tracker: TrackNet detector + Kalman filter, with auto fallback
  ball_legacy.py           # earlier ball tracker (motion + white blob), kept as fallback
  tracknet.py              # TrackNet CNN model (heatmap based, takes 3 stacked frames)
  contact.py               # ball trajectory bend contact detector
  shot_classifier_v2.py    # labels Forehand/Backhand/Smash from a contact event
  event_merger.py          # fuses pose + contact into a confidence tagged shot
  bounce.py                # rule based bounce detector (ball y velocity sign flip)
  court_roi.py             # 4 click court calibration + foot point filter
  logger.py                # writes CSV / JSON / summary, computes direction label
  dashboard.py             # post run matplotlib summary PNG
  requirements.txt
  challenges.txt
  data/                    # input.mp4 goes here at runtime (not committed, file is large)
  outputs/                 # generated at runtime (not committed): output.mp4, events.csv,
                           # events.json, summary.json, dashboard.png
```

The `data/` and `outputs/` folders are referenced by the code but their contents are not in the repo. The TrackNet weights file is also not in the repo because of size (43 MB); it gets downloaded into a fresh `models/` folder during setup. Drive links to the input and output videos are at the bottom of this README.

## Setup

```bash
cd padel
python -m venv .venv
.venv\Scripts\activate           # on Windows. use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

The TrackNet weights file is not in the repo (43 MB). The command below creates a `models/` folder and downloads the weights into it. Run it once before the first run:

```bash
pip install gdown
python -c "import os, gdown; os.makedirs('models', exist_ok=True); gdown.download('https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl', 'models/tracknet_weights.pt', quiet=False)"
```

If the weights are missing or torch fails to load, `ball.py` quietly falls back to the older motion + white blob tracker (`ball_legacy.py`) and prints a log line saying so. The pipeline keeps running either way.

## Run

```bash
# put your video at data/input.mp4
python court_roi.py              # one time: click 4 court corners, press ENTER
python main.py                   # processes the video
```

CPU is slow with TrackNet in the mix. Expect roughly 0.3 to 0.5 fps end to end at `imgsz=1280`. A five minute clip at 30 fps takes a couple of hours on CPU. With a CUDA GPU it drops to a few minutes. Notes on running on a GPU machine are at the bottom.

## Outputs

Every detected shot becomes one row in `outputs/events.csv`:

| field | meaning |
|---|---|
| `frame` | frame index |
| `timestamp` | seconds from start of video |
| `player_id` | ByteTrack ID (stable within the run, not across runs) |
| `shot_type` | `Forehand`, `Backhand` or `Serve/Smash` |
| `side` | raw side classification before naming |
| `confidence` | `high`, `med` or `low` (from event merger) |
| `source` | which detector(s) fired: `contact+pose`, `contact`, or `pose` |
| `contact_x`, `contact_y` | ball position at contact, if known |
| `out_dx`, `out_dy` | post contact velocity vector, if known |
| `direction` | `cross-left`, `cross-right`, `down-the-line`, `lob/up-court`, `deep-left`, etc. |

`summary.json` adds totals, per player counts, breakdown by direction, and bounce count. `dashboard.png` plots all of it.

## Assumptions

1. The camera does not move. The fixed court ROI relies on this.
2. One court of interest. The video has multiple courts in view; we pick one.
3. Players are right handed. Side classification flips for left handers.
4. The ball is small and fast. TrackNet does the primary detection. A court polygon and a person bounding box filter sit on top to reject false positives on court lines, shoes and white clothing.
5. COCO classes generalise to padel. YOLOv8s is COCO pre trained, so its `tennis racket` class covers padel rackets.

## Challenges

- **Racket detection with `yolov8n` was poor.** Rackets were rarely separated from the player boxes. Switching to `yolov8s` and bumping `imgsz` to 1280 fixed it. Cost was speed, but accuracy mattered more here.

- **Off court people kept getting tracked.** The video shows neighbouring courts and bystanders, all of whom got detections, skeletons, and false shot counts. Solved by adding `court_roi.py`, where I click the four corners of the main court once and then filter detections by foot point inside that polygon. Top 4 players by bounding box area are kept.

- **Ball is tiny and almost invisible during fast shots.** YOLO's `sports ball` confidence stays below 0.10 most of the time on a 3 to 5 pixel ball. This was the hardest single problem.

- **Pure colour tracking for the ball didn't work either.** Tried treating the white ball as a colour tracking problem, since the rest of the court is mostly green or blue. It works when the ball is sitting still, but motion blur and the small size dilute the white pixels into the background during actual play. Court lines, the top of the net, and white pants on players also lit up in the white mask.

- **Ball tracker became a fallback chain, then got replaced.** The original `ball.py` tried YOLO first, then a motion + white blob detector with a static mask to subtract court lines, then a constant velocity prediction for short gaps. Better than YOLO alone, but recall on fast shots was still around 30 to 50 percent. That version is kept as `ball_legacy.py`. The current `ball.py` swaps the detector for TrackNet, a small heatmap CNN that takes three stacked frames as input and was purpose built for tracking tiny fast balls in racket sports. A 4 state Kalman filter (x, y, vx, vy) then smooths the position and fills 1 to 3 frame gaps. Recall is meaningfully better and the trajectory is much less jittery, which also helps the bounce detector.

- **TrackNet weights are tennis trained, not padel trained.** Padel balls are similar enough in size and contrast that it generalises, but it is not perfect. Fine tuning on a few hundred padel frames would close the gap. The weights file is not in the repo because it is 43 MB, so the setup instructions include a one line download.

- **First shot classifier (pose only) over fired badly.** Wrist speed peaks fired on walking, casual hand movements, grip adjustments, anything. The log was flooded with false positives.

- **Second attempt (contact only) under fired.** Requiring a ball trajectory bend near a wrist is the right signal, but it depends on the ball being tracked, which it often was not. So real shots got missed. With TrackNet the contact detector now fires more often, which directly raises the count of `high` confidence (contact+pose) shots.

- **Final classifier is a hybrid with confidence tags.** Both detectors run in parallel and `event_merger.py` fuses them. Both fire within 0.4 seconds for the same player gives a `high` confidence shot. Only one fires gives `med` or `low`. Nothing gets thrown away, and downstream code can pick the precision/recall tradeoff.

- **Pose thresholds had to be tightened.** Wrist speed is now measured in shoulder widths per second so it stays scale invariant, and an elbow extension gate of 135 degrees cuts down the worst pose false positives without going so strict that real shots are missed.

- **Tuning felt like whack a mole.** Stricter thresholds killed false positives but missed real shots. Looser thresholds caught more shots but brought back the false positives. Pure heuristics seem to have a ceiling here. The clean fix is a small action recognition model trained on a few hundred labeled clips.

- **Player IDs swap on occlusion.** ByteTrack is good but not perfect. When two players cross, IDs can switch, so `player_id=2` in the first half might be a different person later. A re identification model on top would fix it.

- **MediaPipe runs once per player crop.** It is single person, so I run it on each player bounding box with `static_image_mode=True` (otherwise its temporal smoothing leaks landmarks between players). On CPU this used to dominate the per frame cost; now TrackNet is the bigger cost and a GPU pulls both down sharply.

- **No custom training was done.** Everything uses pre trained weights. Training a small YOLO on padel specific footage, or fine tuning TrackNet on padel ball clips, would close most of the remaining accuracy gap. Out of scope for this prototype.

## Limitations

1. Pose only events still over fire. They are tagged `med` or `low`. For clean analytics, filter to `confidence == "high"`.
2. Right handedness assumption. Side labels invert for left handed players.
3. 2D bounce detection. A wall bounce and a floor bounce both look like a y velocity flip and can't be told apart without depth.
4. Player IDs can swap on occlusion. Stable per run, not globally identifiable.
5. CPU is slow. End to end at ~0.3 fps with TrackNet. A GPU brings it to a comfortable speed.

## What I would build next

1. Fine tune TrackNet on padel ball footage. Even a few hundred labeled frames would push recall further.
2. Fine tune YOLO on padel footage for the racket class.
3. Re identification on top of ByteTrack (e.g. OSNet) for stable player IDs across occlusion.
4. A small action recognition head (TSN or SlowFast) trained on a few hundred labeled swing clips. Use it as a third merge signal alongside pose and contact.
5. Auto court detection via Hough lines on white pixels. Removes the manual calibration step.
6. Multi camera triangulation for true 3D ball tracking and proper bounce vs wall disambiguation.

## Demo

The annotated output and the input video are not in the repo because of size. Drive links:

- [Input video](https://drive.google.com/file/d/1g5jKIXEjbzGJrQJczRPPBeqIQz37TYM_/view?usp=sharing)
- [Annotated output](https://drive.google.com/file/d/1ik4VnMhxhaC_M6jOl656LmbJi0PE3MOr/view?usp=sharing)
- [outputs/ zip (events.csv, events.json, summary.json, dashboard.png)](https://drive.google.com/drive/folders/1ttjDwEkKP6b-O3j0K136FrQVESb-rc5j?usp=sharing)

To reproduce locally, drop `input.mp4` into `padel/data/`, download the TrackNet weights (see Setup), run `python court_roi.py`, then `python main.py`.

## Running on a GPU

The pipeline auto detects CUDA. If `torch.cuda.is_available()` returns true, TrackNet runs on the GPU and the whole thing speeds up by roughly 10x. Steps:

1. Copy the `padel/` folder to the GPU machine. Skip `.venv/` and `outputs/`. The TrackNet weights and YOLO weights are not in the repo either, they get fetched on the new machine.
2. On the GPU machine, recreate the environment:
   ```bash
   cd padel
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   ```
   The second pip install replaces the CPU torch wheel with the CUDA build. Use the `cu121` channel for CUDA 12.1, or `cu118` for CUDA 11.8.
3. Download the TrackNet weights once (see Setup).
4. Drop your video at `data/input.mp4`.
5. Run `python court_roi.py` once to set the court polygon. If you cannot use a GUI on the remote machine, copy the local `court_roi.json` over to the GPU machine instead, or pass the polygon inline via the `PADEL_ROI` env var (see the Run reference at the bottom).
6. Run `python main.py`.

You should see `TrackNet loaded` followed by a much higher fps in the per frame log. A five minute clip should finish in a few minutes rather than a couple of hours.

## Run reference

```bash
python main.py                                                   # standard run, also writes dashboard.png
python dashboard.py                                              # re render dashboard from existing summary.json
PADEL_HEADLESS=1 python main.py                                  # error out if no ROI source available
PADEL_ROI="450,320;1480,320;1700,1020;220,1020" python main.py   # provide ROI inline
```

Outputs land in `outputs/` and are overwritten each run.
