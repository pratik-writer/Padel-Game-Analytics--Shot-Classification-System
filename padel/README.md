# Padel Shot Classifier

A small pipeline that watches a padel match video and figures out who the players are, where the ball is, when a shot is taken, and what kind of shot it was (Forehand, Backhand or Serve/Smash). It writes an annotated video, a CSV and JSON log of every shot, a summary file with per player counts, and a one page dashboard image.

Everything runs on rules and pre trained models. No custom training was done.

## What it can do

- Detect and track the four players, the ball and the rackets across frames.
- Filter out off court people and neighbouring courts using a manual court polygon.
- Run pose estimation (33 landmarks) on each tracked player.
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
  YOLOv8 finds people, balls, rackets
  keep only the 4 players inside the court polygon
  MediaPipe draws a skeleton on each player
  ball tracker tries: YOLO, then motion + white blob, then a predicted position
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
load YOLO model, MediaPipe Pose, court ROI
open input video, create output video writer

for each frame in video:

    detections = YOLO(frame)                       # persons, balls, rackets
    detections = filter_to_court(detections)       # keep top 4 players inside ROI
    ball       = ball_tracker.update(frame)        # YOLO -> motion+white -> predicted
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
  tracker.py               # YOLOv8s + ByteTrack wrapper (person, ball, racket)
  pose.py                  # runs MediaPipe Pose per player crop, remaps to full frame
  classifier.py            # pose only shot detector (wrist speed + elbow angle)
  ball.py                  # hybrid ball tracker: YOLO + motion white + prediction
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

The `data/` and `outputs/` folders are referenced by the code but their contents are not in the repo, since the videos are large. Drive links to the input and output videos are at the bottom of this README.

## Setup

```bash
cd padel
python -m venv .venv
.venv\Scripts\activate           # on Windows. use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Run

```bash
# put your video at data/input.mp4
python court_roi.py              # one time: click 4 court corners, press ENTER
python main.py                   # processes the video
```

CPU is slow. Expect 1 to 3 fps at `imgsz=1280`. Fine for a one off run.

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

1. The camera does not move. The static mask bootstrap and the fixed court ROI both rely on this.
2. One court of interest. The video has multiple courts in view; we pick one.
3. Players are right handed. Side classification flips for left handers.
4. The ball is white. The motion + white fallback uses an HSV brightness/saturation filter.
5. COCO classes generalise to padel. YOLOv8s is COCO pre trained, so its `tennis racket` covers padel rackets and its `sports ball` is the weakest link.

## Challenges

- **Racket detection with `yolov8n` was poor.** Rackets were rarely separated from the player boxes. Switching to `yolov8s` and bumping `imgsz` to 1280 fixed it. Cost was speed, but accuracy mattered more here.

- **Off court people kept getting tracked.** The video shows neighbouring courts and bystanders, all of whom got detections, skeletons, and false shot counts. Solved by adding `court_roi.py`, where I click the four corners of the main court once and then filter detections by foot point inside that polygon. Top 4 players by bounding box area are kept.

- **Ball is tiny and almost invisible during fast shots.** YOLO's `sports ball` confidence stays below 0.10 most of the time on a 3 to 5 pixel ball. This was the hardest single problem.

- **Pure colour tracking for the ball didn't work either.** Tried treating the white ball as a colour tracking problem, since the rest of the court is mostly green or blue. It works when the ball is sitting still, but motion blur and the small size dilute the white pixels into the background during actual play. Court lines, the top of the net, and white pants on players also lit up in the white mask.

- **Ball tracker became a fallback chain.** `ball.py` now tries YOLO first, then a motion + white blob detector with a static mask bootstrap (to subtract always white pixels like court lines and net), then a constant velocity prediction for short gaps. Better than YOLO alone, but recall on fast shots is still around 30 to 50 percent. A model like TrackNet would be the right tool here.

- **First shot classifier (pose only) over fired badly.** Wrist speed peaks fired on walking, casual hand movements, grip adjustments, anything. The log was flooded with false positives.

- **Second attempt (contact only) under fired.** Requiring a ball trajectory bend near a wrist is the right signal, but it depends on the ball being tracked, which it often isn't. So real shots got missed.

- **Final classifier is a hybrid with confidence tags.** Both detectors run in parallel and `event_merger.py` fuses them. Both fire within 0.4 seconds for the same player gives a `high` confidence shot. Only one fires gives `med` or `low`. Nothing gets thrown away, and downstream code can pick the precision/recall tradeoff.

- **Pose thresholds had to be tightened.** Wrist speed is now measured in shoulder widths per second so it stays scale invariant, and an elbow extension gate of 135 degrees cuts down the worst pose false positives without going so strict that real shots are missed.

- **Tuning felt like whack a mole.** Stricter thresholds killed false positives but missed real shots. Looser thresholds caught more shots but brought back the false positives. Pure heuristics seem to have a ceiling here. The clean fix is a small action recognition model trained on a few hundred labeled clips.

- **Player IDs swap on occlusion.** ByteTrack is good but not perfect. When two players cross, IDs can switch, so `player_id=2` in the first half might be a different person later. A re identification model on top would fix it.

- **MediaPipe runs once per player crop.** It is single person, so I run it on each player bounding box with `static_image_mode=True` (otherwise its temporal smoothing leaks landmarks between players). On CPU this dominates the per frame cost.

- **No custom training was done.** Everything uses pre trained weights. Training a small YOLO on padel specific footage, especially for the ball and racket, would close most of the remaining accuracy gap. Out of scope for this prototype.

## Limitations

1. Ball recall on fast shots is moderate. Tiny plus blurry equals hard. Expect to miss some.
2. The static white mask does not update. Long matches with auto exposure drift would slowly leak court lines back into the candidate mask.
3. Player bounding box masking can hide the ball at contact. A ball overlapping a wrist sits inside the bounding box at the exact moment we want to detect it. I shrink the box by 4 pixels as a compromise.
4. Pose only events still over fire. They are tagged `med` or `low`. For clean analytics, filter to `confidence == "high"`.
5. Right handedness assumption. Side labels invert for left handed players.
6. 2D bounce detection. A wall bounce and a floor bounce both look like a y velocity flip and can't be told apart without depth.
7. One court only. The ROI is fixed; multi court analysis would need one ROI per court.
8. Player IDs can swap on occlusion. Stable per run, not globally identifiable.

## What I would build next

1. TrackNet for the ball. Heatmap based, designed for tiny fast balls in racket sports. Single biggest accuracy win available.
2. Fine tune YOLO on padel footage. Even a few hundred annotated frames would massively improve racket and ball detection.
3. Re identification on top of ByteTrack (e.g. OSNet) for stable player IDs across occlusion.
4. A small action recognition head (TSN or SlowFast) trained on a few hundred labeled swing clips. Use it as a third merge signal alongside pose and contact.
5. Auto court detection via Hough lines on white pixels. Removes the manual calibration step.
6. Multi camera triangulation for true 3D ball tracking and proper bounce vs wall disambiguation.

## Demo

The annotated output and the input video are not in the repo because of size. Drive links:

- Input video: `<https://drive.google.com/file/d/14ySwgP3Y2PH2BnsrC8ESpXAOHIQGYnvy/view?usp=sharing>`
- Annotated output: `<paste-drive-link-here>`
- `outputs/` zip (events.csv, events.json, summary.json, dashboard.png): `<paste-drive-link-here>`

To reproduce locally, drop `input.mp4` into `padel/data/`, run `python court_roi.py`, then `python main.py`.

## Run reference

```bash
python main.py                                                   # standard run, also writes dashboard.png
python dashboard.py                                              # re render dashboard from existing summary.json
PADEL_HEADLESS=1 python main.py                                  # error out if no ROI source available
PADEL_ROI="450,320;1480,320;1700,1020;220,1020" python main.py   # provide ROI inline
```

Outputs land in `outputs/` and are overwritten each run.
