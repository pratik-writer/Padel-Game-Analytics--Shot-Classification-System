# Padel Shot Classifier

A small pipeline that watches a padel match video and tries to figure out who the players are, where the ball is, when a shot is taken, and what kind of shot it was (Forehand, Backhand or Serve/Smash). It then writes an annotated video, a CSV and JSON log of every shot, a summary file with per player counts, and a one page dashboard image.

Everything runs on rules and pre trained models. No custom training was done. Every decision the system makes can be traced to a number you can change in code.

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

That is the whole idea.

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

If you only read one section of this README, this is it.

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
  data/input.mp4           # your video goes here
  outputs/
    output.mp4
    events.csv
    events.json
    summary.json
    dashboard.png
```

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

These are not guesses, they are choices baked into the code.

1. The camera doesn't move. The static mask bootstrap and the fixed court ROI both rely on this.
2. One court of interest. The video has multiple courts in view; we pick one.
3. Players are right handed. Side classification flips for left handers.
4. The ball is white. The motion + white fallback uses an HSV brightness/saturation filter.
5. COCO classes generalise to padel. YOLOv8s is COCO pre trained, so its `tennis racket` covers padel rackets and its `sports ball` is the weakest link.

## Challenges

This is the honest version of the journey, not the polished diagram version above.

The first thing that broke was racket detection. With `yolov8n` the rackets just couldn't be picked out from the player boxes most of the time. Moving to `yolov8s` and bumping `imgsz` to 1280 fixed it. Cost was speed, but accuracy mattered more here.

Next problem: off court people. The video is wide enough to show neighbouring courts and bystanders. YOLO happily detected all of them, MediaPipe drew skeletons on all of them, and the shot classifier counted their arm waves as shots. The fix was a manual court calibration step. `court_roi.py` opens the first frame, you click the four corners of the court that matters, and from then on detections are filtered by foot point inside that polygon. We then keep only the top 4 players by bounding box area (closer to camera = bigger). One step, most of the noise gone.

Ball tracking was by far the hardest part of this whole project, and it is still the weakest link. YOLO's `sports ball` class barely sees a 3 to 5 pixel padel ball. Confidence stays below 0.10 most of the time, and motion blur during fast shots makes it almost invisible. So I tried treating it as a pure colour problem. The ball is white, the rest of the court is mostly green or blue, just look for white moving blobs. That works perfectly when the ball is sitting still on the floor, but during actual play the motion blur and small size dilute the white pixels into the background. Court lines, the top of the net, even players' white pants would also light up in the white mask. I added a static mask bootstrap to subtract the always white pixels (lines, net), a motion mask to ignore stationary stuff, and a streak detection branch to catch motion blurred ellipses. All of that is in `ball.py`. It is better than YOLO alone but ball recall on fast shots is still maybe 30 to 50 percent. The honest verdict is that per frame detection of a tiny fast object is what specialised architectures like TrackNet are built for. I worked within the constraint instead of training a model.

Shot classification went through three painful iterations.

First attempt was pure pose based wrist speed peak. Easy to write, easy to over fire. Players walking, swinging arms between rallies, adjusting their grip, all classified as shots. The log was flooded.

Second attempt added a ball contact detector. Only count a shot if the ball trajectory bends near a wrist. This is the right signal in theory, but it depends on the ball being tracked, which we already established is unreliable. So now real shots were being missed because the ball wasn't seen at the contact frame.

Third attempt is what is in the repo now. Both detectors run in parallel and an event merger fuses them with a confidence tag. If both fire within 0.4s for the same player, that is a `high` confidence shot. Only one fires, that is `med` or `low`. This way nothing gets thrown away, and downstream code (or you, when reading the CSV) can pick the precision and recall tradeoff. The pose thresholds were also tightened (wrist speed in shoulder widths per second so it stays scale invariant, plus an elbow extension gate of 135 degrees) to cut down the worst pose false positives without going so strict that real shots got missed.

Tuning the classifier felt like whack a mole. Every time I made it stricter to kill false positives, it started missing real shots. Every time I loosened it, the false positives came back. I eventually accepted that pure heuristics have a ceiling here, documented it, and moved on. The right next step is a small action recognition model trained on a few hundred labeled clips. That is the only thing that resolves "swing without ball".

A few smaller annoyances along the way. Player IDs sometimes swap on occlusion. ByteTrack is good but not perfect, so when two players cross, IDs can switch. So `player_id=2` in the first half might be a different person in the second half. A re identification model on top would fix it.

MediaPipe Pose is single person, so it has to run once per player crop, with `static_image_mode=True` (otherwise its temporal smoothing leaks landmarks between players). On CPU this dominates the per frame cost.

And finally, no custom training was done. Everything uses pre trained weights. Training a small YOLO on padel specific footage, especially for the ball and racket, would almost certainly close most of the remaining accuracy gap. I planned for it but didn't have the data or time within scope.

## Limitations

Things you should know before trusting the output.

1. Ball recall on fast shots is moderate. Tiny plus blurry equals hard. Expect to miss some.
2. The static white mask doesn't update. Long matches with auto exposure drift would slowly leak court lines back into the candidate mask.
3. Player bounding box masking can hide the ball at contact. A ball overlapping a wrist is geometrically inside the bounding box at the exact moment we want to detect it. I shrink the box by 4 pixels as a compromise.
4. Pose only events still over fire. That is why they are tagged `med` or `low`. For clean analytics, filter to `confidence == "high"`.
5. Right handedness assumption. Side labels invert for left handed players.
6. 2D bounce detection. A wall bounce and a floor bounce both look like a y velocity flip. We can't tell them apart without depth.
7. One court only. The ROI is fixed. Multi court analysis would need one ROI per court.
8. Player IDs can swap on occlusion. Stable per run, not globally identifiable.

## What I would build next

1. TrackNet for the ball. Heatmap based, designed exactly for tiny fast balls in racket sports. Single biggest accuracy win available.
2. Fine tune YOLO on padel footage. Even a few hundred annotated frames would massively improve racket and ball detection.
3. Re identification on top of ByteTrack (e.g. OSNet) for stable player IDs.
4. A small action recognition head (TSN or SlowFast) trained on a few hundred labeled swing clips. Use it as a third merge signal alongside pose and contact. Cleanest fix for the "swing without ball" problem.
5. Auto court detection via Hough lines on white pixels. Removes the manual calibration step.
6. Multi camera triangulation for true 3D ball tracking and proper bounce vs wall disambiguation.

## Demo

The annotated output and the input video are too large for git. Drive links:

- Input video: `<paste-drive-link-here>`
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
