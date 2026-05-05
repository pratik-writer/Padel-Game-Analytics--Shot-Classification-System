# Padel Shot Classification Study Guide

This document turns the raw implementation notes into a usable reference for building a padel shot-classification prototype. The goal is not just to copy code, but to understand the full pipeline well enough to explain, debug, and improve it.

---

## 1. System Overview

The prototype combines four core components:

1. YOLOv8 for object detection
2. YOLO tracking for persistent player IDs across frames
3. MediaPipe Pose for upper-body landmark estimation
4. Rule-based heuristics for shot classification

The system reads a video frame by frame, detects players and the ball, tracks the player involved in the shot, estimates that player's arm and shoulder positions, and then classifies the motion as a likely forehand, backhand, or smash.

### End-to-End Flow

```text
  VIDEO INPUT
      |
      v
+-------------------+
| Read next frame   |
+-------------------+
      |
      v
+-------------------+
| YOLO detect       |
| person / ball     |
+-------------------+
      |
      v
+-------------------+
| Track players     |
| keep same IDs     |
+-------------------+
      |
      v
+-------------------+
| Select hitter     |
| nearest to ball   |
+-------------------+
      |
      v
+-------------------+
| Crop player ROI   |
| run MediaPipe     |
+-------------------+
      |
      v
+-------------------+
| Apply heuristics  |
| classify shot     |
+-------------------+
      |
      v
+-------------------+
| Draw + save       |
| video and CSV     |
+-------------------+
```

---

## 2. YOLOv8: Detection and Tracking

YOLO means "You Only Look Once." It is an object detection model that processes an image and predicts bounding boxes around objects it recognizes.

For this project, YOLO is used to find:

- players
- the ball
- optionally the racket

### Relevant COCO Classes

Most pretrained YOLOv8 models are trained on the COCO dataset. The classes that matter here are:

- `0` -> person
- `32` -> sports ball
- `38` -> tennis racket

The racket class is not perfect for padel, but it is often close enough for a prototype.

### Detection vs Tracking

Detection alone does not know identity across frames.

Example:

- frame 1: detects a player
- frame 2: detects the same player again
- without tracking: these are just two independent detections

Tracking solves this by attaching an ID to the object.

```text
Frame 1               Frame 2               Frame 3

Player box            Player box            Player box
   ID=7      ---->       ID=7      ---->       ID=7

Without tracking:
  unknown A            unknown B            unknown C
```

In YOLOv8, tracking is typically enabled with persistent tracking, which allows the system to maintain the same player ID across consecutive frames. This is essential if the final CSV or JSON must indicate which player performed the shot.

### Important YOLO Output Fields

Each detected object usually provides:

- `xyxy`: bounding-box corners as `(x1, y1, x2, y2)`
- `cls`: class ID
- `conf`: confidence score
- `id`: tracking ID if tracking is enabled

These values are the minimum data needed for the rest of the system.

### Practical Rule

Ignore weak detections when possible. A confidence threshold around `0.5` is a sensible starting point for a prototype. Lower thresholds may introduce many false positives.

---

## 3. MediaPipe Pose: Turning a Player Box into Body Landmarks

Bounding boxes tell you where a player is, but not how they are moving. To classify the shot, you need body geometry. That is where MediaPipe Pose helps.

MediaPipe estimates body landmarks, giving you a skeletal representation of the player.

### Landmarks That Matter Most

For a first version, focus only on the upper body:

- left shoulder: `11`
- right shoulder: `12`
- left elbow: `13`
- right elbow: `14`
- left wrist: `15`
- right wrist: `16`

These are enough to reason about arm direction and whether the hand is above shoulder level.

### Conceptual Landmark Layout

```text
        head
         O
        /|\
       / | \
      O  |  O      <- shoulders
      |     \
      O      O     <- elbows
      |       \
      O        O   <- wrists
```

You do not need all 33 landmarks to build a convincing prototype. Restricting the logic to a few upper-body points keeps the system simpler and easier to debug.

### Critical Detail: Normalized Coordinates

MediaPipe does not directly return pixel coordinates. It returns normalized values between `0.0` and `1.0`.

That means:

- `x = 0.0` is the left edge of the image
- `x = 1.0` is the right edge of the image
- `y = 0.0` is the top edge of the image
- `y = 1.0` is the bottom edge of the image

To convert a landmark to pixel space:

$$
Pixel_X = \text{int}(landmark.x \times image\_width)
$$

$$
Pixel_Y = \text{int}(landmark.y \times image\_height)
$$

If you skip this conversion, your drawing and geometry calculations will be wrong.

---

## 4. Coordinate System in OpenCV

OpenCV uses an image coordinate system that is different from the standard math graph used in school.

```text
(0,0) ----------------------------------> x
  |
  |
  |
  v
  y
```

Key implications:

- moving right increases `x`
- moving down increases `y`
- smaller `y` means the point is higher on the screen

This matters a lot for shot logic.

Example:

- if the wrist has a smaller `y` than the shoulder, the wrist is above the shoulder
- that can indicate an overhead motion such as a smash

---

## 5. Shot Classification Logic

This project does not need a full machine-learning classifier for shot type. A simple heuristic system is enough for a solid prototype.

The idea is to use landmark positions and a few geometric rules.

### 5.1 Forehand vs Backhand

Assume the following for the first version:

- the broadcast camera is behind the player or close to that angle
- the player is right-handed

First compute the body's horizontal center using the shoulders:

$$
X_{center} = \frac{X_{left\_shoulder} + X_{right\_shoulder}}{2}
$$

Then compare the right wrist position to that center line.

```text
            player's body

 left side      center line        right side
    |                |                 |
    |                |                 |
----|----------------|-----------------|----> x

if right wrist is here  ---------> likely forehand
if right wrist crosses here <----- likely backhand
```

Heuristic:

- if `right_wrist_x > x_center`, classify as forehand
- if `right_wrist_x < x_center`, classify as backhand

This is not universally correct, but it is acceptable for a prototype if the assumption is stated clearly.

### 5.2 Smash or Overhead Shot

A smash usually happens when the hitting arm is raised well above shoulder height.

```text
      wrist
        O
        |
        |
  shoulder O

smaller y = higher on screen
```

Heuristic:

- if `wrist_y < shoulder_y - threshold`, classify as smash

The threshold is important because raw landmark noise can cause false triggers. A fixed pixel threshold or a body-relative threshold can be used.

### 5.3 Suggested Decision Order

Use a decision order that avoids ambiguity:

```text
if wrist is clearly above shoulder:
    shot = smash
else if wrist is right of body center:
    shot = forehand
else:
    shot = backhand
```

This makes the logic predictable and easy to explain.

---

## 6. How to Identify the Hitting Player

If multiple players are visible, the system must infer which player is taking the shot.

The simplest usable strategy is:

1. detect the ball
2. detect all players
3. compute the distance from the ball to each player box center
4. choose the nearest player as the likely hitter

### Simple Distance Logic

```text
Ball:          o

Player A:   [ box ]
Player B:                 [ box ]

Measure distance from ball center
to each player-box center.

Smaller distance -> more likely hitter
```

This is not perfect, but it is practical for a prototype and easy to implement.

---

## 7. OpenCV Video Processing Pipeline

OpenCV handles the frame-by-frame mechanics.

### Read the Video

You typically start with:

```python
cap = cv2.VideoCapture("video.mp4")
```

You also need:

- frame width
- frame height
- FPS

These values are necessary to preserve the output video's shape and timing.

### Main Processing Loop

The usual loop is:

1. read a frame
2. run YOLO detection and tracking
3. identify the likely hitter
4. crop the hitter region of interest
5. run MediaPipe Pose on that crop
6. convert landmarks to pixel coordinates
7. apply shot heuristics
8. draw overlays
9. write the frame to output

### Pipeline Diagram

```text
+-------------+
| Read frame  |
+-------------+
       |
       v
+-------------+
| Detect      |
| players/ball|
+-------------+
       |
       v
+-------------+
| Track IDs   |
+-------------+
       |
       v
+-------------+
| Pick hitter |
+-------------+
       |
       v
+-------------+
| Pose on ROI |
+-------------+
       |
       v
+-------------+
| Classify    |
+-------------+
       |
       v
+-------------+
| Draw/write  |
+-------------+
```

### Writing the Output Video

For MP4 output, a common codec choice is:

```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
```

If the codec, FPS, or frame size are wrong, the output video may not play correctly.

---

## 8. Structured Output: CSV or JSON

Besides the demo video, the system should produce structured data for analysis.

A common record format is:

```python
shot_data.append(
    {
        "timestamp": current_time_in_seconds,
        "player_id": player_track_id,
        "shot_type": predicted_shot,
    }
)
```

At the end of processing, this list can be converted into a DataFrame and saved as CSV.

### Example Output Table

```text
+-----------+-----------+------------+
| timestamp | player_id | shot_type  |
+-----------+-----------+------------+
|   1.24    |     7     | Forehand   |
|   2.08    |     7     | Smash      |
|   3.11    |     3     | Backhand   |
+-----------+-----------+------------+
```

This output is useful for both reporting and debugging.

---

## 9. Assumptions and Limitations

Any prototype like this must clearly state its limits.

Important assumptions:

- players are visible enough for detection
- the ball is detected reliably often enough
- the hitter is the player nearest to the ball
- the camera angle is close to a standard broadcast view
- players are treated as right-handed unless additional logic is added

Important limitations:

- occlusion can hide wrists, shoulders, or the ball
- fast ball motion may break reliable hitter selection
- unusual camera angles reduce heuristic accuracy
- left-handed players may be misclassified by right-hand rules
- racket-only motion is not fully captured if pose landmarks are noisy

These are not failures of the project. They are expected constraints of a lightweight prototype.

---

## 10. Recommended Implementation Order

Build the system in small layers.

### Stage 1

Get YOLO detection working on the video.

Target result:

- bounding boxes around players and the ball

### Stage 2

Enable tracking.

Target result:

- players keep stable IDs across frames

### Stage 3

Add hitter selection.

Target result:

- one likely hitting player chosen per relevant frame

### Stage 4

Add MediaPipe Pose on the selected player crop.

Target result:

- wrist and shoulder landmarks available in pixel coordinates

### Stage 5

Add heuristic shot classification.

Target result:

- forehand / backhand / smash label appears on the frame

### Stage 6

Save output video and CSV.

Target result:

- annotated video
- structured shot log

---

## 11. Mental Model to Keep While Coding

If the code starts becoming confusing, reduce it to this question sequence:

1. What objects are in the frame?
2. Which detected player is most likely hitting the ball?
3. Where are that player's shoulders and wrists?
4. Is the hitting wrist above the shoulder?
5. Is the wrist to the right or left of the body center?
6. What label should be saved and drawn?

That is the whole prototype in compressed form.

---

## 12. Final Summary

The full approach can be expressed in one line:

Detect the players and ball, track the players, choose the likely hitter, estimate upper-body pose, and classify the shot using simple geometry.

If you understand each step in that sentence, you understand the core of the system.