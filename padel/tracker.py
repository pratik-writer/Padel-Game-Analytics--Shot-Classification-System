from ultralytics import YOLO
import numpy as np

COCO_PERSON       = 0
COCO_SPORTS_BALL  = 32
COCO_TENNIS_RACKET = 38
TARGET_CLASSES = [COCO_PERSON, COCO_SPORTS_BALL, COCO_TENNIS_RACKET]
NAME_BY_CLS = {0: "person", 32: "ball", 38: "racket"}

class Tracker:
    def __init__(self, weights="yolov8n.pt", conf=0.30, iou=0.5,
                 tracker_cfg="bytetrack.yaml", device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.tracker_cfg = tracker_cfg
        self.device = device

    def update(self, frame):
        results = self.model.track(
            source=frame,
            classes=TARGET_CLASSES,
            conf=self.conf,
            iou=self.iou,
            persist=True,                 
            tracker=self.tracker_cfg,
            verbose=False,
            device=self.device,
        )[0]

        tracks = []
        if results.boxes is None or len(results.boxes) == 0:
            return tracks

        boxes = results.boxes
        xyxy  = boxes.xyxy.cpu().numpy()
        cls   = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids   = (boxes.id.cpu().numpy().astype(int)
                 if boxes.id is not None
                 else np.full(len(boxes), -1, dtype=int))

        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, cls, confs, ids):
            tracks.append({
                "id":   int(tid),
                "cls":  int(c),
                "name": NAME_BY_CLS.get(int(c), str(c)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(k),
            })
        return tracks