from ultralytics import YOLO
import numpy as np

COCO_PERSON        = 0
COCO_SPORTS_BALL   = 32
COCO_TENNIS_RACKET = 38
TARGET_CLASSES = [COCO_PERSON, COCO_SPORTS_BALL, COCO_TENNIS_RACKET]
NAME_BY_CLS = {0: "person", 32: "ball", 38: "racket"}

PER_CLASS_MIN_CONF = {
    COCO_PERSON:        0.30,
    COCO_SPORTS_BALL:   0.10,
    COCO_TENNIS_RACKET: 0.10,
}

class Tracker:
    def __init__(self, weights="yolov8s.pt",
                 base_conf=0.10, iou=0.5, imgsz=1280,
                 tracker_cfg="bytetrack.yaml", device=None):
        self.model = YOLO(weights)
        self.base_conf = base_conf       
        self.iou = iou
        self.imgsz = imgsz
        self.tracker_cfg = tracker_cfg
        self.device = device

    def update(self, frame):
        results = self.model.track(
            source=frame,
            classes=TARGET_CLASSES,
            conf=self.base_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker_cfg,
            verbose=False,
            device=self.device,
        )[0]

        tracks = []
        if results.boxes is None or len(results.boxes) == 0:
            return tracks

        b = results.boxes
        xyxy  = b.xyxy.cpu().numpy()
        cls   = b.cls.cpu().numpy().astype(int)
        confs = b.conf.cpu().numpy()
        ids   = (b.id.cpu().numpy().astype(int)
                 if b.id is not None
                 else np.full(len(b), -1, dtype=int))

        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, cls, confs, ids):
            if k < PER_CLASS_MIN_CONF.get(int(c), self.base_conf):
                continue
            tracks.append({
                "id":   int(tid),
                "cls":  int(c),
                "name": NAME_BY_CLS.get(int(c), str(c)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(k),
            })
        return tracks