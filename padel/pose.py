import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

KEYPOINTS_OF_INTEREST = {
    "nose":           PoseLandmark.NOSE,
    "l_shoulder":     PoseLandmark.LEFT_SHOULDER,
    "r_shoulder":     PoseLandmark.RIGHT_SHOULDER,
    "l_elbow":        PoseLandmark.LEFT_ELBOW,
    "r_elbow":        PoseLandmark.RIGHT_ELBOW,
    "l_wrist":        PoseLandmark.LEFT_WRIST,
    "r_wrist":        PoseLandmark.RIGHT_WRIST,
    "l_hip":          PoseLandmark.LEFT_HIP,
    "r_hip":          PoseLandmark.RIGHT_HIP,
}

POSE_CONNECTIONS = [
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
]

class PoseEstimator:
    def __init__(self, min_det_conf=0.3, min_track_conf=0.3,
                 model_complexity=1, pad_ratio=0.10):
        self._make_pose = lambda: mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.pose = self._make_pose()
        self.pad_ratio = pad_ratio

    def close(self):
        self.pose.close()

    def estimate_for_tracks(self, frame, tracks):
        """Return {track_id: {keypoint: (x_frame, y_frame, visibility)}} for persons only."""
        H, W = frame.shape[:2]
        out = {}
        for t in tracks:
            if t["name"] != "person" or t["id"] < 0:
                continue
            x1, y1, x2, y2 = t["bbox"]
            pad_x = (x2 - x1) * self.pad_ratio
            pad_y = (y2 - y1) * self.pad_ratio
            cx1 = max(0, int(x1 - pad_x)); cy1 = max(0, int(y1 - pad_y))
            cx2 = min(W, int(x2 + pad_x)); cy2 = min(H, int(y2 + pad_y))
            if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                continue

            crop = frame[cy1:cy2, cx1:cx2]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res  = self.pose.process(rgb)
            if not res.pose_landmarks:
                continue

            cw, ch = cx2 - cx1, cy2 - cy1
            kps = {}
            for name, idx in KEYPOINTS_OF_INTEREST.items():
                lm = res.pose_landmarks.landmark[idx]
                fx = cx1 + lm.x * cw
                fy = cy1 + lm.y * ch
                kps[name] = (float(fx), float(fy), float(lm.visibility))
            out[t["id"]] = kps
        return out