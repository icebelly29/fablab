"""
Padayani Eye Tracker Prototype (ZED 1 + Bench ROI Lock + Sitting Check)
------------------------------------------------------------------------

Requirements (inside your venv):
    pip install opencv-python numpy tensorflow mediapipe pyzed.sl

Model:
    Download MoveNet MultiPose Lightning TFLite from TF Hub:
      https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1
    Save it as: ./models/movenet_multipose_lightning.tflite

Behavior:
    - Uses ZED 1 stereo camera with depth and optional body tracking
    - Falls back to MoveNet MultiPose if ZED body tracking unavailable
    - Bench ROI: first person who SITS in 3D ROI for N frames becomes LOCKED
    - Sitting detected from skeleton keypoints (torso vs leg ratio)
    - Face orientation check using MediaPipe Face Mesh
    - Eyes follow locked target until track disappears
    - Three pairs of eyes; pupils track locked target's head position
    - Logs lock events to ./logs/locks.csv

Test mode (no camera):
    python padayani_zed_prototype.py --test-sim

Keys:
    'q' or ESC -> quit
"""

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime
from collections import deque

# ----------------------------
# Configuration
# ----------------------------

# ZED camera settings
USE_ZED_BODY_TRACKING = True          # Prefer ZED body tracking if available
ZED_RESOLUTION = "HD720"              # HD720 or HD1080
ZED_FPS = 30

# Bench ROI in 2D pixel coordinates (fractional of frame size)
# Example: bottom-middle region of the frame where bench is visible
BENCH_ROI_X_MIN_FRAC = 0.25           # Left edge (0.0 = leftmost, 1.0 = rightmost)
BENCH_ROI_X_MAX_FRAC = 0.75           # Right edge
BENCH_ROI_Y_MIN_FRAC = 0.55           # Top edge (0.0 = top, 1.0 = bottom)
BENCH_ROI_Y_MAX_FRAC = 0.95           # Bottom edge

# Locking parameters
MIN_LOCK_SIT_TIME = 2.0               # Seconds of continuous sitting required
ID_PERSISTENCE_TIMEOUT = 1.2          # Seconds to maintain lock after disappearance

# Sitting detection
SITTING_RATIO_THRESHOLD = 1.3         # leg_vertical / torso_vertical < this => sitting
KEYPOINT_CONF_THRESH = 0.15

# Model and processing
# Try multiple possible paths
MOVENET_MODEL_PATHS = [
    "./model/movenet_multipose_lightning.tflite",  # Reference style
    "./models/movenet_multipose_lightning.tflite",  # Plural
    "movenet_multipose_lightning.tflite",           # Same directory
]
MOVENET_INPUT_SIZE = 256
NUM_TFLITE_THREADS = 2
INFER_EVERY_N_FRAMES = 1
FRAME_DOWNSCALE = 0.5                 # Scale for pose inference

# Face orientation
FACE_YAW_THRESHOLD = 30.0             # Max yaw angle (degrees) for "facing camera"

# Eyes canvas
EYES_CANVAS_WIDTH = 640
EYES_CANVAS_HEIGHT = 480
EYE_RADIUS = 30
PUPIL_RADIUS = 10
PUPIL_TRAVEL_PX = 20
PUPIL_SMOOTH_ALPHA = 0.3

# Tracker settings
TRACKER_MAX_DISTANCE = 180             # pixels
TRACKER_MAX_AGE = 30                  # frames
TRACK_SMOOTH_ALPHA = 0.15

# ----------------------------
# ZED Camera wrapper
# ----------------------------

class ZEDCamera:
    """Simple ZED camera wrapper"""
    
    def __init__(self):
        self.camera = None
        self.bodies = None
        self.is_initialized = False
        self.camera_info = None
        
    def initialize(self):
        """Initialize ZED camera"""
        try:
            import pyzed.sl as sl
            self.sl = sl
        except ImportError:
            print("ERROR: pyzed.sl not available. Install ZED SDK Python bindings.")
            return False
        
        self.camera = sl.Camera()
        init_params = sl.InitParameters()
        
        if ZED_RESOLUTION == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD720
        
        init_params.camera_fps = ZED_FPS
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: Failed to open ZED camera: {err}")
            return False
        
        self.camera_info = self.camera.get_camera_information()
        print(f"[INFO] ZED Camera initialized: {self.camera_info.camera_model}")
        
        # Enable body tracking if requested
        global USE_ZED_BODY_TRACKING
        if USE_ZED_BODY_TRACKING:
            body_params = sl.BodyTrackingParameters()
            body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
            body_params.enable_tracking = True
            body_params.enable_body_fitting = True
            body_params.body_format = sl.BODY_FORMAT.BODY_34
            
            err = self.camera.enable_body_tracking(body_params)
            if err == sl.ERROR_CODE.SUCCESS:
                self.bodies = sl.Bodies()
                print("[INFO] ZED Body Tracking enabled")
            else:
                print(f"[WARN] Body tracking not available: {err}. Using MoveNet fallback.")
                USE_ZED_BODY_TRACKING = False
        
        self.runtime_params = sl.RuntimeParameters()
        self.mat_left = sl.Mat()
        self.mat_depth = sl.Mat()
        self.is_initialized = True
        return True
    
    def grab_frame(self):
        """Grab and return frame data"""
        if not self.is_initialized:
            return None
        
        if self.camera.grab(self.runtime_params) != self.sl.ERROR_CODE.SUCCESS:
            return None
        
        self.camera.retrieve_image(self.mat_left, self.sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.mat_depth, self.sl.MEASURE.DEPTH)
        
        frame_data = {
            'image': self.mat_left.get_data()[:, :, :3].copy(),
            'depth': self.mat_depth.get_data().copy(),
            'bodies': None
        }
        
        if USE_ZED_BODY_TRACKING and self.bodies:
            self.camera.retrieve_bodies(self.bodies, self.sl.BODY_FORMAT.BODY_34)
            frame_data['bodies'] = self.bodies.body_list
        
        return frame_data
    
    def close(self):
        """Close camera"""
        if self.is_initialized and self.camera:
            self.camera.disable_body_tracking()
            self.camera.close()
            self.is_initialized = False


# ----------------------------
# MoveNet MultiPose wrapper
# ----------------------------

class MoveNetMultiPose:
    """MoveNet MultiPose TFLite wrapper using TensorFlow"""
    
    def __init__(self, model_path, input_size=256, num_threads=2):
        try:
            import tensorflow as tf
            # In TF 2.13, this import is stable and sufficient
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
            )
        except ImportError:
            print("ERROR: tensorflow not installed.")
            raise
        except AttributeError:
             # Fallback for some edge cases
            import tensorflow.lite
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
            )

        input_details = self.interpreter.get_input_details()
        self.input_index = input_details[0]["index"]
        
        self.interpreter.resize_tensor_input(
            self.input_index,
            [1, input_size, input_size, 3],
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_index = self.output_details[0]["index"]
        self.input_size = input_size
        self.input_dtype = self.input_details[0]["dtype"]
        
        print(f"[INFO] MoveNet loaded, input dtype: {self.input_dtype}")
    
    def infer(self, frame_bgr):
        """Run MoveNet on BGR frame, return raw output (1, 6, 56)"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size))
        
        if self.input_dtype in (np.float32, np.float16):
            input_img = rgb_resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0).astype(self.input_dtype)
        else:
            input_img = rgb_resized.astype(np.uint8)
            input_img = np.expand_dims(input_img, axis=0)
        
        self.interpreter.set_tensor(self.input_index, input_img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        return output


# ----------------------------
# MoveNet parsing
# ----------------------------

def parse_movenet_multipose(raw_output, frame_width, frame_height,
                            min_pose_score=0.15, min_kp_score=0.15):
    """Parse MoveNet output into detections"""
    detections = []
    people = raw_output[0]  # (6, 56)
    
    for det in people:
        det = np.asarray(det, dtype=np.float32)
        keypoints_flat = det[:51]  # 17 * (y, x, score)
        ymin, xmin, ymax, xmax, pose_score = det[51:]
        
        if float(pose_score) < min_pose_score:
            continue
        
        x_min_px = max(0, min(frame_width - 1, int(xmin * frame_width)))
        y_min_px = max(0, min(frame_height - 1, int(ymin * frame_height)))
        x_max_px = max(0, min(frame_width - 1, int(xmax * frame_width)))
        y_max_px = max(0, min(frame_height - 1, int(ymax * frame_height)))
        
        # Parse keypoints
        kps = []
        for i in range(17):
            y_norm = keypoints_flat[3 * i + 0]
            x_norm = keypoints_flat[3 * i + 1]
            kp_score = keypoints_flat[3 * i + 2]
            
            x_px = float(x_norm * frame_width)
            y_px = float(y_norm * frame_height)
            
            if kp_score < min_kp_score:
                kp_score = 0.0
            
            kps.append([x_px, y_px, float(kp_score)])
        
        detections.append({
            "keypoints": np.array(kps, dtype=np.float32),
            "bbox": (x_min_px, y_min_px, x_max_px, y_max_px),
            "pose_score": float(pose_score),
        })
    
    return detections


# ----------------------------
# Tracker
# ----------------------------

class RobustTracker:
    """Simple centroid tracker with smoothing"""
    
    def __init__(self, max_distance=TRACKER_MAX_DISTANCE,
                 max_age=TRACKER_MAX_AGE, smooth_alpha=TRACK_SMOOTH_ALPHA):
        self.max_distance = max_distance
        self.max_age = max_age
        self.smooth_alpha = smooth_alpha
        self.tracks = {}
        self.next_id = 1
    
    @staticmethod
    def _centroid_from_bbox(bbox):
        x_min, y_min, x_max, y_max = bbox
        return np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)
    
    def update(self, detections, frame_idx):
        """Update tracks with detections"""
        dets = []
        for d in detections:
            bbox = d["bbox"]
            centroid = self._centroid_from_bbox(bbox)
            dets.append({
                "bbox": bbox,
                "keypoints": d["keypoints"],
                "pose_score": d["pose_score"],
                "centroid": centroid,
                "3d_pos": d.get("3d_pos", None),  # Optional 3D position
            })
        
        for tid, tr in self.tracks.items():
            tr["matched_in_frame"] = False
        
        track_ids = list(self.tracks.keys())
        used_tracks = set()
        
        # Greedy matching
        for det in dets:
            best_tid = None
            best_dist = float("inf")
            
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                tr = self.tracks[tid]
                dist = np.linalg.norm(det["centroid"] - tr["centroid"])
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            
            if best_tid is not None and best_dist <= self.max_distance:
                tr = self.tracks[best_tid]
                # Smooth update
                tr["centroid"] = (self.smooth_alpha * det["centroid"] +
                                 (1.0 - self.smooth_alpha) * tr["centroid"])
                bbox_old = tr["bbox"]
                bbox_new = det["bbox"]
                tr["bbox"] = tuple(int(self.smooth_alpha * bbox_new[i] +
                                      (1 - self.smooth_alpha) * bbox_old[i])
                                  for i in range(4))
                tr["keypoints"] = det["keypoints"]
                tr["pose_score"] = det["pose_score"]
                if det.get("3d_pos"):
                    tr["3d_pos"] = det["3d_pos"]
                tr["last_seen"] = frame_idx
                tr["matched_in_frame"] = True
                used_tracks.add(best_tid)
            else:
                # New track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "id": tid,
                    "bbox": det["bbox"],
                    "keypoints": det["keypoints"],
                    "centroid": det["centroid"],
                    "pose_score": det["pose_score"],
                    "3d_pos": det.get("3d_pos", None),
                    "last_seen": frame_idx,
                    "matched_in_frame": True,
                    "sitting_grace": 0,
                    "facing_grace": 0,
                }
                used_tracks.add(tid)
        
        # Age out stale tracks
        to_delete = [tid for tid, tr in self.tracks.items()
                    if frame_idx - tr["last_seen"] > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]
        
        return self.tracks


# ----------------------------
# Sitting detection
# ----------------------------

def is_sitting(track):
    """Check if person is sitting using keypoints"""
    kps = track["keypoints"]
    
    # MoveNet indices
    L_SH, R_SH = 5, 6
    L_HP, R_HP = 11, 12
    L_KN, R_KN = 13, 14
    L_AN, R_AN = 15, 16
    
    # --- Check 1: Keypoint confidence ---
    needed_for_ratio = [L_SH, R_SH, L_HP, R_HP]
    for idx in needed_for_ratio:
        if kps[idx, 2] < KEYPOINT_CONF_THRESH:
            return False
    
    # --- Check 2: Torso vs. Leg Ratio ---
    sh_y = 0.5 * (kps[L_SH, 1] + kps[R_SH, 1])
    hp_y = 0.5 * (kps[L_HP, 1] + kps[R_HP, 1])
    torso_v = abs(sh_y - hp_y)
    
    if torso_v < 1.0: return False # Avoid division by zero / unstable calcs
    
    leg_verticals = []
    if kps[L_AN, 2] >= KEYPOINT_CONF_THRESH:
        leg_verticals.append(abs(kps[L_HP, 1] - kps[L_AN, 1]))
    if kps[R_AN, 2] >= KEYPOINT_CONF_THRESH:
        leg_verticals.append(abs(kps[R_HP, 1] - kps[R_AN, 1]))
    
    if not leg_verticals: return False
    
    ratio = min(leg_verticals) / torso_v
    if ratio >= SITTING_RATIO_THRESHOLD:
        return False

    # --- Check 3: Hips must be below knees (stricter check) ---
    needed_for_hip_check = [L_KN, R_KN]
    for idx in needed_for_hip_check:
        if kps[idx, 2] < KEYPOINT_CONF_THRESH:
            return False # Can't be sure without seeing knees
            
    hip_y = 0.5 * (kps[L_HP, 1] + kps[R_HP, 1])
    knee_y = 0.5 * (kps[L_KN, 1] + kps[R_KN, 1])

    if hip_y < knee_y: # In image coords, higher y is lower on screen
        return False # Hips are higher than knees, must be standing

    # All checks passed
    return True


def compute_bench_roi(frame_width, frame_height):
    """Compute bench ROI in pixel coordinates"""
    x_min = int(frame_width * BENCH_ROI_X_MIN_FRAC)
    x_max = int(frame_width * BENCH_ROI_X_MAX_FRAC)
    y_min = int(frame_height * BENCH_ROI_Y_MIN_FRAC)
    y_max = int(frame_height * BENCH_ROI_Y_MAX_FRAC)
    return (x_min, y_min, x_max, y_max)


def is_inside_roi(centroid_x, centroid_y, roi):
    """Check if 2D centroid is inside bench ROI"""
    x_min, y_min, x_max, y_max = roi
    return (x_min <= centroid_x <= x_max) and (y_min <= centroid_y <= y_max)


# ----------------------------
# Face orientation
# ----------------------------

class FaceDetector:
    """MediaPipe face mesh for face orientation"""
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.available = True
        except ImportError:
            print("[WARN] MediaPipe not available. Face orientation check disabled.")
            self.available = False
    
    def get_face_yaw(self, image, bbox=None):
        """Get face yaw angle in degrees"""
        if not self.available:
            return 0.0, True
        
        if bbox:
            x, y, w, h = bbox
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2*margin)
            h = min(image.shape[0] - y, h + 2*margin)
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                return 0.0, False
            image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            nose = lm[1]
            left_eye = lm[33]
            right_eye = lm[263]
            
            eye_center_x = (left_eye.x + right_eye.x) / 2
            nose_offset = nose.x - eye_center_x
            yaw_degrees = nose_offset * 60.0  # Rough calibration
            
            is_facing = abs(yaw_degrees) < FACE_YAW_THRESHOLD
            return yaw_degrees, is_facing
        
        return 0.0, False
    
    def close(self):
        if self.available:
            self.face_mesh.close()


# ----------------------------
# Eyes controller
# ----------------------------

class EyesController:
    """Three pairs of eyes that track target"""
    
    def __init__(self, num_pairs=3):
        self.num_pairs = num_pairs
        self.smoothed_offsets_x = [0.0] * num_pairs
        self.smoothed_offsets_y = [0.0] * num_pairs
    
    def update_target(self, target_x, target_y, canvas_width, canvas_height):
        """Update eye target position"""
        # Normalize to [-1, 1]
        norm_x = (target_x / canvas_width - 0.5) * 2.0
        norm_y = (target_y / canvas_height - 0.5) * 2.0
        
        for i in range(self.num_pairs):
            self.smoothed_offsets_x[i] = (
                PUPIL_SMOOTH_ALPHA * norm_x +
                (1.0 - PUPIL_SMOOTH_ALPHA) * self.smoothed_offsets_x[i]
            )
            self.smoothed_offsets_y[i] = (
                PUPIL_SMOOTH_ALPHA * norm_y +
                (1.0 - PUPIL_SMOOTH_ALPHA) * self.smoothed_offsets_y[i]
            )
    
    def render(self):
        """Render eyes canvas"""
        canvas = np.ones((EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8) * 50
        
        rows_y = [
            int(EYES_CANVAS_HEIGHT * 0.22),
            int(EYES_CANVAS_HEIGHT * 0.5),
            int(EYES_CANVAS_HEIGHT * 0.78),
        ]
        center_x = EYES_CANVAS_WIDTH // 2
        eye_spacing = int(EYES_CANVAS_WIDTH * 0.20)
        
        for i in range(self.num_pairs):
            row_y = rows_y[i]
            offset_x = self.smoothed_offsets_x[i]
            offset_y = self.smoothed_offsets_y[i]
            
            pupil_dx = int(offset_x * PUPIL_TRAVEL_PX)
            pupil_dy = int(offset_y * PUPIL_TRAVEL_PX * 0.5)
            
            left_center = (center_x - eye_spacing // 2, row_y)
            right_center = (center_x + eye_spacing // 2, row_y)
            
            for cx, cy in [left_center, right_center]:
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (255, 255, 255), -1)
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (0, 0, 0), 2)
                
                pupil_cx = max(cx - EYE_RADIUS + PUPIL_RADIUS,
                              min(cx + EYE_RADIUS - PUPIL_RADIUS, cx + pupil_dx))
                pupil_cy = max(cy - EYE_RADIUS + PUPIL_RADIUS,
                              min(cy + EYE_RADIUS - PUPIL_RADIUS, cy + pupil_dy))
                cv2.circle(canvas, (pupil_cx, pupil_cy), PUPIL_RADIUS, (0, 0, 0), -1)
        
        return canvas


# ----------------------------
# Main application
# ----------------------------

def project_3d_to_2d(point_3d, fx, fy, cx, cy):
    """Project 3D point to 2D pixel coordinates"""
    x, y, z = point_3d
    if z <= 0:
        return (int(cx), int(cy))
    u = int((x * fx) / z + cx)
    v = int((y * fy) / z + cy)
    return (u, v)


def main(test_sim=False):
    """Main application loop"""
    
    # Test simulation mode
    if test_sim:
        print("[INFO] Running in test simulation mode")
        eyes = EyesController(num_pairs=3)
        start_time = time.time()
        
        while True:
            t = time.time() - start_time
            target_x = EYES_CANVAS_WIDTH // 2 + int(np.sin(t * 0.3) * 150)
            target_y = EYES_CANVAS_HEIGHT // 2 + int(np.cos(t * 0.2) * 100)
            
            eyes.update_target(target_x, target_y, EYES_CANVAS_WIDTH, EYES_CANVAS_HEIGHT)
            eyes_canvas = eyes.render()
            
            cv2.imshow("Eyes Feed (TEST MODE)", eyes_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return
    
    # Initialize ZED camera
    zed = ZEDCamera()
    if not zed.initialize():
        print("ERROR: Failed to initialize ZED camera")
        print("Please check:")
        print("  1. ZED 1 camera is connected via USB 3.0")
        print("  2. ZED SDK is installed")
        print("  3. pyzed.sl is installed: pip install pyzed.sl")
        return
    
    # Initialize MoveNet (fallback or hybrid)
    pose_model = None
    if not USE_ZED_BODY_TRACKING or True:  # Always load for fallback
        model_path = None
        for path in MOVENET_MODEL_PATHS:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            pose_model = MoveNetMultiPose(model_path, MOVENET_INPUT_SIZE, NUM_TFLITE_THREADS)
            print(f"[INFO] MoveNet model loaded from: {model_path}")
        else:
            print(f"[WARN] MoveNet model not found in any of these paths:")
            for path in MOVENET_MODEL_PATHS:
                print(f"  - {path}")
            print("[WARN] System will continue but pose detection will not work.")
    
    # Check if we have at least one detection method
    if not USE_ZED_BODY_TRACKING and pose_model is None:
        print("\n[ERROR] No pose detection method available!")
        print("  - ZED body tracking: Not compatible with ZED 1")
        print("  - MoveNet: Model not found")
        print("\nTo fix: Download MoveNet model and place it in one of these locations:")
        for path in MOVENET_MODEL_PATHS:
            print(f"  - {path}")
        print("\nDownload from: https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1")
        zed.close()
        return
    
    face_detector = FaceDetector()
    tracker = RobustTracker()
    eyes = EyesController(num_pairs=3)
    
    # Get camera intrinsics
    calib = zed.camera_info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
    
    # Lock state
    locked_target_id = None
    lock_start_time = None
    sitting_timers = {}  # tid -> accumulated sitting time
    lock_events = []
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    frame_idx = 0
    last_fps_time = time.time()
    last_loop_time = time.time()
    frame_count_for_fps = 0
    fps = 0.0
    
    print("Press 'q' or ESC to quit.")
    
    try:
        while True:
            frame_data = zed.grab_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            image = frame_data['image']
            image = cv2.flip(image, -1) # Flip image 180 degrees
            depth = frame_data['depth']
            bodies = frame_data['bodies']
            
            h, w = image.shape[:2]
            current_time = time.time()
            dt = current_time - last_loop_time
            last_loop_time = current_time
            frame_idx += 1
            
            # Compute bench ROI in pixels for this frame
            bench_roi = compute_bench_roi(w, h)
            bx_min, by_min, bx_max, by_max = bench_roi
            
            overlay = image.copy()
            
            # --- 1. Get all raw detections from the active model ---
            all_raw_detections = []
            if USE_ZED_BODY_TRACKING and bodies:
                # Use ZED body tracking
                for body in bodies:
                    if len(body.keypoint) < 13: continue
                    
                    pelvis_3d = body.keypoint[12].get()
                    head_3d = body.keypoint[0].get()
                    
                    pelvis_2d = project_3d_to_2d(pelvis_3d, fx, fy, cx, cy)
                    head_2d = project_3d_to_2d(head_3d, fx, fy, cx, cy)
                    
                    x_min = max(0, pelvis_2d[0] - 100)
                    y_min = max(0, head_2d[1] - 50)
                    x_max = min(w, pelvis_2d[0] + 100)
                    y_max = min(h, pelvis_2d[1] + 100)
                    
                    kps_list = [[p[0], p[1], 0.9] for p in (project_3d_to_2d(kp.get(), fx, fy, cx, cy) for kp in body.keypoint)]

                    all_raw_detections.append({
                        "keypoints": np.array(kps_list, dtype=np.float32),
                        "bbox": (x_min, y_min, x_max, y_max),
                        "pose_score": 0.9, "3d_pos": pelvis_3d, "head_2d": head_2d
                    })

            elif pose_model and (frame_idx % (INFER_EVERY_N_FRAMES + 1) == 0):
                # Use MoveNet fallback
                small_image = cv2.resize(image, (int(w * FRAME_DOWNSCALE), int(h * FRAME_DOWNSCALE)))
                pose_output = pose_model.infer(small_image)
                all_raw_detections = parse_movenet_multipose(pose_output, w, h)
                
                # DEBUG: Draw all raw detections in blue
                for det in all_raw_detections:
                    x_min, y_min, x_max, y_max = det["bbox"]
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 100, 100), 1)
                    score = det["pose_score"]
                    cv2.putText(overlay, f"{score:.2f}", (x_min, max(0, y_min - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            
            # --- 2. Decide which detections to process ---
            detections_to_process = []
            if locked_target_id is None:
                # No lock: only look for people inside the ROI
                for det in all_raw_detections:
                    px, py = RobustTracker._centroid_from_bbox(det["bbox"])
                    if is_inside_roi(px, py, bench_roi):
                        detections_to_process.append(det)
            else:
                # Lock acquired: track everyone in the frame to follow the target
                detections_to_process = all_raw_detections

            # --- 3. Add 3D data and prepare for tracker ---
            detections = []
            for det in detections_to_process:
                # Add depth data if not already present (from MoveNet)
                if "3d_pos" not in det:
                    x_min, y_min, x_max, y_max = det["bbox"]
                    
                    # Define a central region to sample depth from (more robust)
                    w_box, h_box = x_max - x_min, y_max - y_min
                    cx_box, cy_box = x_min + w_box // 2, y_min + h_box // 2
                    
                    sample_w, sample_h = int(w_box * 0.5), int(h_box * 0.8)
                    x_start = max(0, cx_box - sample_w // 2)
                    y_start = max(0, cy_box - sample_h // 2)
                    x_end = min(depth.shape[1], x_start + sample_w)
                    y_end = min(depth.shape[0], y_start + sample_h)

                    depth_roi = depth[y_start:y_end, x_start:x_end]
                    valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
                    
                    if valid_depths.size > 0:
                        z_depth = np.median(valid_depths)
                        
                        px, py = RobustTracker._centroid_from_bbox(det["bbox"])
                        x_3d = (px - cx) * z_depth / fx
                        y_3d = (py - cy) * z_depth / fy
                        det["3d_pos"] = (x_3d, y_3d, z_depth)
                        
                        if det["keypoints"][0, 2] > 0.3:
                            hx, hy = int(det["keypoints"][0, 0]), int(det["keypoints"][0, 1])
                            if 0 <= hy < depth.shape[0] and 0 <= hx < depth.shape[1]:
                                det["head_2d"] = (hx, hy)
                detections.append(det)

            # Update tracker
            tracks = tracker.update(detections, frame_idx)
            
            # Check locks
            if locked_target_id is not None:
                # Release lock if person disappeared
                if locked_target_id not in tracks:
                    # Person disappeared
                    if lock_start_time:
                        duration = current_time - lock_start_time
                        lock_events.append({
                            'timestamp': datetime.now().isoformat(),
                            'person_id': locked_target_id,
                            'duration': duration
                        })
                        print(f"[INFO] Released lock on ID {locked_target_id} (disappeared), duration: {duration:.2f}s")
                    locked_target_id = None
                    lock_start_time = None

            
            if locked_target_id is None:
                # Look for new person to lock
                sitting_timers = {tid: t for tid, t in sitting_timers.items() if tid in tracks}
                
                for tid, tr in tracks.items():
                    # --- Grace Period Logic ---
                    # Update sitting grace period
                    if is_sitting(tr):
                        tr['sitting_grace'] = 5  # Reset to 5 frames of grace
                    else:
                        tr['sitting_grace'] = max(0, tr.get('sitting_grace', 0) - 1)
                        
                    # Update facing grace period
                    _, is_facing = face_detector.get_face_yaw(image, tr["bbox"])
                    if is_facing:
                        tr['facing_grace'] = 5  # Reset to 5 frames of grace
                    else:
                        tr['facing_grace'] = max(0, tr.get('facing_grace', 0) - 1)

                    # Check for lock condition using grace periods
                    is_reliably_sitting = tr['sitting_grace'] > 0
                    is_reliably_facing = tr['facing_grace'] > 0

                    if is_reliably_sitting and is_reliably_facing:
                        sitting_timers[tid] = sitting_timers.get(tid, 0.0) + dt
                        
                        if sitting_timers[tid] >= MIN_LOCK_SIT_TIME:
                            locked_target_id = tid
                            lock_start_time = current_time
                            print(f"[INFO] Locked on ID {tid} (sitting on bench)")
                            break 
                    else:
                        # Reset timer if conditions are not reliably met
                        sitting_timers[tid] = 0.0
            
            # Update eyes
            if locked_target_id and locked_target_id in tracks:
                tr = tracks[locked_target_id]
                head_2d = tr.get("head_2d")
                if head_2d:
                    target_x, target_y = head_2d[0], head_2d[1]
                else:
                    cx, cy = tr["centroid"]
                    target_x, target_y = cx, cy
                
                eyes.update_target(target_x, target_y, w, h)
            else:
                # Idle animation
                eyes.update_target(EYES_CANVAS_WIDTH // 2, EYES_CANVAS_HEIGHT // 2,
                                 EYES_CANVAS_WIDTH, EYES_CANVAS_HEIGHT)
            
            eyes_canvas = eyes.render()
            
            # Draw overlay
            
            
            # Draw bench ROI (2D rectangle in pixel coordinates)
            cv2.rectangle(overlay, (bx_min, by_min), (bx_max, by_max), (255, 0, 255), 2)
            cv2.putText(overlay, "Bench ROI", (bx_min, max(0, by_min - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            
            # Draw tracks
            for tid, tr in tracks.items():
                color = (0, 255, 255) if tid == locked_target_id else (0, 255, 0)
                x_min, y_min, x_max, y_max = tr["bbox"]
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
                
                cx, cy = tr["centroid"]
                cv2.circle(overlay, (int(cx), int(cy)), 5, color, -1)
                
                label = f"ID:{tid}"

                sit_grace = tr.get('sitting_grace', 0)
                face_grace = tr.get('facing_grace', 0)
                label += f" (Sit:{sit_grace}, Face:{face_grace})"

                if tr.get("3d_pos"):
                    _, _, z = tr["3d_pos"]
                    label += f" ({z:.2f}m)"
                
                # Show sitting timer progress
                if tid in sitting_timers and sitting_timers[tid] > 0:
                    label += f" | T:{sitting_timers[tid]:.1f}s"
                
                if tid == locked_target_id:
                    label += " [LOCKED]"

                cv2.putText(overlay, label, (x_min, max(0, y_min - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
            # Status text
            if locked_target_id:
                status_text = f"LOCKED ID: {locked_target_id} - TRACKING BENCH SITTER"
                color = (0, 255, 0)
            else:
                status_text = f"WAITING FOR BENCH SITTER (ROI: {len(tracks)} person(s) in bench area)"
                color = (128, 128, 128)
            cv2.putText(overlay, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # FPS
            frame_count_for_fps += 1
            if current_time - last_fps_time >= 1.0:
                fps = frame_count_for_fps / (current_time - last_fps_time)
                last_fps_time = current_time
                frame_count_for_fps = 0
            
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Camera Feed", overlay)
            cv2.imshow("Eyes Feed", eyes_canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        # Save logs
        if lock_events:
            csv_path = "./logs/locks.csv"
            with open(csv_path, 'w') as f:
                f.write("timestamp,person_id,duration_seconds\n")
                for event in lock_events:
                    f.write(f"{event['timestamp']},{event['person_id']},{event['duration']:.2f}\n")
            print(f"[INFO] Saved {len(lock_events)} lock events to {csv_path}")
        
        if locked_target_id and lock_start_time:
            duration = time.time() - lock_start_time
            print(f"[INFO] Final lock duration: {duration:.2f}s")
        
        face_detector.close()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Padayani Eye Tracking Prototype")
    parser.add_argument("--test-sim", action="store_true",
                       help="Run in test simulation mode (no camera required)")
    args = parser.parse_args()
    
    main(test_sim=args.test_sim)
