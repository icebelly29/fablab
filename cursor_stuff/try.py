"""
Padayani Eye Tracking Prototype - ZED 1 Camera Version
=======================================================

README
------
This prototype detects and tracks people sitting on a bench using a ZED 1 stereo camera.
When a person sits and faces the installation for MIN_LOCK_SIT_TIME seconds, the system
locks onto them and three pairs of animated eyes follow their head position.

Requirements:
- Windows 10/11 with CUDA 11.7 installed
- ZED SDK (compatible with ZED 1) installed
- Python 3.9 or 3.10

Installation:
    pip install opencv-python numpy mediapipe pyzed tflite-runtime imutils

Note: If pyzed is not available via pip, download ZED SDK for Windows and set PYTHONPATH:
    set PYTHONPATH=C:\Program Files (x86)\ZED SDK\bin;%PYTHONPATH%
    
    Or on Windows PowerShell:
    $env:PYTHONPATH = "C:\Program Files (x86)\ZED SDK\bin"

Verify CUDA availability in ZED SDK:
    Check ZED SDK tools or verify CUDA is accessible by ZED runtime.

Usage:
    python padayani_zed_prototype.py
    
    Test mode (no camera required):
    python padayani_zed_prototype.py --test-sim

Expected Behavior:
    - Two windows open: "Camera Feed" (ZED image with overlays) and "Eyes Feed" (animated eyes)
    - When someone sits on the bench and faces the installation for 2 seconds, eyes lock and track them
    - Lock events are logged to ./logs/locks.csv
    - Press 'q' to quit

Output:
    - CSV logs: ./logs/locks.csv (timestamp, tracked_id, lock_duration)
    - Visual feedback in both OpenCV windows
"""

import cv2
import numpy as np
import time
import csv
import os
import argparse
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Try to import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("ERROR: pyzed.sl not available. Install ZED SDK and set PYTHONPATH.")

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available. Face orientation checks will be disabled.")

# Try to import TFLite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("WARNING: TFLite not available. Will rely on ZED body tracking only.")


# ============================================================================
# CONFIGURATION CONSTANTS - EDIT THESE TO ADJUST BEHAVIOR
# ============================================================================

# Bench ROI in 3D world coordinates (meters) relative to ZED camera
# Z is depth (forward from camera), X is left-right, Y is vertical (up)
BENCH_DEPTH_M = 2.8              # Depth of bench center from camera
BENCH_DEPTH_TOLERANCE = 0.15     # ±tolerance around bench depth
BENCH_X_MIN_M = -0.6             # Left edge of bench
BENCH_X_MAX_M = 0.6              # Right edge of bench
BENCH_Y_MIN_M = -0.2             # Bottom of seat (relative to camera height)
BENCH_Y_MAX_M = 0.3              # Top of seat

# Lock timing parameters (seconds)
MIN_LOCK_SIT_TIME = 2.0          # Continuous sitting time required to lock
ID_PERSISTENCE_TIMEOUT = 1.2     # How long to maintain lock if person disappears
STAND_RESET_TIME = 0.5           # If person stands for this long, reset sitting timer

# Pose detection settings
MODEL_PATH = "./models/movenet_multipose_lightning.tflite"
FRAME_DOWNSCALE = 0.5            # Downscale factor for MoveNet inference (1.0 = full res)
MAX_PEOPLE = 6                   # Maximum people to track simultaneously
INFERENCE_SKIP_FRAMES = 2        # Run inference every N frames (1 = every frame)

# Camera settings
ZED_RESOLUTION = sl.RESOLUTION.HD720  # Options: HD720, HD1080, HD2K
ZED_FPS = 30
ZED_DEPTH_MODE = sl.DEPTH_MODE.ULTRA  # ULTRA, QUALITY, PERFORMANCE

# Display settings
EYES_CANVAS_WIDTH = 640
EYES_CANVAS_HEIGHT = 480
DISPLAY_SCALE = 1.0              # Scale camera feed display (1.0 = native)

# Sitting detection heuristics (tune these based on bench height)
SEAT_HEIGHT_TOLERANCE = 0.1      # ±tolerance around expected seat Y coordinate
KNEE_TO_HIP_TOLERANCE = 0.2      # Allowable knee-to-hip vertical difference when sitting
TORSO_ANGLE_THRESH = 30.0        # Maximum torso angle from vertical (degrees)
BENCH_DEPTH_FILTER_SIZE = 5      # Median filter size for depth stabilization

# Face orientation thresholds (degrees)
FACE_YAW_THRESHOLD = 45.0        # Maximum face yaw angle (left/right rotation)
FACE_PITCH_THRESHOLD = 30.0      # Maximum face pitch angle (up/down)

# Eye tracking settings
EYE_SMOOTHING_ALPHA = 0.7        # Exponential smoothing (0.0-1.0, higher = smoother)
IDLE_ANIMATION_SPEED = 0.02      # Speed of idle eye movement

# Preprocessing options
ENABLE_CLAHE = False             # Enable CLAHE for contrast adjustment
ENABLE_BRIGHTNESS_CLAMP = False  # Clamp brightness values
GAMMA_CORRECTION = 1.0           # Gamma correction (1.0 = no correction)

# File paths
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "locks.csv")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_log_dir():
    """Create logs directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize CSV with headers if file doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'tracked_id', 'lock_duration_s'])


def log_lock_event(tracked_id: int, lock_duration: float):
    """Log a lock event to CSV file."""
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, tracked_id, lock_duration])


def project_3d_to_2d(point_3d: np.ndarray, camera_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Project 3D world point to 2D image coordinates.
    
    Args:
        point_3d: [x, y, z] in camera coordinate system
        camera_matrix: 3x3 camera intrinsic matrix
        
    Returns:
        (u, v) pixel coordinates
    """
    if len(point_3d) < 3 or point_3d[2] <= 0:
        return (0, 0)
    
    # Project to normalized coordinates
    x_norm = point_3d[0] / point_3d[2]
    y_norm = point_3d[1] / point_3d[2]
    
    # Apply camera matrix
    u = int(camera_matrix[0, 0] * x_norm + camera_matrix[0, 2])
    v = int(camera_matrix[1, 1] * y_norm + camera_matrix[1, 2])
    
    return (u, v)


def smooth_interpolate(current: float, target: float, alpha: float) -> float:
    """
    Exponential smoothing interpolation.
    
    Args:
        current: Current value
        target: Target value
        alpha: Smoothing factor (0.0-1.0), higher = more smoothing
        
    Returns:
        Smoothed value
    """
    return current * alpha + target * (1.0 - alpha)


def check_sitting_pose_zed(skeleton: sl.BodyData, bench_y: float) -> Tuple[bool, str]:
    """
    Check if person is sitting using ZED skeleton data.
    
    Args:
        skeleton: ZED BodyData object with skeleton joints
        bench_y: Expected Y coordinate of bench seat in camera coordinates
        
    Returns:
        (is_sitting, reason_string)
    """
    # Get key joints (these indices depend on ZED skeleton model)
    # ZED typically uses COCO or BODY_38 format
    # Adjust joint indices based on your ZED SDK version
    
    try:
        # Try to access keypoint positions
        # ZED skeleton format varies - adapt these indices
        hip_center_idx = 11  # Approximate - adjust for your ZED version
        knee_left_idx = 13
        knee_right_idx = 14
        shoulder_left_idx = 5
        shoulder_right_idx = 6
        
        # Get joint positions in 3D
        joints = skeleton.keypoint
        if len(joints) < 15:
            return False, "insufficient_joints"
        
        hip_pos = np.array([joints[hip_center_idx][0], joints[hip_center_idx][1], joints[hip_center_idx][2]])
        knee_left = np.array([joints[knee_left_idx][0], joints[knee_left_idx][1], joints[knee_left_idx][2]])
        knee_right = np.array([joints[knee_right_idx][0], joints[knee_right_idx][1], joints[knee_right_idx][2]])
        shoulder_left = np.array([joints[shoulder_left_idx][0], joints[shoulder_left_idx][1], joints[shoulder_left_idx][2]])
        shoulder_right = np.array([joints[shoulder_right_idx][0], joints[shoulder_right_idx][1], joints[shoulder_right_idx][2]])
        
        # Check 1: Hip Y is near seat height
        if abs(hip_pos[1] - bench_y) > SEAT_HEIGHT_TOLERANCE:
            return False, f"hip_height_mismatch (hip_y={hip_pos[1]:.2f}, bench_y={bench_y:.2f})"
        
        # Check 2: Knees are below or at hip level (sitting posture)
        knee_avg_y = (knee_left[1] + knee_right[1]) / 2.0
        if knee_avg_y > hip_pos[1] + KNEE_TO_HIP_TOLERANCE:
            return False, f"knee_position (knee_y={knee_avg_y:.2f} > hip_y={hip_pos[1]:.2f})"
        
        # Check 3: Torso is reasonably upright
        shoulder_center = (shoulder_left + shoulder_right) / 2.0
        torso_vec = shoulder_center - hip_pos
        torso_angle = np.degrees(np.arctan2(abs(torso_vec[0]), abs(torso_vec[1])))
        
        if torso_angle > TORSO_ANGLE_THRESH:
            return False, f"torso_angle ({torso_angle:.1f}deg > {TORSO_ANGLE_THRESH}deg)"
        
        return True, "sitting_detected"
        
    except (AttributeError, IndexError) as e:
        return False, f"skeleton_error: {str(e)}"


def check_sitting_pose_movenet(keypoints: np.ndarray, depth: float, bbox_height: int) -> Tuple[bool, str]:
    """
    Check if person is sitting using MoveNet keypoints and depth.
    
    Args:
        keypoints: Array of keypoints from MoveNet (shape: [num_people, num_keypoints, 3])
                  Last dimension: [x, y, confidence]
        depth: Depth value at person centroid (meters)
        bbox_height: Bounding box height in pixels
        
    Returns:
        (is_sitting, reason_string)
    """
    if len(keypoints) == 0:
        return False, "no_keypoints"
    
    # MoveNet multipose keypoint indices (COCO format)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    person_kp = keypoints[0]  # Use first person
    if len(person_kp) < 17:
        return False, "insufficient_keypoints"
    
    left_hip = person_kp[11]
    right_hip = person_kp[12]
    left_knee = person_kp[13]
    right_knee = person_kp[14]
    
    # Check confidence
    min_confidence = 0.3
    if (left_hip[2] < min_confidence or right_hip[2] < min_confidence or
        left_knee[2] < min_confidence or right_knee[2] < min_confidence):
        return False, "low_keypoint_confidence"
    
    # Hip center
    hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
    knee_avg_y = (left_knee[1] + right_knee[1]) / 2.0
    
    # Heuristic: Sitting if knees are near or above hips in image
    # Also check if bbox height suggests sitting (sitting person is shorter)
    sitting_threshold_pixels = 400  # Adjust based on your camera setup
    
    if knee_avg_y < hip_center_y - 50:  # Knees significantly above hips = likely standing
        return False, f"knee_above_hip (knee_y={knee_avg_y:.0f}, hip_y={hip_center_y:.0f})"
    
    if bbox_height > sitting_threshold_pixels:
        return False, f"bbox_too_tall (height={bbox_height})"
    
    # Depth check: person should be near bench depth
    if abs(depth - BENCH_DEPTH_M) > BENCH_DEPTH_TOLERANCE:
        return False, f"depth_mismatch (depth={depth:.2f}m, expected={BENCH_DEPTH_M:.2f}m)"
    
    return True, "sitting_detected_movenet"


def check_bench_roi(point_3d: np.ndarray) -> bool:
    """
    Check if a 3D point is within the bench ROI.
    
    Args:
        point_3d: [x, y, z] in camera coordinates (meters)
        
    Returns:
        True if point is in ROI
    """
    x, y, z = point_3d[0], point_3d[1], point_3d[2]
    
    # Check depth (Z)
    if z < (BENCH_DEPTH_M - BENCH_DEPTH_TOLERANCE) or z > (BENCH_DEPTH_M + BENCH_DEPTH_TOLERANCE):
        return False
    
    # Check X (left-right)
    if x < BENCH_X_MIN_M or x > BENCH_X_MAX_M:
        return False
    
    # Check Y (vertical)
    if y < BENCH_Y_MIN_M or y > BENCH_Y_MAX_M:
        return False
    
    return True


def get_face_yaw_mediapipe(face_landmarks, image_shape) -> Optional[float]:
    """
    Estimate face yaw angle using MediaPipe face mesh.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        image_shape: (height, width) of image
        
    Returns:
        Yaw angle in degrees (negative = left, positive = right), or None if unavailable
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    # MediaPipe face mesh landmark indices
    # 1: left eye, 33: right eye, 4: nose tip
    nose_tip = face_landmarks.landmark[4]
    left_eye = face_landmarks.landmark[1]
    right_eye = face_landmarks.landmark[33]
    
    # Convert to pixel coordinates
    h, w = image_shape[:2]
    nose_x = nose_tip.x * w
    left_eye_x = left_eye.x * w
    right_eye_x = right_eye.x * w
    
    # Estimate yaw from eye-nose geometry
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    offset_x = nose_x - eye_center_x
    
    # Rough estimate: ~1 degree per pixel at typical distance (adjust scaling)
    yaw_estimate = offset_x * 0.1  # Scale factor - tune based on camera setup
    
    return yaw_estimate


def draw_eyes_canvas(canvas: np.ndarray, target_pos: Optional[Tuple[float, float]], 
                    locked: bool, eye_pairs: List[Tuple[int, int, int, int]]):
    """
    Draw three pairs of cartoon eyes on canvas.
    
    Args:
        canvas: OpenCV image to draw on (BGR format)
        target_pos: (x, y) normalized position (0-1) of target head, or None for idle
        locked: Whether system is currently locked
        eye_pairs: List of (x1, y1, x2, y2) for each eye pair center positions
    """
    canvas[:] = (240, 240, 240)  # Light gray background
    
    if target_pos is None or not locked:
        # Idle animation: slow wandering eyes
        import math
        t = time.time() * IDLE_ANIMATION_SPEED
        idle_offset_x = math.sin(t) * 0.15
        idle_offset_y = math.cos(t * 0.7) * 0.1
        target_pos = (0.5 + idle_offset_x, 0.5 + idle_offset_y)
    
    # Map normalized target position to canvas
    target_x = int(target_pos[0] * canvas.shape[1])
    target_y = int(target_pos[1] * canvas.shape[0])
    
    # Draw each eye pair
    for eye_x1, eye_y1, eye_x2, eye_y2 in eye_pairs:
        eye_radius = 30
        pupil_radius = 12
        
        # Convert to canvas coordinates
        cx1 = int(eye_x1 * canvas.shape[1])
        cy1 = int(eye_y1 * canvas.shape[0])
        cx2 = int(eye_x2 * canvas.shape[1])
        cy2 = int(eye_y2 * canvas.shape[0])
        
        # Draw eye whites
        cv2.circle(canvas, (cx1, cy1), eye_radius, (255, 255, 255), -1)
        cv2.circle(canvas, (cx2, cy2), eye_radius, (255, 255, 255), -1)
        cv2.circle(canvas, (cx1, cy1), eye_radius, (0, 0, 0), 2)
        cv2.circle(canvas, (cx2, cy2), eye_radius, (0, 0, 0), 2)
        
        # Calculate pupil position (look toward target)
        for cx, cy in [(cx1, cy1), (cx2, cy2)]:
            dx = target_x - cx
            dy = target_y - cy
            dist = max(1, np.sqrt(dx*dx + dy*dy))
            
            # Limit pupil movement to eye radius
            max_offset = eye_radius - pupil_radius - 2
            offset_x = dx / dist * min(abs(dx), max_offset)
            offset_y = dy / dist * min(abs(dy), max_offset)
            
            pupil_x = int(cx + offset_x)
            pupil_y = int(cy + offset_y)
            
            # Draw pupil
            cv2.circle(canvas, (pupil_x, pupil_y), pupil_radius, (0, 0, 0), -1)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to frame (CLAHE, brightness clamp, gamma).
    
    Args:
        frame: Input BGR frame
        
    Returns:
        Processed frame
    """
    processed = frame.copy()
    
    # Gamma correction
    if GAMMA_CORRECTION != 1.0:
        inv_gamma = 1.0 / GAMMA_CORRECTION
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if ENABLE_CLAHE:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Brightness clamp
    if ENABLE_BRIGHTNESS_CLAMP:
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    return processed


# ============================================================================
# TRACKER CLASS
# ============================================================================

class PersonTracker:
    """Tracks individual people and their lock eligibility."""
    
    def __init__(self, person_id: int):
        self.person_id = person_id
        self.last_seen = time.time()
        self.sitting_time = 0.0
        self.standing_time = 0.0
        self.is_locked = False
        self.lock_start_time = None
        self.position_3d = None
        self.face_yaw = None
        self.in_roi = False
        self.sitting_reason = ""
        
    def update(self, position_3d: np.ndarray, is_sitting: bool, 
              sitting_reason: str, face_yaw: Optional[float], in_roi: bool):
        """Update tracker with new observations."""
        current_time = time.time()
        dt = current_time - self.last_seen
        self.last_seen = current_time
        
        self.position_3d = position_3d
        self.face_yaw = face_yaw
        self.in_roi = in_roi
        self.sitting_reason = sitting_reason
        
        if is_sitting and in_roi:
            self.sitting_time += dt
            self.standing_time = 0.0
        else:
            self.sitting_time = 0.0
            if not is_sitting:
                self.standing_time += dt
        
        # Reset sitting timer if person stood up
        if self.standing_time > STAND_RESET_TIME:
            self.sitting_time = 0.0
        
        # Check lock eligibility
        if not self.is_locked:
            if (self.sitting_time >= MIN_LOCK_SIT_TIME and 
                in_roi and 
                (face_yaw is None or abs(face_yaw) < FACE_YAW_THRESHOLD)):
                self.is_locked = True
                self.lock_start_time = current_time
                print(f"[LOCK] Person {self.person_id} locked! (sitting_time={self.sitting_time:.2f}s)")
    
    def should_maintain_lock(self) -> bool:
        """Check if lock should be maintained."""
        if not self.is_locked:
            return False
        
        # Maintain lock if person disappeared recently or is still in ROI
        time_since_seen = time.time() - self.last_seen
        if time_since_seen < ID_PERSISTENCE_TIMEOUT:
            return True
        
        if self.in_roi:
            return True
        
        return False
    
    def release_lock(self) -> float:
        """Release lock and return lock duration."""
        if not self.is_locked:
            return 0.0
        
        duration = time.time() - self.lock_start_time
        self.is_locked = False
        self.lock_start_time = None
        print(f"[UNLOCK] Person {self.person_id} released (duration={duration:.2f}s)")
        return duration


# ============================================================================
# MOVE NET MODEL LOADER
# ============================================================================

class MoveNetMultipose:
    """Wrapper for MoveNet Multipose TFLite model."""
    
    def __init__(self, model_path: str):
        if not TFLITE_AVAILABLE:
            raise ImportError("TFLite not available")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Input shape: [1, height, width, 3]
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        print(f"MoveNet loaded: input size {self.input_width}x{self.input_height}")
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on image.
        
        Args:
            image: BGR image
            
        Returns:
            Array of detections [num_people, 6, 17, 3]
            Format: [x, y, confidence] for 17 keypoints per person
        """
        # Resize image
        input_image = cv2.resize(image, (self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.uint8)
        
        # Prepare input
        input_data = np.expand_dims(input_image, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data[0]  # Shape: [num_people, 6, 17, 3]


# ============================================================================
# ZED CAMERA MANAGER
# ============================================================================

class ZEDCamera:
    """Manages ZED camera initialization and frame capture."""
    
    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        if not ZED_AVAILABLE:
            raise RuntimeError("ZED SDK not available")
        
        self.camera = sl.Camera()
        self.runtime_parameters = sl.RuntimeParameters()
        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Init parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        init_params.depth_mode = ZED_DEPTH_MODE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        
        # Enable body tracking if available
        self.body_tracking_available = False
        self.body_tracker = None
        
        try:
            # Try to enable body tracking
            body_tracking_params = sl.BodyTrackingParameters()
            body_tracking_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
            body_tracking_params.body_format = sl.BODY_FORMAT.BODY_38  # or BODY_18/34 depending on SDK
            
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED camera open failed: {status}")
            
            # Enable body tracking
            self.body_tracker = sl.BodyTracker()
            body_tracking_params.enable_body_fitting = True
            body_tracking_params.enable_tracking = True
            
            self.body_tracker.init(body_tracking_params)
            self.body_tracking_available = True
            print("[ZED] Body tracking enabled")
            
        except Exception as e:
            print(f"[ZED] Body tracking not available: {e}")
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED camera open failed: {status}")
        
        # Get camera info
        cam_info = self.camera.get_camera_information()
        print(f"[ZED] Camera opened: {cam_info.camera_model}")
        print(f"[ZED] Resolution: {cam_info.camera_resolution.width}x{cam_info.camera_resolution.height}")
        
        # Get camera intrinsics
        calibration = cam_info.camera_configuration.calibration_parameters.left_cam
        self.camera_matrix = np.array([
            [calibration.fx, 0, calibration.cx],
            [0, calibration.fy, calibration.cy],
            [0, 0, 1]
        ])
        
        self.width = cam_info.camera_resolution.width
        self.height = cam_info.camera_resolution.height
    
    def grab_frame(self) -> bool:
        """Grab a new frame. Returns True if successful."""
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.camera.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            return True
        return False
    
    def get_image(self) -> np.ndarray:
        """Get left color image as numpy array."""
        return self.image_left.get_data()
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get depth value at pixel (x, y) in meters."""
        depth_value = self.depth_map.get_value(x, y)[1]
        if np.isnan(depth_value) or np.isinf(depth_value):
            return 0.0
        return depth_value
    
    def get_3d_point(self, x: int, y: int) -> Optional[np.ndarray]:
        """Get 3D point at pixel (x, y) in camera coordinates."""
        point = self.point_cloud.get_value(x, y)[1]
        if np.isnan(point[0]) or np.isinf(point[0]):
            return None
        return np.array([point[0], point[1], point[2]])
    
    def get_bodies(self) -> List[sl.BodyData]:
        """Get tracked bodies from body tracker."""
        if not self.body_tracking_available:
            return []
        
        bodies = sl.Bodies()
        if self.body_tracker.retrieve_bodies(bodies) == sl.ERROR_CODE.SUCCESS:
            return bodies.body_list
        return []
    
    def close(self):
        """Close camera and release resources."""
        if self.body_tracker is not None:
            self.body_tracker.close()
        self.camera.close()
        print("[ZED] Camera closed")


# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class PadayaniEyeTracker:
    """Main application class."""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.zed_camera = None
        self.movenet = None
        self.face_mesh = None
        self.trackers: Dict[int, PersonTracker] = {}
        self.locked_person_id = None
        self.frame_count = 0
        
        # Smooth eye position
        self.eye_target_x = 0.5
        self.eye_target_y = 0.5
        
        # Eye pair positions (normalized 0-1)
        self.eye_pairs = [
            (0.25, 0.4, 0.35, 0.4),  # Left pair
            (0.5, 0.4, 0.6, 0.4),    # Center pair
            (0.65, 0.4, 0.75, 0.4),  # Right pair
        ]
        
        # Depth filter for bench
        self.depth_history = deque(maxlen=BENCH_DEPTH_FILTER_SIZE)
        
        create_log_dir()
        
        if not test_mode:
            self.init_camera()
        self.init_models()
    
    def init_camera(self):
        """Initialize ZED camera."""
        try:
            self.zed_camera = ZEDCamera(resolution=ZED_RESOLUTION, fps=ZED_FPS)
            print("[INIT] ZED camera initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ZED camera: {e}")
            if not ZED_AVAILABLE:
                print("[ERROR] ZED SDK not available. Install ZED SDK and set PYTHONPATH.")
            raise
    
    def init_models(self):
        """Initialize ML models."""
        # MoveNet
        if TFLITE_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self.movenet = MoveNetMultipose(MODEL_PATH)
                print("[INIT] MoveNet model loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load MoveNet: {e}")
        
        # MediaPipe Face Mesh
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=MAX_PEOPLE,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[INIT] MediaPipe Face Mesh initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize MediaPipe: {e}")
    
    def process_frame_zed_bodies(self, bodies: List[sl.BodyData]) -> Dict[int, Dict]:
        """Process ZED body tracking data."""
        results = {}
        bench_y = BENCH_Y_MIN_M + (BENCH_Y_MAX_M - BENCH_Y_MIN_M) / 2.0  # Approximate seat height
        
        for body in bodies:
            body_id = body.id
            if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                continue
            
            # Get 3D position (typically pelvis/hip center)
            # ZED provides root keypoint position
            try:
                position_3d = np.array([body.position[0], body.position[1], body.position[2]])
            except:
                continue
            
            # Check ROI
            in_roi = check_bench_roi(position_3d)
            
            # Check sitting pose
            is_sitting, sitting_reason = check_sitting_pose_zed(body, bench_y)
            
            # Get face yaw from skeleton if available, or use MediaPipe later
            face_yaw = None
            
            results[body_id] = {
                'position_3d': position_3d,
                'is_sitting': is_sitting,
                'sitting_reason': sitting_reason,
                'face_yaw': face_yaw,
                'in_roi': in_roi,
                'skeleton': body
            }
        
        return results
    
    def process_frame_movenet(self, image: np.ndarray) -> Dict[int, Dict]:
        """Process frame using MoveNet."""
        if self.movenet is None:
            return {}
        
        # Downscale for inference
        h, w = image.shape[:2]
        small_h = int(h * FRAME_DOWNSCALE)
        small_w = int(w * FRAME_DOWNSCALE)
        small_image = cv2.resize(image, (small_w, small_h))
        
        # Run MoveNet
        detections = self.movenet.detect(small_image)
        
        results = {}
        
        # Process each detected person
        for i, detection in enumerate(detections):
            # MoveNet multipose output format: [num_people, 6, 17, 3]
            # First 4 values are bbox, then 17 keypoints
            if len(detection) < 6:
                continue
            
            # Get bounding box
            bbox = detection[0:4]  # [ymin, xmin, ymax, xmax] in normalized coords
            keypoints = detection[4:]  # 17 keypoints
            
            # Scale bbox to original image size
            ymin = int(bbox[0] * h)
            xmin = int(bbox[1] * w)
            ymax = int(bbox[2] * h)
            xmax = int(bbox[3] * w)
            
            bbox_height = ymax - ymin
            bbox_center_x = (xmin + xmax) // 2
            bbox_center_y = (ymin + ymax) // 2
            
            # Get depth at person center
            depth = self.zed_camera.get_depth_at_point(bbox_center_x, bbox_center_y)
            if depth <= 0:
                continue
            
            # Get 3D position
            point_3d = self.zed_camera.get_3d_point(bbox_center_x, bbox_center_y)
            if point_3d is None:
                continue
            
            # Scale keypoints to original image
            scaled_keypoints = keypoints.copy()
            for kp in scaled_keypoints:
                kp[0] *= w  # x
                kp[1] *= h  # y
            
            # Check ROI
            in_roi = check_bench_roi(point_3d)
            
            # Check sitting pose
            is_sitting, sitting_reason = check_sitting_pose_movenet(scaled_keypoints, depth, bbox_height)
            
            # Use centroid matching for ID (simple approach)
            person_id = i  # Could use more sophisticated tracking
            
            results[person_id] = {
                'position_3d': point_3d,
                'is_sitting': is_sitting,
                'sitting_reason': sitting_reason,
                'face_yaw': None,  # Will be updated with MediaPipe
                'in_roi': in_roi,
                'bbox': (xmin, ymin, xmax, ymax),
                'keypoints': scaled_keypoints
            }
        
        return results
    
    def process_face_orientation(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[float]:
        """Process face orientation using MediaPipe."""
        if self.face_mesh is None:
            return None
        
        # Crop to head region if bbox provided
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            # Expand bbox for head region (top portion)
            head_h = int((ymax - ymin) * 0.4)
            head_bbox = (xmin, max(0, ymin), xmax, min(image.shape[0], ymin + head_h))
            cropped = image[head_bbox[1]:head_bbox[3], head_bbox[0]:head_bbox[2]]
            if cropped.size == 0:
                cropped = image
        else:
            cropped = image
        
        # Run MediaPipe
        rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            return get_face_yaw_mediapipe(results.multi_face_landmarks[0], cropped.shape)
        
        return None
    
    def update_trackers(self, frame_results: Dict[int, Dict]):
        """Update person trackers with new frame results."""
        current_time = time.time()
        
        # Update existing trackers or create new ones
        for person_id, data in frame_results.items():
            if person_id not in self.trackers:
                self.trackers[person_id] = PersonTracker(person_id)
            
            tracker = self.trackers[person_id]
            tracker.update(
                position_3d=data['position_3d'],
                is_sitting=data['is_sitting'],
                sitting_reason=data['sitting_reason'],
                face_yaw=data['face_yaw'],
                in_roi=data['in_roi']
            )
        
        # Remove old trackers
        expired_ids = []
        for person_id, tracker in self.trackers.items():
            if current_time - tracker.last_seen > ID_PERSISTENCE_TIMEOUT * 2:
                if tracker.is_locked:
                    duration = tracker.release_lock()
                    log_lock_event(person_id, duration)
                expired_ids.append(person_id)
        
        for person_id in expired_ids:
            del self.trackers[person_id]
        
        # Update locked person
        if self.locked_person_id is not None:
            if self.locked_person_id not in self.trackers:
                # Lost locked person
                if self.locked_person_id in self.trackers:
                    duration = self.trackers[self.locked_person_id].release_lock()
                    log_lock_event(self.locked_person_id, duration)
                self.locked_person_id = None
            else:
                tracker = self.trackers[self.locked_person_id]
                if not tracker.should_maintain_lock():
                    # Release lock
                    duration = tracker.release_lock()
                    log_lock_event(self.locked_person_id, duration)
                    self.locked_person_id = None
        
        # Find new person to lock
        if self.locked_person_id is None:
            for person_id, tracker in self.trackers.items():
                if tracker.is_locked:
                    self.locked_person_id = person_id
                    break
    
    def draw_overlays(self, image: np.ndarray) -> np.ndarray:
        """Draw overlays on camera feed."""
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Draw bench ROI in image space
        # Project ROI corners to image
        roi_corners_3d = [
            [BENCH_X_MIN_M, BENCH_Y_MIN_M, BENCH_DEPTH_M - BENCH_DEPTH_TOLERANCE],
            [BENCH_X_MAX_M, BENCH_Y_MIN_M, BENCH_DEPTH_M - BENCH_DEPTH_TOLERANCE],
            [BENCH_X_MAX_M, BENCH_Y_MAX_M, BENCH_DEPTH_M - BENCH_DEPTH_TOLERANCE],
            [BENCH_X_MIN_M, BENCH_Y_MAX_M, BENCH_DEPTH_M - BENCH_DEPTH_TOLERANCE],
        ]
        
        roi_corners_2d = []
        for corner in roi_corners_3d:
            u, v = project_3d_to_2d(corner, self.zed_camera.camera_matrix)
            roi_corners_2d.append((u, v))
        
        # Draw ROI box
        if len(roi_corners_2d) == 4:
            pts = np.array(roi_corners_2d, np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            cv2.putText(overlay, "BENCH ROI", (roi_corners_2d[0][0], roi_corners_2d[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw tracked people
        locked_id = self.locked_person_id
        
        for person_id, tracker in self.trackers.items():
            if tracker.position_3d is None:
                continue
            
            # Project 3D position to 2D
            u, v = project_3d_to_2d(tracker.position_3d, self.zed_camera.camera_matrix)
            
            # Draw bounding box or skeleton
            is_locked = (person_id == locked_id)
            color = (0, 255, 0) if is_locked else (255, 0, 0)
            thickness = 3 if is_locked else 2
            
            cv2.circle(overlay, (u, v), 10, color, thickness)
            
            # Draw ID and status
            label = f"ID:{person_id}"
            if is_locked:
                label += " [LOCKED]"
            elif tracker.in_roi:
                label += f" [ROI]"
            
            cv2.putText(overlay, label, (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
            
            # Draw debug info
            info_y = v + 20
            if tracker.position_3d is not None:
                cv2.putText(overlay, f"3D: [{tracker.position_3d[0]:.2f}, {tracker.position_3d[1]:.2f}, {tracker.position_3d[2]:.2f}]",
                           (u + 15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                info_y += 15
            
            cv2.putText(overlay, f"sit_time: {tracker.sitting_time:.1f}s",
                       (u + 15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 15
            
            if tracker.face_yaw is not None:
                cv2.putText(overlay, f"yaw: {tracker.face_yaw:.1f}deg",
                           (u + 15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                info_y += 15
            
            cv2.putText(overlay, tracker.sitting_reason,
                       (u + 15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw status text
        status_y = 30
        cv2.putText(overlay, f"Frame: {self.frame_count}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 25
        
        if locked_id is not None:
            cv2.putText(overlay, f"LOCKED: Person {locked_id}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "Waiting for sitter...", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return overlay
    
    def update_eyes(self):
        """Update eye target position based on locked person."""
        if self.locked_person_id is None or self.zed_camera is None:
            target_pos = None
        else:
            tracker = self.trackers.get(self.locked_person_id)
            if tracker and tracker.position_3d is not None:
                # Project 3D head position to normalized 2D
                # Estimate head position (slightly above pelvis)
                head_3d = tracker.position_3d.copy()
                head_3d[1] += 0.2  # Head is ~20cm above pelvis
                
                u, v = project_3d_to_2d(head_3d, self.zed_camera.camera_matrix)
                
                # Normalize to 0-1
                h, w = self.zed_camera.height, self.zed_camera.width
                norm_x = max(0, min(1, u / w))
                norm_y = max(0, min(1, v / h))
                
                # Smooth interpolation
                self.eye_target_x = smooth_interpolate(self.eye_target_x, norm_x, EYE_SMOOTHING_ALPHA)
                self.eye_target_y = smooth_interpolate(self.eye_target_y, norm_y, EYE_SMOOTHING_ALPHA)
                
                target_pos = (self.eye_target_x, self.eye_target_y)
            else:
                target_pos = None
        
        # Draw eyes canvas
        eyes_canvas = np.zeros((EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8)
        draw_eyes_canvas(eyes_canvas, target_pos, self.locked_person_id is not None, self.eye_pairs)
        cv2.imshow("Eyes Feed", eyes_canvas)
    
    def run_test_sim(self):
        """Run in test simulation mode."""
        print("[TEST] Running in simulation mode (no camera required)")
        
        # Simulate a person sitting on bench
        import math
        
        eyes_canvas = np.zeros((EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        t = 0
        while True:
            # Simulate head position moving slowly
            target_x = 0.5 + 0.15 * math.sin(t * 0.05)
            target_y = 0.5 + 0.1 * math.cos(t * 0.03)
            target_pos = (target_x, target_y)
            
            # Simulate lock after 2 seconds
            locked = t > 2.0
            
            # Draw eyes
            draw_eyes_canvas(eyes_canvas, target_pos, locked, self.eye_pairs)
            cv2.imshow("Eyes Feed", eyes_canvas)
            
            # Draw test image
            test_image[:] = 50
            cv2.putText(test_image, "TEST MODE - No Camera", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(test_image, f"Time: {t:.1f}s | Locked: {locked}", (50, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera Feed", test_image)
            
            t += 0.033  # ~30 FPS
            
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop."""
        if self.test_mode:
            self.run_test_sim()
            return
        
        print("[START] Starting Padayani Eye Tracker")
        print("[INFO] Press 'q' to quit")
        
        try:
            while True:
                # Grab frame
                if not self.zed_camera.grab_frame():
                    continue
                
                self.frame_count += 1
                
                # Get image
                image = self.zed_camera.get_image()
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # ZED returns BGRA
                
                # Preprocess
                image = preprocess_frame(image)
                
                # Process frame
                frame_results = {}
                
                # Try ZED body tracking first
                if self.zed_camera.body_tracking_available:
                    bodies = self.zed_camera.get_bodies()
                    frame_results = self.process_frame_zed_bodies(bodies)
                else:
                    # Fall back to MoveNet
                    if self.frame_count % INFERENCE_SKIP_FRAMES == 0:
                        frame_results = self.process_frame_movenet(image)
                
                # Process face orientation with MediaPipe
                for person_id, data in frame_results.items():
                    if 'bbox' in data:
                        bbox = data['bbox']
                    else:
                        # Estimate bbox from 3D position
                        u, v = project_3d_to_2d(data['position_3d'], self.zed_camera.camera_matrix)
                        bbox_size = 100
                        bbox = (u - bbox_size//2, v - bbox_size//2, 
                               u + bbox_size//2, v + bbox_size//2)
                    
                    face_yaw = self.process_face_orientation(image, bbox)
                    data['face_yaw'] = face_yaw
                
                # Update trackers
                self.update_trackers(frame_results)
                
                # Draw overlays
                overlay = self.draw_overlays(image)
                
                # Resize for display if needed
                if DISPLAY_SCALE != 1.0:
                    h, w = overlay.shape[:2]
                    overlay = cv2.resize(overlay, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                
                cv2.imshow("Camera Feed", overlay)
                
                # Update eyes
                self.update_eyes()
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user")
        finally:
            # Cleanup
            if self.locked_person_id is not None:
                tracker = self.trackers.get(self.locked_person_id)
                if tracker:
                    duration = tracker.release_lock()
                    log_lock_event(self.locked_person_id, duration)
            
            if self.zed_camera:
                self.zed_camera.close()
            
            cv2.destroyAllWindows()
            print("[STOP] Application closed")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Padayani Eye Tracking Prototype")
    parser.add_argument('--test-sim', action='store_true',
                       help='Run in test simulation mode (no camera required)')
    args = parser.parse_args()
    
    # Check dependencies
    if not args.test_sim:
        if not ZED_AVAILABLE:
            print("[ERROR] ZED SDK not available. Use --test-sim for test mode.")
            return
    
    # Create and run application
    app = PadayaniEyeTracker(test_mode=args.test_sim)
    app.run()


if __name__ == "__main__":
    main()

"""
TODO and Notes:
---------------

1. Servo Control Integration:
   To drive physical servo motors for the eyes, add servo control in the draw_eyes_canvas()
   function. Map normalized eye target positions (0-1) to servo angles (e.g., 0-180 degrees).
   Example: servo_angle = int(eye_target_x * 180). Use a servo library like adafruit-circuitpython-servokit
   or pca9685 for PWM control.

2. Jetson/Raspberry Pi Porting:
   - Use TFLite GPU delegate on Jetson: interpreter.set_delegate(tflite.GpuDelegate())
   - Quantize MoveNet model to INT8 for better performance on edge devices
   - Reduce resolution and frame rate for lower-power devices
   - Consider using TensorRT on Jetson for optimized inference

3. ZED Body Tracking Fallback:
   The code already supports falling back to MoveNet when ZED body tracking is unavailable.
   To force MoveNet mode, set zed_camera.body_tracking_available = False after
"""