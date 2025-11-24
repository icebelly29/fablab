"""
Padayani Eye Tracker Prototype – Edge-Optimized
----------------------------------------------

This version is tuned for SMALL EDGE DEVICES (Jetson, mini PC, etc.):

- Uses MoveNet MultiPose Lightning (TFLite) with:
    * Optional tflite-runtime (preferred on edge) or TensorFlow's tf.lite.
    * Configurable NUM_TFLITE_THREADS.
    * Dynamic input resized to [1, 256, 256, 3].

- Edge-friendly choices:
    * Optional downscale of the camera/test video to <= TARGET_PROCESSING_WIDTH x TARGET_PROCESSING_HEIGHT
      (e.g., 960x540) so you don't process 4K directly.
    * Option to run pose inference only every N frames and reuse last result.
    * Lightweight noise reduction + brightness/glare mitigation.

- Two windows:
    * "Padayani Camera" – resized / processed feed with keypoints & tracking.
    * "Padayani Eyes"   – three pairs of perfectly circular eyes, pupils move ONLY left/right.

To use a 4096x2160 TEST VIDEO instead of webcam:
    - Set VIDEO_SOURCE = "your_4k_video.mp4"
    - Keep EDGE_DOWNSCALE = True (recommended) so we process at ~960x540 internally.

To use a CAMERA on an edge device:
    - Set VIDEO_SOURCE = 0 (or appropriate index).
    - Keep EDGE_DOWNSCALE = True and adjust CAP_WIDTH / CAP_HEIGHT if needed.

MoveNet MultiPose Lightning TFLite model:
    Download from TF Hub:
      https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1?lite-format=tflite
    Save it next to this script as:
      movenet_multipose_lightning.tflite

Install:
    pip install opencv-python numpy
    # On edge (recommended):
    pip install tflite-runtime
    # Or on a laptop:
    pip install tensorflow
"""
# ZED File path: C:\Program Files (x86)\ZED SDK\tools\

import cv2
import numpy as np
import time
import math

# ----------------------------
# Global configuration
# ----------------------------

# Path to MoveNet MultiPose Lightning model
MOVENET_MODEL_PATH = "movenet_multipose_lightning.tflite"

# Video source:
#   0           -> default webcam
#   "video.mp4" -> path to test video (e.g., 4096x2160)
VIDEO_SOURCE = 0  # change to your 4K video path when testing offline

# Selection mode: "single" (all three pairs follow one person) or "multi"
SELECTION_MODE = "multi"

# MoveNet input size (the TFLite model will be resized to this)
MOVENET_INPUT_SIZE = 256

# Edge-optimization switches
EDGE_DOWNSCALE = True          # downscale very large inputs (e.g., 4K) for processing
TARGET_PROCESSING_WIDTH = 960  # max width after downscale
TARGET_PROCESSING_HEIGHT = 540 # max height after downscale

INFER_EVERY_N_FRAMES = 2       # run MoveNet only every N frames (e.g., 2 -> ~half rate)
NUM_TFLITE_THREADS = 2         # number of CPU threads for TFLite interpreter

ENABLE_FACE_DETECTOR = True    # set False on tiny devices to save CPU (use body orientation only)

# Keypoint & instance thresholds
KEYPOINT_CONF_THRESH = 0.2
INSTANCE_SCORE_THRESH = 0.2
CORE_KEYPOINT_MIN_COUNT = 4  # nose / shoulders / hips, etc.

# Perimeter region configuration (normalized)
PERIMETER_X_MARGIN = 0.01   # central 70% width
PERIMETER_Y_MARGIN = 0.01   # central 80% height

# Bounding-box area threshold as fraction of frame area (approx distance / size)
# For 4096x2160, 0.005 ~= 0.5% of frame area.
MIN_BBOX_AREA_RATIO = 0.005

# Brightness / glare mitigation thresholds
BRIGHTNESS_LOW = 60
BRIGHTNESS_HIGH = 190
BRIGHTNESS_VERY_HIGH = 220

# Centroid tracker configuration
TRACKER_MAX_DISTANCE = 60
TRACKER_MAX_AGE = 30

# Eyes canvas configuration
EYES_CANVAS_WIDTH = 800
EYES_CANVAS_HEIGHT = 400
EYE_RADIUS = 40
PUPIL_RADIUS = 14
PUPIL_TRAVEL_PX = 25  # how far pupils can move left/right inside each eye

# Smoothing for pupil motion
SMOOTHING_ALPHA = 0.25

# Face detection (Haar cascade)
FACE_MIN_SIZE_PX = 40  # ignore faces smaller than this width/height


# ----------------------------
# Utility: load MoveNet interpreter (edge-friendly)
# ----------------------------

def load_movenet_interpreter(model_path):
    """
    Load MoveNet TFLite model using:
      - tflite_runtime.Interpreter (preferred on edge), or
      - TensorFlow's tf.lite.Interpreter
    and resize input to [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3].

    NUM_TFLITE_THREADS is used when the backend supports it.
    """
    interpreter = None

    # Try tflite-runtime first (common on edge devices)
    try:
        import tflite_runtime.interpreter as tflite
        try:
            interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=NUM_TFLITE_THREADS
            )
        except TypeError:
            interpreter = tflite.Interpreter(model_path=model_path)
            try:
                interpreter.set_num_threads(NUM_TFLITE_THREADS)
            except AttributeError:
                pass
    except ImportError:
        # Fall back to full TensorFlow
        import tensorflow as tf
        try:
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=NUM_TFLITE_THREADS
            )
        except TypeError:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            try:
                interpreter.set_num_threads(NUM_TFLITE_THREADS)
            except AttributeError:
                pass

    if interpreter is None:
        raise ImportError(
            "Could not create a TFLite Interpreter. "
            "Install either 'tflite-runtime' (preferred on edge) or 'tensorflow'."
        )

    # Resize dynamic input from [1, 1, 1, 3] to [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3]
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]

    interpreter.resize_tensor_input(
        input_index,
        [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3]
    )
    interpreter.allocate_tensors()
    return interpreter


# ----------------------------
# Preprocessing: glare, brightness, noise reduction
# ----------------------------

def apply_gamma(image, gamma):
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_frame_for_detection(frame_bgr):
    """
    Reduce glare, normalize brightness, and apply light noise reduction.

    Steps:
      - compute mean brightness
      - gamma correction (brighten/darken)
      - CLAHE on L channel (LAB space) for local contrast
      - if extremely bright, fallback to global histogram equalization
      - apply small Gaussian blur to reduce high-frequency noise

    Returns:
      processed BGR frame
    """
    img = frame_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Gamma correction
    if mean_brightness < BRIGHTNESS_LOW:
        img = apply_gamma(img, 1.4)  # brighten
    elif mean_brightness > BRIGHTNESS_HIGH:
        img = apply_gamma(img, 0.7)  # darken

    # Very bright -> histogram equalization fallback
    if mean_brightness > BRIGHTNESS_VERY_HIGH:
        eq = cv2.equalizeHist(gray)
        img = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        # mild blur to reduce harsh noise edges
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Light noise reduction: small Gaussian blur preserves edges reasonably
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ----------------------------
# MoveNet MultiPose: parsing output
# ----------------------------

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def parse_movenet_multipose(output_1x6x56, frame_width, frame_height):
    """
    Parse MoveNet MultiPose output (1, 6, 56) -> list of persons:
      {
        "keypoints": np.array shape (17, 3) [x, y, score],
        "bbox": (x_min, y_min, x_max, y_max),
        "instance_score": float
      }
    in pixel coordinates.
    """
    persons = []
    detections = output_1x6x56[0]  # (6, 56)

    for det in detections:
        keypoints_flat = det[:51]  # 17 * 3
        bbox_and_score = det[51:]  # ymin, xmin, ymax, xmax, score

        ymin, xmin, ymax, xmax, instance_score = bbox_and_score.tolist()
        if instance_score < INSTANCE_SCORE_THRESH:
            continue

        x_min_px = int(xmin * frame_width)
        y_min_px = int(ymin * frame_height)
        x_max_px = int(xmax * frame_width)
        y_max_px = int(ymax * frame_height)

        # bound box inside frame
        x_min_px = max(0, min(frame_width - 1, x_min_px))
        y_min_px = max(0, min(frame_height - 1, y_min_px))
        x_max_px = max(0, min(frame_width - 1, x_max_px))
        y_max_px = max(0, min(frame_height - 1, y_max_px))

        kp = []
        for i in range(17):
            y_norm = keypoints_flat[3 * i + 0]
            x_norm = keypoints_flat[3 * i + 1]
            score = keypoints_flat[3 * i + 2]
            x_px = x_norm * frame_width
            y_px = y_norm * frame_height
            kp.append([x_px, y_px, score])
        keypoints = np.array(kp, dtype=np.float32)

        persons.append(
            {
                "keypoints": keypoints,
                "bbox": (x_min_px, y_min_px, x_max_px, y_max_px),
                "instance_score": float(instance_score),
            }
        )

    return persons


# ----------------------------
# Simple centroid tracker
# ----------------------------

class SimpleCentroidTracker:
    def __init__(self, max_distance=TRACKER_MAX_DISTANCE, max_age=TRACKER_MAX_AGE):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}  # id -> dict
        self.next_id = 1

    def _compute_centroid(self, keypoints, bbox):
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    def update(self, persons, frame_idx):
        """Update tracker with new detections (persons)."""
        detections = []
        for p in persons:
            bbox = p["bbox"]
            centroid = self._compute_centroid(p["keypoints"], bbox)
            detections.append(
                {
                    "bbox": bbox,
                    "keypoints": p["keypoints"],
                    "instance_score": p["instance_score"],
                    "centroid": centroid,
                }
            )

        # mark tracks unmatched
        for tid, t in self.tracks.items():
            t["matched_in_frame"] = False

        track_ids = list(self.tracks.keys())
        used_tracks = set()

        # greedy matching
        for det in detections:
            best_id = None
            best_dist = float("inf")
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                track = self.tracks[tid]
                dist = np.linalg.norm(det["centroid"] - track["centroid"])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist <= self.max_distance:
                track = self.tracks[best_id]
                track["bbox"] = det["bbox"]
                track["keypoints"] = det["keypoints"]
                track["centroid"] = det["centroid"]
                track["instance_score"] = det["instance_score"]
                track["last_seen"] = frame_idx
                track["matched_in_frame"] = True
                used_tracks.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "id": tid,
                    "bbox": det["bbox"],
                    "keypoints": det["keypoints"],
                    "centroid": det["centroid"],
                    "instance_score": det["instance_score"],
                    "last_seen": frame_idx,
                    "matched_in_frame": True,
                    "is_eligible": False,
                    "eligible_start_frame": None,
                }
                used_tracks.add(tid)

        # remove stale tracks
        to_delete = []
        for tid, t in self.tracks.items():
            if frame_idx - t["last_seen"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks


# ----------------------------
# Facing / gaze heuristic
# ----------------------------

def is_person_facing_exhibit(track, gray_frame, face_detector):
    """
    Returns (facing_bool, used_face_detector_bool).

    - Prefer face-based heuristic when we can detect a reasonably large face.
    - Otherwise, fall back to body-orientation-based heuristic using shoulders and nose.
    """
    keypoints = track["keypoints"]
    bbox = track["bbox"]
    x_min, y_min, x_max, y_max = bbox

    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    used_face = False
    facing = False

    # 1) Face-based heuristic (if enabled and detector available)
    if ENABLE_FACE_DETECTOR and face_detector is not None and not face_detector.empty():
        head_top = y_min
        head_bottom = (y_min + y_max) * 0.5
        if left_shoulder[2] > KEYPOINT_CONF_THRESH and right_shoulder[2] > KEYPOINT_CONF_THRESH:
            head_bottom = (left_shoulder[1] + right_shoulder[1]) / 2.0

        head_top = int(max(0, head_top))
        head_bottom = int(max(0, min(gray_frame.shape[0] - 1, head_bottom)))
        x_min_i = int(max(0, x_min))
        x_max_i = int(max(0, min(gray_frame.shape[1] - 1, x_max)))

        if head_bottom > head_top and x_max_i > x_min_i:
            roi = gray_frame[head_top:head_bottom, x_min_i:x_max_i]
            if roi.size > 0:
                faces = face_detector.detectMultiScale(
                    roi,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(FACE_MIN_SIZE_PX, FACE_MIN_SIZE_PX),
                )
                if len(faces) > 0:
                    used_face = True
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                    if (
                        nose[2] > KEYPOINT_CONF_THRESH
                        and left_eye[2] > KEYPOINT_CONF_THRESH
                        and right_eye[2] > KEYPOINT_CONF_THRESH
                    ):
                        mid_eye_x = 0.5 * (left_eye[0] + right_eye[0])
                        dx = abs(nose[0] - mid_eye_x)
                        thresh = fw * 0.12
                        facing = dx < thresh
                    else:
                        aspect = fw / float(fh + 1e-6)
                        if 0.8 <= aspect <= 1.4:
                            facing = True

                    if used_face:
                        return facing, used_face

    # 2) Fallback: body orientation via shoulders + nose
    nose_ok = nose[2] > KEYPOINT_CONF_THRESH
    ls_ok = left_shoulder[2] > KEYPOINT_CONF_THRESH
    rs_ok = right_shoulder[2] > KEYPOINT_CONF_THRESH

    if nose_ok and ls_ok and rs_ok:
        shoulder_mid_x = 0.5 * (left_shoulder[0] + right_shoulder[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_width > 1:
            offset = nose[0] - shoulder_mid_x
            if abs(offset) < shoulder_width * 0.35:
                facing = True

    return facing, used_face


# ----------------------------
# Eligibility & perimeter
# ----------------------------

def compute_perimeter(frame_width, frame_height):
    x_min = int(frame_width * PERIMETER_X_MARGIN)
    x_max = int(frame_width * (1.0 - PERIMETER_X_MARGIN))
    y_min = int(frame_height * PERIMETER_Y_MARGIN)
    y_max = int(frame_height * (1.0 - PERIMETER_Y_MARGIN))
    return x_min, y_min, x_max, y_max


def update_eligibility(tracks, frame_idx, frame_width, frame_height, gray_frame, face_detector):
    frame_area = frame_width * frame_height
    min_area = frame_area * MIN_BBOX_AREA_RATIO
    px_min, py_min, px_max, py_max = compute_perimeter(frame_width, frame_height)

    core_indices = [0, 5, 6, 11, 12]  # nose, shoulders, hips

    for tid, track in tracks.items():
        bbox = track["bbox"]
        x_min, y_min, x_max, y_max = bbox
        cx, cy = track["centroid"]

        inside_perimeter = (px_min <= cx <= px_max) and (py_min <= cy <= py_max)
        bbox_area = max(0, x_max - x_min) * max(0, y_max - y_min)
        big_enough = bbox_area >= min_area

        kps = track["keypoints"]
        core_good = sum(1 for idx in core_indices if kps[idx, 2] >= KEYPOINT_CONF_THRESH)
        has_good_keypoints = core_good >= CORE_KEYPOINT_MIN_COUNT

        facing, _ = is_person_facing_exhibit(track, gray_frame, face_detector)

        prev_eligible = track.get("is_eligible", False)
        is_eligible = inside_perimeter and big_enough and has_good_keypoints and facing

        if is_eligible and not prev_eligible:
            track["eligible_start_frame"] = frame_idx
        elif not is_eligible:
            track["eligible_start_frame"] = None

        track["is_eligible"] = is_eligible


# ----------------------------
# Target selection (single / multi) – with edge-friendly fallbacks
# ----------------------------

def select_single_focus(tracks, frame_idx, locked_target_id):
    """
    Single focus mode:
      - If current target is still eligible, keep it.
      - Else choose eligible person with longest continuous eligibility.
      - If NO eligible people exist at all, fall back to the largest person
        so that the eyes still move on edge devices.
    Returns (new_locked_target_id, primary_track).
    """
    if not tracks:
        return None, None

    eligible_tracks = [t for t in tracks.values() if t["is_eligible"]]

    if locked_target_id is not None and locked_target_id in tracks:
        t = tracks[locked_target_id]
        if t["is_eligible"]:
            return locked_target_id, t

    if eligible_tracks:
        def eligibility_key(t):
            start = t.get("eligible_start_frame")
            dur = (frame_idx - start) if start is not None else 0
            x_min, y_min, x_max, y_max = t["bbox"]
            area = max(0, x_max - x_min) * max(0, y_max - y_min)
            return (dur, area)

        best_track = max(eligible_tracks, key=eligibility_key)
        return best_track["id"], best_track

    # Fallback: no eligible tracks, pick largest by area
    def area(t):
        x_min, y_min, x_max, y_max = t["bbox"]
        return max(0, x_max - x_min) * max(0, y_max - y_min)

    best_track = max(tracks.values(), key=area)
    return best_track["id"], best_track


def select_multi_focus(tracks, frame_idx):
    """
    Multi focus mode:
      - Up to 3 eligible people, FCFS by eligible_start_frame.
      - If no eligible people exist, fall back to the 3 largest tracks
        so the eyes still move on edge devices.
    Returns:
      - pair_target_ids: list of length 3 with track IDs or None.
      - primary_track: track dict or None.
    """
    # No tracks at all
    if not tracks:
        return [None, None, None], None

    # Use eligibility flag computed in update_eligibility
    eligible_tracks = [t for t in tracks.values() if t["is_eligible"]]

    if eligible_tracks:
        # First-come, first-served: earliest eligible_start_frame wins
        def start_frame_or_now(t):
            start = t.get("eligible_start_frame")
            return start if start is not None else frame_idx

        eligible_sorted = sorted(eligible_tracks, key=start_frame_or_now)
        primary = eligible_sorted[0]

        pair_target_ids = []
        for i in range(3):
            if i < len(eligible_sorted):
                pair_target_ids.append(eligible_sorted[i]["id"])
            else:
                # leftover pairs follow primary
                pair_target_ids.append(primary["id"])

        return pair_target_ids, primary

    # --- Fallback: no eligible tracks, use 3 largest by bbox area ---
    def area(t):
        x_min, y_min, x_max, y_max = t["bbox"]
        return max(0, x_max - x_min) * max(0, y_max - y_min)

    all_sorted = sorted(tracks.values(), key=area, reverse=True)
    primary = all_sorted[0]

    pair_target_ids = []
    for i in range(3):
        if i < len(all_sorted):
            pair_target_ids.append(all_sorted[i]["id"])
        else:
            pair_target_ids.append(primary["id"])

    return pair_target_ids, primary


# ----------------------------
# Eyes controller & rendering
# ----------------------------

class EyesController:
    def __init__(self, num_pairs=3):
        self.num_pairs = num_pairs
        self.smoothed_offsets = [0.0 for _ in range(num_pairs)]

    def update_offsets(self, raw_offsets):
        for i in range(self.num_pairs):
            x = raw_offsets[i] if raw_offsets[i] is not None else 0.0
            x = max(-1.0, min(1.0, x))
            prev = self.smoothed_offsets[i]
            self.smoothed_offsets[i] = SMOOTHING_ALPHA * x + (1.0 - SMOOTHING_ALPHA) * prev

    def render(self):
        canvas = np.ones((EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8) * 255

        rows_y = [
            int(EYES_CANVAS_HEIGHT * 0.22),
            int(EYES_CANVAS_HEIGHT * 0.5),
            int(EYES_CANVAS_HEIGHT * 0.78),
        ]
        pair_center_x = EYES_CANVAS_WIDTH // 2
        eye_spacing = int(EYES_CANVAS_WIDTH * 0.20)

        for i in range(self.num_pairs):
            row_y = rows_y[i]
            offset_norm = self.smoothed_offsets[i]
            pupil_dx = int(offset_norm * PUPIL_TRAVEL_PX)

            left_center = (pair_center_x - eye_spacing // 2, row_y)
            right_center = (pair_center_x + eye_spacing // 2, row_y)

            for center in [left_center, right_center]:
                cx, cy = center
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (255, 255, 255), thickness=-1)
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (0, 0, 0), thickness=2)

                pupil_center = (cx + pupil_dx, cy)
                cv2.circle(canvas, pupil_center, PUPIL_RADIUS, (0, 0, 0), thickness=-1)

        return canvas


# ----------------------------
# Main
# ----------------------------

def main():
    global SELECTION_MODE

    print("Loading MoveNet MultiPose Lightning model...")
    interpreter = load_movenet_interpreter(MOVENET_MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    print("Model input details:", input_details)

    # Face detector (optional)
    face_detector = None
    if ENABLE_FACE_DETECTOR:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(face_cascade_path)
        if face_detector.empty():
            print("Warning: Haar cascade failed to load. Falling back to body-only facing heuristic.")
            face_detector = None

    # Video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Could not open video source:", VIDEO_SOURCE)
        return

    # For cameras, hint a modest resolution (ignored by file inputs)
    if isinstance(VIDEO_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = SimpleCentroidTracker()
    eyes = EyesController(num_pairs=3)
    locked_target_id = None
    last_persons = []

    frame_idx = 0
    last_fps_time = time.time()
    frame_count_for_fps = 0
    fps = 0.0

    print("Press 'q' to quit. Press 'm' to toggle selection mode (single/multi).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # If this is a camera, mirror for natural interaction
        if isinstance(VIDEO_SOURCE, int):
            frame = cv2.flip(frame, 1)

        h0, w0, _ = frame.shape

        # Downscale large frames on edge devices (e.g., from 4096x2160 to <=960x540)
        if EDGE_DOWNSCALE:
            scale = min(
                TARGET_PROCESSING_WIDTH / float(w0),
                TARGET_PROCESSING_HEIGHT / float(h0),
                1.0,
            )
            if scale < 1.0:
                new_w = int(w0 * scale)
                new_h = int(h0 * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w, _ = frame.shape

        proc_frame = preprocess_frame_for_detection(frame)
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

        # Run MoveNet at reduced frame rate for edge devices
        run_inference = (frame_idx % INFER_EVERY_N_FRAMES == 1)
        persons = last_persons

        if run_inference or not last_persons:
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
            input_img = rgb_resized.astype(np.uint8)
            input_img = np.expand_dims(input_img, axis=0)

            interpreter.set_tensor(input_index, input_img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)

            persons = parse_movenet_multipose(output, w, h)
            last_persons = persons

        tracks = tracker.update(persons, frame_idx)
        update_eligibility(tracks, frame_idx, w, h, gray, face_detector)

        # Selection & mapping to eyes
        primary_track = None
        pair_target_ids = [None, None, None]

        if SELECTION_MODE == "single":
            locked_target_id, primary_track = select_single_focus(
                tracks, frame_idx, locked_target_id
            )
            if primary_track is not None:
                cx, _ = primary_track["centroid"]
                norm_x = (cx / w - 0.5) * 2.0
            else:
                norm_x = 0.0
            pair_offsets = [norm_x, norm_x, norm_x]
        else:
            pair_target_ids, primary_track = select_multi_focus(tracks, frame_idx)
            pair_offsets = []
            for tid in pair_target_ids:
                if tid is not None and tid in tracks:
                    cx, _ = tracks[tid]["centroid"]
                    norm_x = (cx / w - 0.5) * 2.0
                    pair_offsets.append(norm_x)
                else:
                    pair_offsets.append(0.0)

        eyes.update_offsets(pair_offsets)
        eyes_canvas = eyes.render()

        # Draw perimeter
        px_min, py_min, px_max, py_max = compute_perimeter(w, h)
        cv2.rectangle(frame, (px_min, py_min), (px_max, py_max), (255, 255, 0), 2)

        # Draw tracks
        for tid, t in tracks.items():
            x_min, y_min, x_max, y_max = t["bbox"]
            cx, cy = t["centroid"]
            is_eligible = t["is_eligible"]
            color = (0, 255, 0) if is_eligible else (0, 0, 255)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

            for (kx, ky, ks) in t["keypoints"]:
                if ks >= KEYPOINT_CONF_THRESH:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (255, 0, 0), -1)

            label = f"ID {tid}"
            if is_eligible:
                label += " [ELIGIBLE]"
            if primary_track is not None and tid == primary_track["id"]:
                label += " [PRIMARY]"

            cv2.putText(
                frame, label, (x_min, max(0, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # FPS
        frame_count_for_fps += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count_for_fps / (now - last_fps_time)
            last_fps_time = now
            frame_count_for_fps = 0

        mode_text = f"Mode: {SELECTION_MODE.upper()}  FPS: {fps:.1f}"
        cv2.putText(
            frame, mode_text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Press 'm' to toggle mode, 'q' to quit.",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Optional debug: show first pair's horizontal offset
        debug_norm = pair_offsets[0] if pair_offsets else 0.0
        cv2.putText(
            frame, f"norm_x: {debug_norm:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.imshow("Padayani Camera", frame)
        cv2.imshow("Padayani Eyes", eyes_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            SELECTION_MODE = "multi" if SELECTION_MODE == "single" else "single"
            locked_target_id = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()