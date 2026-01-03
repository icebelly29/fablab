"""
Padayani Eye Tracker Prototype
------------------------------

Requirements (Python packages):
    pip install opencv-python numpy
    # EITHER:
    pip install tflite-runtime
    # OR (heavier, but common):
    pip install tensorflow

MoveNet MultiPose Lightning TFLite model:
    Download from TF Hub:
      https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1?lite-format=tflite
    Save it next to this script as:
      movenet_multipose_lightning.tflite

Model I/O (from MoveNet.MultiPose model card):
  - Input: 1 x H x W x 3, RGB uint8, H and W multiples of 32.
    Recommended: resize so the larger side is 256, keep aspect ratio.
  - Output: [1, 6, 56]
      For each of 6 possible people:
        - First 17*3 = 51 values: [y0, x0, s0, y1, x1, s1, ..., y16, x16, s16]
          (normalized coords y,x in [0,1] and confidence s)
        - Last 5 values: [ymin, xmin, ymax, xmax, instance_score]

This script:
  - Opens webcam.
  - Runs MoveNet MultiPose Lightning for body keypoints.
  - Uses a Haar cascade for light face detection (only for a coarse facing / gaze check).
  - Applies brightness / glare mitigation (gamma + CLAHE, with very-bright fallback to hist eq).
  - Tracks people with a simple centroid tracker and assigns stable IDs.
  - Defines a configurable central perimeter region (~3m interaction zone).
  - Filters eligible people by:
      * inside perimeter
      * bounding box area (approx distance)
      * keypoint confidence
      * facing-toward heuristic
  - Has two selection modes:
      * "single": all three eye pairs lock onto one person.
      * "multi": up to 3 different people, FCFS.
  - Opens two OpenCV windows:
      * "Padayani Camera" – annotated camera feed.
      * "Padayani Eyes"   – cartoon eyes with horizontal-only pupil motion.
"""

import cv2
import numpy as np
import time
import math

# ----------------------------
# Configuration
# ----------------------------

# Path to MoveNet MultiPose Lightning model (see header for download URL)
MOVENET_MODEL_PATH = "movenet_multipose_lightning.tflite"

# Selection mode: "single" or "multi"
SELECTION_MODE = "multi"  # change to "multi" if you want one pair per person

# MoveNet input size (we'll resize frames to 256x256)
MOVENET_INPUT_SIZE = 256

# Keypoint & instance thresholds
KEYPOINT_CONF_THRESH = 0.2     # per-keypoint confidence
INSTANCE_SCORE_THRESH = 0.2    # overall instance score from MoveNet
CORE_KEYPOINT_MIN_COUNT = 4    # how many reliable core keypoints to require (nose/shoulders/hips)

# Perimeter region configuration (central rectangle in normalized coords)
# These are fractions of width/height. For a 3m-ish distance on a 640x480 stream,
# the default (0.2 margin on left/right, 0.15 on top/bottom) gives a central band.
PERIMETER_X_MARGIN = 0.01  # 0.2 => central 60% of width
PERIMETER_Y_MARGIN = 0.01  # 0.15 => central 70% of height

# Bounding-box area threshold as fraction of full frame
# Increase this if people appear too small at 3m; decrease if too strict.
MIN_BBOX_AREA_RATIO = 0.10  # 2% of frame area

# Brightness / glare mitigation
BRIGHTNESS_LOW = 60     # mean gray below this => brighten
BRIGHTNESS_HIGH = 190   # mean gray above this => darken
BRIGHTNESS_VERY_HIGH = 220  # fallback to histogram equalization when above this

# Centroid tracker configuration
TRACKER_MAX_DISTANCE = 60  # max pixels to match detections to an existing track
TRACKER_MAX_AGE = 30       # frames before a lost track is removed

# Eyes canvas configuration
EYES_CANVAS_WIDTH = 800
EYES_CANVAS_HEIGHT = 400
EYE_RADIUS = 40
PUPIL_RADIUS = 14
PUPIL_TRAVEL_PX = 25  # how far pupils can move left/right inside each eye

# Smoothing for pupil motion (exponential smoothing)
SMOOTHING_ALPHA = 0.25  # 0<alpha<=1 ; smaller => smoother but laggier

# Face detection (Haar cascade)
FACE_MIN_SIZE_PX = 40  # ignore faces smaller than this width/height


# ----------------------------
# Utility: load MoveNet interpreter
# ----------------------------

def load_movenet_interpreter(model_path):
    """
    Load MoveNet TFLite model using TensorFlow's TFLite Interpreter
    and resize the dynamic input to [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3].
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get current input details (will show shape [1, 1, 1, 3] with a dynamic signature)
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]

    # Resize input tensor to the desired size, e.g. 256x256
    interpreter.resize_tensor_input(
        input_index,
        [1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3]
    )

    # Now allocate tensors with the new size
    interpreter.allocate_tensors()

    return interpreter


# ----------------------------
# Preprocessing: glare / brightness mitigation
# ----------------------------

def apply_gamma(image, gamma):
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_frame_for_detection(frame_bgr):
    """
    Reduce glare and improve contrast:
      - compute mean brightness
      - apply gamma correction (brighten/darken)
      - apply CLAHE on L channel
      - if very bright, fallback to histogram equalization on grayscale
    Returns a new BGR frame.
    """
    # Work on a copy
    img = frame_bgr.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Gamma correction based on brightness
    if mean_brightness < BRIGHTNESS_LOW:
        img = apply_gamma(img, 1.4)  # brighten
    elif mean_brightness > BRIGHTNESS_HIGH:
        img = apply_gamma(img, 0.7)  # darken

    # If extremely bright (heavy glare), fallback to histogram equalization on grayscale
    if mean_brightness > BRIGHTNESS_VERY_HIGH:
        eq = cv2.equalizeHist(gray)
        img = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        return img

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
    Parse MoveNet MultiPose output (1, 6, 56) -> list of persons.
    Each person is:
        {
          "keypoints": np.array shape (17, 3) with columns [x, y, score],
          "bbox": (x_min, y_min, x_max, y_max),
          "instance_score": float
        }
    Coordinates are in pixel space.
    """
    persons = []
    detections = output_1x6x56[0]  # (6, 56)

    for det in detections:
        keypoints_flat = det[:51]  # 17*3
        bbox_and_score = det[51:]  # ymin, xmin, ymax, xmax, score

        ymin, xmin, ymax, xmax, instance_score = bbox_and_score.tolist()
        if instance_score < INSTANCE_SCORE_THRESH:
            continue

        # Convert bbox from normalized to pixel coords
        x_min_px = int(xmin * frame_width)
        y_min_px = int(ymin * frame_height)
        x_max_px = int(xmax * frame_width)
        y_max_px = int(ymax * frame_height)

        # Ensure bbox in bounds
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
        # Use bbox center; for distant people this is more stable than averaging sparse keypoints
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    def update(self, persons, frame_idx):
        """
        persons: list of dicts from parse_movenet_multipose
        frame_idx: current frame index
        Returns self.tracks (mutated).
        """
        # Build detection objects with centroid
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

        # Mark all tracks as unmatched initially
        for tid, t in self.tracks.items():
            t["matched_in_frame"] = False

        track_ids = list(self.tracks.keys())
        used_tracks = set()

        # Greedy matching det -> nearest track within max_distance
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
                # Update existing track
                track = self.tracks[best_id]
                track["bbox"] = det["bbox"]
                track["keypoints"] = det["keypoints"]
                track["centroid"] = det["centroid"]
                track["instance_score"] = det["instance_score"]
                track["last_seen"] = frame_idx
                track["matched_in_frame"] = True
                used_tracks.add(best_id)
            else:
                # Create new track
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
                    # Eligibility related
                    "is_eligible": False,
                    "eligible_start_frame": None,
                }
                used_tracks.add(tid)

        # Remove stale tracks
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

    # 1) Try face detection in head region
    used_face = False
    facing = False

    # Define a plausible head region: from bbox top down to shoulders mid Y (if shoulders known)
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
                # use the largest face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_width = fw
                face_height = fh

                # If we have good nose + eye keypoints, use nose vs eye midpoint
                if (
                    nose[2] > KEYPOINT_CONF_THRESH
                    and left_eye[2] > KEYPOINT_CONF_THRESH
                    and right_eye[2] > KEYPOINT_CONF_THRESH
                ):
                    mid_eye_x = 0.5 * (left_eye[0] + right_eye[0])
                    dx = abs(nose[0] - mid_eye_x)
                    # threshold ~ 12% of face width
                    thresh = face_width * 0.12
                    facing = dx < thresh
                else:
                    # Fallback: use face aspect ratio (rough heuristic)
                    aspect = face_width / float(face_height + 1e-6)
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
            # If nose is roughly centered over the shoulders, treat as facing
            if abs(offset) < shoulder_width * 0.35:
                facing = True

    return facing, used_face


# ----------------------------
# Eligibility & perimeter
# ----------------------------

def compute_perimeter(frame_width, frame_height):
    """
    Returns perimeter as (x_min, y_min, x_max, y_max) in pixels.
    """
    x_min = int(frame_width * PERIMETER_X_MARGIN)
    x_max = int(frame_width * (1.0 - PERIMETER_X_MARGIN))
    y_min = int(frame_height * PERIMETER_Y_MARGIN)
    y_max = int(frame_height * (1.0 - PERIMETER_Y_MARGIN))
    return x_min, y_min, x_max, y_max


def update_eligibility(tracks, frame_idx, frame_width, frame_height, gray_frame, face_detector):
    """
    For each track, determine if it is 'eligible' based on:
      - centroid inside perimeter
      - bbox area threshold
      - keypoint confidence
      - facing-toward check
    Updates track dicts in-place:
      - track["is_eligible"]
      - track["eligible_start_frame"]
    """
    frame_area = frame_width * frame_height
    min_area = frame_area * MIN_BBOX_AREA_RATIO
    perim = compute_perimeter(frame_width, frame_height)
    px_min, py_min, px_max, py_max = perim

    # indices of "core" keypoints we care about for robustness (nose, shoulders, hips)
    core_indices = [0, 5, 6, 11, 12]

    for tid, track in tracks.items():
        bbox = track["bbox"]
        x_min, y_min, x_max, y_max = bbox
        cx, cy = track["centroid"]

        # inside perimeter?
        inside_perimeter = (px_min <= cx <= px_max) and (py_min <= cy <= py_max)

        # bounding box area check
        bbox_area = max(0, x_max - x_min) * max(0, y_max - y_min)
        big_enough = bbox_area >= min_area

        # keypoint confidence check
        kps = track["keypoints"]
        core_good = sum(
            1 for idx in core_indices if kps[idx, 2] >= KEYPOINT_CONF_THRESH
        )
        has_good_keypoints = core_good >= CORE_KEYPOINT_MIN_COUNT

        # facing check (use body first, refine with face if possible)
        facing, _ = is_person_facing_exhibit(track, gray_frame, face_detector)

        prev_eligible = track.get("is_eligible", False)
        is_eligible = inside_perimeter and big_enough and has_good_keypoints and facing

        if is_eligible and not prev_eligible:
            track["eligible_start_frame"] = frame_idx
        elif not is_eligible:
            track["eligible_start_frame"] = None

        track["is_eligible"] = is_eligible


# ----------------------------
# Target selection (single vs multi)
# ----------------------------

def select_single_focus(tracks, frame_idx, locked_target_id):
    """
    Single focus mode:
      - If current target is still eligible, keep it.
      - Else choose eligible person with longest continuous eligibility.
      - Break ties by largest bbox area.
    Returns (new_locked_target_id, primary_track) where primary_track may be None.
    """
    # Filter eligible tracks
    eligible_tracks = [t for t in tracks.values() if t["is_eligible"]]

    # Keep the existing lock if still eligible
    if locked_target_id is not None and locked_target_id in tracks:
        t = tracks[locked_target_id]
        if t["is_eligible"]:
            return locked_target_id, t

    if not eligible_tracks:
        return None, None

    # Choose track with longest continuous eligibility
    def eligibility_key(t):
        start = t.get("eligible_start_frame")
        if start is None:
            dur = 0
        else:
            dur = frame_idx - start
        # break ties by bbox area
        x_min, y_min, x_max, y_max = t["bbox"]
        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        return (dur, area)

    best_track = max(eligible_tracks, key=eligibility_key)
    return best_track["id"], best_track


def select_multi_focus(tracks, frame_idx):
    """
    Multi focus mode:
      - Up to 3 eligible people.
      - FCFS: sorted by eligible_start_frame ascending.
      - Pair 1 -> earliest, Pair 2 -> second, Pair 3 -> third.
      - If fewer than 3 eligible:
          leftover pairs follow primary person (earliest), or stay neutral if none.
    Returns:
      - pair_target_ids: list of length 3 with track IDs or None.
      - primary_track: track dict or None.
    """
    eligible_tracks = [t for t in tracks.values() if t["is_eligible"]]

    if not eligible_tracks:
        return [None, None, None], None

    # If eligible_start_frame is None (just became eligible), treat that as "now"
    def start_frame_or_now(t):
        return t["eligible_start_frame"] if t["eligible_start_frame"] is not None else frame_idx

    # FCFS order
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


# ----------------------------
# Eyes controller & rendering
# ----------------------------

class EyesController:
    def __init__(self, num_pairs=3):
        self.num_pairs = num_pairs
        self.smoothed_offsets = [0.0 for _ in range(num_pairs)]  # in [-1, 1]

    def update_offsets(self, raw_offsets):
        """
        raw_offsets: list of length num_pairs, each in [-1, 1] or None.
        """
        for i in range(self.num_pairs):
            x = raw_offsets[i]
            if x is None:
                x = 0.0
            x = max(-1.0, min(1.0, x))
            prev = self.smoothed_offsets[i]
            new_val = SMOOTHING_ALPHA * x + (1.0 - SMOOTHING_ALPHA) * prev
            self.smoothed_offsets[i] = new_val

    def render(self):
        """
        Draw three pairs of circular eyes on a canvas, with perfectly circular pupils
        moving only left/right based on smoothed offsets.
        """
        canvas = np.ones(
            (EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8
        ) * 255

        # We'll place the three pairs in three horizontal rows
        rows_y = [
            int(EYES_CANVAS_HEIGHT * 0.22),
            int(EYES_CANVAS_HEIGHT * 0.5),
            int(EYES_CANVAS_HEIGHT * 0.78),
        ]
        # Each pair is centered horizontally; left/right eye around center
        pair_center_x = EYES_CANVAS_WIDTH // 2
        eye_spacing = int(EYES_CANVAS_WIDTH * 0.20)  # distance between left & right eye center

        for i in range(self.num_pairs):
            row_y = rows_y[i]
            offset_norm = self.smoothed_offsets[i]
            pupil_dx = int(offset_norm * PUPIL_TRAVEL_PX)

            # Left and right eye centers
            left_center = (pair_center_x - eye_spacing // 2, row_y)
            right_center = (pair_center_x + eye_spacing // 2, row_y)

            for center in [left_center, right_center]:
                cx, cy = center

                # Draw eyeball (white circle with black outline)
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (255, 255, 255), thickness=-1)
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (0, 0, 0), thickness=2)

                # Pupil center (horizontal offset only)
                pupil_center = (cx + pupil_dx, cy)
                cv2.circle(canvas, pupil_center, PUPIL_RADIUS, (0, 0, 0), thickness=-1)

        return canvas


# ----------------------------
# Main
# ----------------------------

def main():
    # Load MoveNet MultiPose Lightning
    global SELECTION_MODE
    print("Loading MoveNet MultiPose Lightning model...")
    interpreter = load_movenet_interpreter(MOVENET_MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Basic sanity check on input shape (dynamic height/width allowed)
    # We'll resize to MOVENET_INPUT_SIZE x MOVENET_INPUT_SIZE for simplicity.
    print("Model input details:", input_details)

    # Load Haar cascade for face detection
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    if face_detector.empty():
        print("Warning: failed to load Haar cascade for faces. Facing heuristic will rely on body only.")

    # Video capture
    cap = cv2.VideoCapture(0)  # Change to 0 for webcam  or 'testvideocomp.mp4' for test video
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Optional: set a moderate resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #480

    tracker = SimpleCentroidTracker()
    eyes = EyesController(num_pairs=3)
    locked_target_id = None  # for single-focus mode

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
        frame = cv2.flip(frame, 0)  # mirror for a more natural interaction

        h, w, _ = frame.shape

        # Preprocess for glare/brightness
        proc_frame = preprocess_frame_for_detection(frame)
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

        # Prepare input for MoveNet: resize to MOVENET_INPUT_SIZE x MOVENET_INPUT_SIZE
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
        input_img = rgb_resized.astype(np.uint8)
        input_img = np.expand_dims(input_img, axis=0)

        # Run inference
        interpreter.set_tensor(input_index, input_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)  # shape: (1, 6, 56)

        # Parse detections to persons (in original frame pixel coordinates)
        persons = parse_movenet_multipose(output, w, h)

        # Update centroid tracker
        tracks = tracker.update(persons, frame_idx)

        # Update eligibility per track
        update_eligibility(tracks, frame_idx, w, h, gray, face_detector)

        # Selection & mapping to eyes offsets
        primary_track = None
        pair_target_ids = [None, None, None]
        if SELECTION_MODE == "single":
            locked_target_id, primary_track = select_single_focus(
                tracks, frame_idx, locked_target_id
            )
            if primary_track is not None:
                cx, cy = primary_track["centroid"]
                norm_x = (cx / w - 0.5) * 2.0  # [-1, 1]
            else:
                norm_x = 0.0
            pair_offsets = [norm_x, norm_x, norm_x]
        else:  # "multi"
            pair_target_ids, primary_track = select_multi_focus(tracks, frame_idx)
            pair_offsets = []
            for tid in pair_target_ids:
                if tid is not None and tid in tracks:
                    cx, cy = tracks[tid]["centroid"]
                    norm_x = (cx / w - 0.5) * 2.0
                    pair_offsets.append(norm_x)
                else:
                    pair_offsets.append(0.0)

        eyes.update_offsets(pair_offsets)
        eyes_canvas = eyes.render()

        # Draw annotations on camera frame
        # Draw perimeter rectangle
        px_min, py_min, px_max, py_max = compute_perimeter(w, h)
        cv2.rectangle(frame, (px_min, py_min), (px_max, py_max), (255, 255, 0), 2)

        # Draw each track
        for tid, t in tracks.items():
            x_min, y_min, x_max, y_max = t["bbox"]
            cx, cy = t["centroid"]
            is_eligible = t["is_eligible"]

            # Bounding box: green if eligible, red otherwise
            color = (0, 255, 0) if is_eligible else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

            # Keypoints (small circles)
            kps = t["keypoints"]
            for i, (kx, ky, ks) in enumerate(kps):
                if ks >= KEYPOINT_CONF_THRESH:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (255, 0, 0), -1)

            # ID and status label
            label = f"ID {tid}"
            if is_eligible:
                label += " [ELIGIBLE]"
            if primary_track is not None and tid == primary_track["id"]:
                label += " [PRIMARY]"
            cv2.putText(
                frame, label, (x_min, max(0, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # FPS estimation
        frame_count_for_fps += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count_for_fps / (now - last_fps_time)
            last_fps_time = now
            frame_count_for_fps = 0

        # Info overlay
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

        # Show windows
        cv2.imshow("Padayani Camera", frame)
        cv2.imshow("Padayani Eyes", eyes_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m'):
            # toggle selection mode
            SELECTION_MODE = "multi" if SELECTION_MODE == "single" else "single"
            # reset lock when switching modes
            locked_target_id = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
