"""
Padayani Eye Tracker Prototype (Webcam + Bench ROI Lock)
--------------------------------------------------------

Requirements (inside your venv):
    pip install opencv-python numpy tensorflow

Model:
    Download MoveNet MultiPose Lightning TFLite from TF Hub:
      https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1
    Save it next to this script as:
      movenet_multipose_lightning.tflite

Behavior:
    - Uses MoveNet MultiPose Lightning (17 keypoints, up to 6 people).
    - Robust centroid tracker with smoothing.
    - Defines a "bench ROI" region; first person to stay in that ROI
      for N frames becomes the LOCKED target.
    - Eyes follow ONLY the locked target until their track disappears.
    - Three pairs of circular eyes; pupils move left/right based on
      locked target's horizontal position.

Keys:
    'q' or ESC -> quit
"""

import cv2
import numpy as np
import time
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------

# Use 0 for built-in webcam, 1/2 if your ZED shows as another webcam
VIDEO_SOURCE = 2

# Path to MoveNet MultiPose Lightning model
MOVENET_MODEL_PATH = "./model/movenet_multipose_lightning.tflite"

# MoveNet input size
MOVENET_INPUT_SIZE = 256

# Run inference every N frames (1 = every frame)
INFER_EVERY_N_FRAMES = 1

# Downscaling for performance (useful for HD/4K sources)
EDGE_DOWNSCALE = False
TARGET_PROCESSING_WIDTH = 960
TARGET_PROCESSING_HEIGHT = 540

# TFLite threads
NUM_TFLITE_THREADS = 2

# Detection thresholds
INSTANCE_SCORE_THRESH = 0.15
KEYPOINT_CONF_THRESH = 0.15

# Tracker settings
TRACKER_MAX_DISTANCE = 80       # px
TRACKER_MAX_AGE = 20            # frames
TRACK_SMOOTH_ALPHA = 0.4        # EMA for centroid/bbox

# Eyes canvas
EYES_CANVAS_WIDTH = 800
EYES_CANVAS_HEIGHT = 400
EYE_RADIUS = 40
PUPIL_RADIUS = 14
PUPIL_TRAVEL_PX = 25            # max left/right movement inside eye
PUPIL_SMOOTH_ALPHA = 0.25       # EMA for pupil movement

# Bench ROI: fractional coords of frame (tune for your bench position)
# Example: bottom-middle region of the frame.
BENCH_ROI_X_MIN_FRAC = 0.25
BENCH_ROI_X_MAX_FRAC = 0.75
BENCH_ROI_Y_MIN_FRAC = 0.55
BENCH_ROI_Y_MAX_FRAC = 0.95


# How many consecutive frames inside bench ROI before locking
BENCH_LOCK_FRAMES = 60


# ----------------------------
# MoveNet MultiPose wrapper
# ----------------------------

class MoveNetMultiPose:
    def __init__(self, model_path, input_size=256, num_threads=2):
        """
        Wraps MoveNet MultiPose TFLite model using tf.lite.Interpreter.
        Auto-handles float or uint8 input.
        """
        import tensorflow as tf
        import numpy as np

        try:
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
            )
        except TypeError:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        input_details = self.interpreter.get_input_details()
        self.input_index = input_details[0]["index"]

        # Resize input tensor to fixed size
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

        print("[INFO] MoveNet input dtype:", self.input_dtype)

    def infer(self, frame_bgr):
        """
        Run MoveNet MultiPose on a BGR frame and return raw output (1, 6, 56).
        """
        import numpy as np

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size))

        # Prepare input according to model dtype
        if self.input_dtype in (np.float32, np.float16):
            input_img = rgb_resized.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0).astype(self.input_dtype)
        else:
            input_img = rgb_resized.astype(np.uint8)
            input_img = np.expand_dims(input_img, axis=0)

        self.interpreter.set_tensor(self.input_index, input_img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)  # (1, 6, 56)
        return output


# ----------------------------
# Minimal preprocessing
# ----------------------------

def preprocess_frame_for_detection(frame_bgr):
    """
    Simple noise reduction. Keep it light so we don't break the model.
    """
    img = cv2.GaussianBlur(frame_bgr, (3, 3), 0)
    return img


# ----------------------------
# MoveNet MultiPose parsing
# ----------------------------

def parse_movenet_multipose(raw_output, frame_width, frame_height,
                            min_pose_score=INSTANCE_SCORE_THRESH,
                            min_kp_score=KEYPOINT_CONF_THRESH):
    """
    Parse MoveNet MultiPose output (1, 6, 56) into list of detections:

    Each detection:
        {
            "keypoints": np.array (17, 3) [x_px, y_px, score],
            "bbox": (x_min, y_min, x_max, y_max),
            "pose_score": float
        }
    """
    detections = []
    people = raw_output[0]  # (6, 56)

    for det in people:
        det = np.asarray(det, dtype=np.float32)

        keypoints_flat = det[:51]  # 17 * (y, x, score)
        ymin, xmin, ymax, xmax, pose_score = det[51:]

        pose_score = float(pose_score)
        if pose_score < min_pose_score:
            continue

        x_min_px = int(xmin * frame_width)
        y_min_px = int(ymin * frame_height)
        x_max_px = int(xmax * frame_width)
        y_max_px = int(ymax * frame_height)

        x_min_px = max(0, min(frame_width - 1, x_min_px))
        y_min_px = max(0, min(frame_height - 1, y_min_px))
        x_max_px = max(0, min(frame_width - 1, x_max_px))
        y_max_px = max(0, min(frame_height - 1, y_max_px))

        # Keypoints -> [x_px, y_px, score]
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

        keypoints = np.array(kps, dtype=np.float32)

        detections.append(
            {
                "keypoints": keypoints,
                "bbox": (x_min_px, y_min_px, x_max_px, y_max_px),
                "pose_score": pose_score,
            }
        )

    return detections


# ----------------------------
# Robust centroid tracker with smoothing
# ----------------------------

class RobustTracker:
    """
    Simple but robust tracker:

    - Greedy centroid matching with distance gating.
    - Per-track smoothed centroid & bbox (EMA).
    - Tracks age out after TRACKER_MAX_AGE frames.
    """

    def __init__(self, max_distance=TRACKER_MAX_DISTANCE,
                 max_age=TRACKER_MAX_AGE,
                 smooth_alpha=TRACK_SMOOTH_ALPHA):
        self.max_distance = max_distance
        self.max_age = max_age
        self.smooth_alpha = smooth_alpha

        self.tracks = {}   # id -> dict
        self.next_id = 1

    @staticmethod
    def _centroid_from_bbox(bbox):
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    def update(self, detections, frame_idx):
        """
        Update tracks with new detections.

        detections: list of { "keypoints", "bbox", "pose_score" }
        """
        dets = []
        for d in detections:
            bbox = d["bbox"]
            centroid = self._centroid_from_bbox(bbox)
            dets.append(
                {
                    "bbox": bbox,
                    "keypoints": d["keypoints"],
                    "pose_score": d["pose_score"],
                    "centroid": centroid,
                }
            )

        # Mark tracks unmatched
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

                # Smooth centroid & bbox
                new_centroid = det["centroid"]
                old_centroid = tr["centroid"]
                tr["centroid"] = (
                    self.smooth_alpha * new_centroid
                    + (1.0 - self.smooth_alpha) * old_centroid
                )

                x_min, y_min, x_max, y_max = tr["bbox"]
                nx_min, ny_min, nx_max, ny_max = det["bbox"]

                x_min = self.smooth_alpha * nx_min + (1 - self.smooth_alpha) * x_min
                y_min = self.smooth_alpha * ny_min + (1 - self.smooth_alpha) * y_min
                x_max = self.smooth_alpha * nx_max + (1 - self.smooth_alpha) * x_max
                y_max = self.smooth_alpha * ny_max + (1 - self.smooth_alpha) * y_max

                tr["bbox"] = (int(x_min), int(y_min), int(x_max), int(y_max))
                tr["keypoints"] = det["keypoints"]
                tr["pose_score"] = det["pose_score"]
                tr["last_seen"] = frame_idx
                tr["matched_in_frame"] = True
                used_tracks.add(best_tid)
            else:
                # New track
                tid = self.next_id
                self.next_id += 1
                bbox = det["bbox"]
                centroid = det["centroid"]
                self.tracks[tid] = {
                    "id": tid,
                    "bbox": bbox,
                    "keypoints": det["keypoints"],
                    "centroid": centroid,
                    "pose_score": det["pose_score"],
                    "last_seen": frame_idx,
                    "matched_in_frame": True,
                }
                used_tracks.add(tid)

        # Age out stale tracks
        to_delete = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_seen"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks


# ----------------------------
# Eyes controller
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
            self.smoothed_offsets[i] = (
                PUPIL_SMOOTH_ALPHA * x + (1.0 - PUPIL_SMOOTH_ALPHA) * prev
            )

    def render(self):
        canvas = np.ones((EYES_CANVAS_HEIGHT, EYES_CANVAS_WIDTH, 3), dtype=np.uint8) * 255

        rows_y = [
            int(EYES_CANVAS_HEIGHT * 0.22),
            int(EYES_CANVAS_HEIGHT * 0.5),
            int(EYES_CANVAS_HEIGHT * 0.78),
        ]
        center_x = EYES_CANVAS_WIDTH // 2
        eye_spacing = int(EYES_CANVAS_WIDTH * 0.20)

        for i in range(self.num_pairs):
            row_y = rows_y[i]
            offset_norm = self.smoothed_offsets[i]
            pupil_dx = int(offset_norm * PUPIL_TRAVEL_PX)

            left_center = (center_x - eye_spacing // 2, row_y)
            right_center = (center_x + eye_spacing // 2, row_y)

            for cx, cy in [left_center, right_center]:
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (255, 255, 255), thickness=-1)
                cv2.circle(canvas, (cx, cy), EYE_RADIUS, (0, 0, 0), thickness=2)

                pupil_center = (cx + pupil_dx, cy)
                cv2.circle(canvas, pupil_center, PUPIL_RADIUS, (0, 0, 0), thickness=-1)

        return canvas


# ----------------------------
# Bench ROI helpers
# ----------------------------

def compute_bench_roi(frame_width, frame_height):
    x_min = int(frame_width * BENCH_ROI_X_MIN_FRAC)
    x_max = int(frame_width * BENCH_ROI_X_MAX_FRAC)
    y_min = int(frame_height * BENCH_ROI_Y_MIN_FRAC)
    y_max = int(frame_height * BENCH_ROI_Y_MAX_FRAC)
    return x_min, y_min, x_max, y_max


def is_inside_roi(cx, cy, roi):
    x_min, y_min, x_max, y_max = roi
    return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)


# ----------------------------
# Main loop
# ----------------------------

def main():
    print("Loading MoveNet MultiPose Lightning model...")
    pose_model = MoveNetMultiPose(
        MOVENET_MODEL_PATH,
        input_size=MOVENET_INPUT_SIZE,
        num_threads=NUM_TFLITE_THREADS,
    )

    # Video capture
    if isinstance(VIDEO_SOURCE, int):
        print(f"[INFO] Using webcam source: {VIDEO_SOURCE}")
    else:
        print(f"[INFO] Using video file: {VIDEO_SOURCE}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video source:", VIDEO_SOURCE)
        return

    # Set webcam resolution hint
    if isinstance(VIDEO_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = RobustTracker()
    eyes = EyesController(num_pairs=3)

    frame_idx = 0
    last_fps_time = time.time()
    frame_count_for_fps = 0
    fps = 0.0

    # Bench lock state
    locked_target_id = None
    bench_roi_counters = {}   # tid -> consecutive frames inside ROI

    print("Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Mirror webcam for natural interaction
        if isinstance(VIDEO_SOURCE, int):
            frame = cv2.flip(frame, 1)

        h0, w0, _ = frame.shape

        # Downscale large frames for speed
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

        # Run inference at configured frequency
        if INFER_EVERY_N_FRAMES <= 1:
            run_inference = True
        else:
            run_inference = (frame_idx % INFER_EVERY_N_FRAMES == 0)

        if run_inference:
            pose_output = pose_model.infer(proc_frame)
            detections = parse_movenet_multipose(pose_output, w, h)
        else:
            detections = []

        # Update tracker
        tracks = tracker.update(detections, frame_idx)

        # Compute bench ROI (in pixel coordinates)
        bench_roi = compute_bench_roi(w, h)
        bx_min, by_min, bx_max, by_max = bench_roi

        # --- Bench lock logic ---

        # Clean counters for removed tracks
        existing_ids = set(tracks.keys())
        bench_roi_counters = {tid: c for tid, c in bench_roi_counters.items() if tid in existing_ids}

        # If locked target disappeared, release lock
        if locked_target_id is not None and locked_target_id not in tracks:
            locked_target_id = None

        if locked_target_id is None:
            # We are free to lock a new person sitting on the bench
            for tid, tr in tracks.items():
                cx, cy = tr["centroid"]
                if is_inside_roi(cx, cy, bench_roi):
                    bench_roi_counters[tid] = bench_roi_counters.get(tid, 0) + 1
                else:
                    bench_roi_counters[tid] = 0

            # Check if any track earned the lock
            for tid, count in bench_roi_counters.items():
                if count >= BENCH_LOCK_FRAMES:
                    locked_target_id = tid
                    current_datetime = datetime.now()
                    formatted_timestamp_1 = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[INFO] @ {formatted_timestamp_1}: Locked on ID {tid} (bench sitter)")
                    break
        else:
            # We keep lock until track disappears; no need to re-count
            pass

        # --- Eyes control ---

                # --- Eyes control ---

        # Default: eyes neutral (looking straight ahead)
        pair_offsets = [0.0, 0.0, 0.0]

        if locked_target_id is not None and locked_target_id in tracks:
            # Only the locked bench sitter drives the eyes
            tr = tracks[locked_target_id]
            cx, cy = tr["centroid"]
            norm_x = (cx / w - 0.5) * 2.0  # [-1, 1]
            pair_offsets = [norm_x, norm_x, norm_x]
        else:
            # No locked target -> eyes stay neutral,
            # do NOT track any other person.
            pass

        eyes.update_offsets(pair_offsets)
        eyes_canvas = eyes.render()

        # Draw bench ROI
        cv2.rectangle(frame, (bx_min, by_min), (bx_max, by_max), (255, 0, 255), 2)
        cv2.putText(
            frame, "Bench ROI",
            (bx_min, max(0, by_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA
        )

        # Draw tracks (boxes + keypoints)
        for tid, tr in tracks.items():
            x_min, y_min, x_max, y_max = tr["bbox"]
            cx, cy = tr["centroid"]

            # Color: locked target in cyan, others green
            if tid == locked_target_id:
                color = (255, 255, 0)  # cyan-ish
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

            for (kx, ky, ks) in tr["keypoints"]:
                if ks >= KEYPOINT_CONF_THRESH:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (255, 0, 0), -1)

            label = f"ID {tid}"
            if tid == locked_target_id:
                label += " [LOCKED]"
            cv2.putText(
                frame, label, (x_min, max(0, y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # FPS counter
        frame_count_for_fps += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count_for_fps / (now - last_fps_time)
            last_fps_time = now
            frame_count_for_fps = 0

        # HUD
        lock_text = f"LOCKED ID: {locked_target_id}" if locked_target_id is not None else "LOCKED ID: None"
        cv2.putText(
            frame, f"FPS: {fps:.1f} | Tracks: {len(tracks)} | {lock_text}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        if pair_offsets:
            cv2.putText(
                frame, f"norm_x: {pair_offsets[0]:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow("Padayani Camera", frame)
        cv2.imshow("Padayani Eyes", eyes_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
