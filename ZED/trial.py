#!/usr/bin/env python3
"""
Padayani Eye-Tracking Prototype (ZED 1 + MoveNet MultiPose TFLite)

- ZED 1: RGB + depth / XYZ (only physical sensor)
- MoveNet MultiPose (TFLite): multi-person 2D skeletons
- ZED point cloud: used to get per-person 3D X/Z in meters

Behavior:
- Window 1:
    * ZED feed with:
        - Multi-person skeletons (2D)
        - Depth from camera (meters)
        - Confidence scores
        - Facing / not-facing
        - Color coding:
            Green = selected (drives one eye pair)
            Yellow = eligible but not selected
            Red   = detected but not eligible
- Window 2:
    * Three pairs of perfectly circular eyes
    * Pupils are circular and move LEFT–RIGHT only
    * One pair per selected person (up to 3)
    * Exponential smoothing, pupils clamped inside sclera
"""

import sys
import math
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import pyzed.sl as sl

import tensorflow as tf


# ----------------------------
# Paths
# ----------------------------

# If the model is in the same folder as this script:
MOVENET_TFLITE_PATH = "movenet_multipose_lightning.tflite"
# If it is in a subfolder, e.g. "models/movenet_multipose_lightning.tflite", set:
# MOVENET_TFLITE_PATH = r"C:\Users\nikhi\fablab\models\movenet_multipose_lightning.tflite"


# ----------------------------
# Parameters
# ----------------------------

# Depth range (meters) for "eligible" persons and optional depth mask
DEPTH_MIN_M = 2.0
DEPTH_MAX_M = 4.0

# MoveNet MultiPose thresholds
MIN_INSTANCE_SCORE = 0.25   # person score
MIN_KEYPOINT_SCORE = 0.3    # per-joint score
MIN_MEAN_KP_SCORE = 0.4

# Eye canvas parameters
EYE_CANVAS_WIDTH = 900
EYE_CANVAS_HEIGHT = 300
EYE_SCLERA_RADIUS = 45          # outer circle radius
EYE_PUPIL_RADIUS = 18           # inner circle radius
EYE_HORIZONTAL_SPACING = 130    # distance between eyes in a pair
MAX_PUPIL_OFFSET = EYE_SCLERA_RADIUS - EYE_PUPIL_RADIUS - 4

# Eye smoothing (per slot)
PUPIL_SMOOTHING_ALPHA = 0.2     # 0..1; higher = faster
MAX_X_RANGE_M = 2.0             # +/- meters of X mapped to full pupil offset

# Colors (BGR)
COLOR_SELECTED = (0, 255, 0)    # Green
COLOR_ELIGIBLE = (0, 255, 255)  # Yellow
COLOR_IGNORED = (0, 0, 255)     # Red
COLOR_TEXT = (255, 255, 255)    # White
COLOR_SKELETON_POINT = (255, 255, 255)

# Denoiser
USE_FAST_NL_MEANS = False

# Gamma correction
GAMMA_VALUE = 0.9


# ----------------------------
# Gamma LUT
# ----------------------------

def build_gamma_table(gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255.0 for i in range(256)]).astype("uint8")
    return table


GAMMA_TABLE = build_gamma_table(GAMMA_VALUE)


# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_frame(bgra_image: np.ndarray, depth_map: Optional[np.ndarray]) -> np.ndarray:
    """
    Lightweight preprocessing for ZED frame:
    - BGRA -> BGR
    - Denoise
    - CLAHE on L-channel
    - Gamma correction
    - Optional depth mask [DEPTH_MIN_M, DEPTH_MAX_M]
    """
    # BGRA -> BGR
    bgr = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)

    # Denoising
    if USE_FAST_NL_MEANS:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
    else:
        bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

    # CLAHE on L-channel
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, lb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, lb))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma correction
    bgr = cv2.LUT(bgr, GAMMA_TABLE)

    # Optional depth mask
    if depth_map is not None:
        valid = (depth_map > 0.0) & np.isfinite(depth_map)
        in_range = (depth_map >= DEPTH_MIN_M) & (depth_map <= DEPTH_MAX_M) & valid
        mask = in_range.astype(np.uint8)
        mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
        bgr = bgr * mask_3c

    return bgr


# ----------------------------
# MoveNet MultiPose TFLite
# ----------------------------

def load_movenet_multipose_tflite(model_path: str):
    """
    Load MoveNet MultiPose TFLite model and return:
    - interpreter
    - input_details
    - output_details

    The model reports a default shape of [1,1,1,3], so we explicitly
    resize it to [1,256,256,3] (MoveNet expected input).
    """
    print(f"Loading MoveNet MultiPose TFLite from: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    # Get original input info
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    orig_shape = input_details[0]["shape"]
    print("Original TFLite input:", orig_shape, input_details[0]["dtype"])

    # Desired MoveNet input shape
    desired_shape = [1, 256, 256, 3]

    # If shape is not already [1,256,256,3], resize it
    if list(orig_shape) != desired_shape:
        print("Resizing TFLite input tensor to", desired_shape)
        interpreter.resize_tensor_input(input_index, desired_shape)

    # Now allocate tensors with the new shape
    interpreter.allocate_tensors()

    # Refresh details after allocation
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite input (after resize):", input_details[0]["shape"], input_details[0]["dtype"])

    return interpreter, input_details, output_details


def run_movenet_multipose(
    interpreter,
    input_details,
    output_details,
    frame_bgr: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Run MoveNet MultiPose TFLite on a single BGR frame.

    Returns:
        List of dict:
        {
            "keypoints": [(u, v, score), ... 17 entries],
            "instance_score": float,
            "bbox": (ymin, xmin, ymax, xmax) in pixels
        }
    """
    h, w, _ = frame_bgr.shape

    # BGR -> RGB
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize & pad to 256x256, keeping aspect ratio
    # We use tf.image for simplicity; it's CPU-only here and small.
    image_tf = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    resized = tf.image.resize_with_pad(image_tf, 256, 256)
    resized_np = resized.numpy()

    # Prepare input according to TFLite model dtype
    inp_info = input_details[0]
    inp_index = inp_info["index"]
    inp_dtype = inp_info["dtype"]  # usually np.float32 or np.uint8 / np.int32

    if inp_dtype == np.float32:
        # Normalize to [0,1]
        input_data = resized_np.astype(np.float32) / 255.0
    elif inp_dtype == np.uint8:
        input_data = resized_np.astype(np.uint8)
    elif inp_dtype == np.int32:
        input_data = resized_np.astype(np.int32)
    else:
        # Fallback: cast to model's dtype without scaling
        input_data = resized_np.astype(inp_dtype)

    # Add batch dim
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference
    interpreter.set_tensor(inp_index, input_data)
    interpreter.invoke()

    # Assume first (and only) output is keypoints_with_scores
    out_info = output_details[0]
    out_data = interpreter.get_tensor(out_info["index"])  # shape: (1, 6, 56)
    keypoints_with_scores = out_data
    detections = keypoints_with_scores[0]  # shape: (6, 56)

    persons: List[Dict[str, Any]] = []

    for person_data in detections:
        instance_score = float(person_data[51])
        if instance_score < MIN_INSTANCE_SCORE:
            continue

        # 17 keypoints
        kps = []
        for k in range(17):
            kp_y = float(person_data[k * 3 + 0]) * h
            kp_x = float(person_data[k * 3 + 1]) * w
            kp_score = float(person_data[k * 3 + 2])

            u = int(np.clip(kp_x, 0, w - 1))
            v = int(np.clip(kp_y, 0, h - 1))
            kps.append((u, v, kp_score))

        # Bounding box y1,x1,y2,x2 (normalized)
        ymin = float(person_data[52]) * h
        xmin = float(person_data[53]) * w
        ymax = float(person_data[54]) * h
        xmax = float(person_data[55]) * w

        persons.append(
            {
                "keypoints": kps,
                "instance_score": instance_score,
                "bbox": (ymin, xmin, ymax, xmax),
            }
        )

    return persons


# ----------------------------
# Pose Skeleton Drawing (COCO 17 kp)
# ----------------------------

POSE_CONNECTIONS = [
    (5, 6),      # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (11, 12),         # hips
    (5, 11), (11, 13), (13, 15),  # left leg
    (6, 12), (12, 14), (14, 16),  # right leg
]


def draw_pose_skeleton(frame: np.ndarray, keypoints, color: tuple[int, int, int]):
    h, w, _ = frame.shape

    # Joints
    for (u, v, kp_score) in keypoints:
        if kp_score < MIN_KEYPOINT_SCORE:
            continue
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(frame, (u, v), 3, COLOR_SKELETON_POINT, -1, lineType=cv2.LINE_AA)

    # Bones
    for i1, i2 in POSE_CONNECTIONS:
        u1, v1, s1 = keypoints[i1]
        u2, v2, s2 = keypoints[i2]
        if s1 < MIN_KEYPOINT_SCORE or s2 < MIN_KEYPOINT_SCORE:
            continue
        if 0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h:
            cv2.line(frame, (u1, v1), (u2, v2), color, 2, lineType=cv2.LINE_AA)


# ----------------------------
# Eligibility / facing per person
# ----------------------------

def analyze_person(
    person: Dict[str, Any],
    point_cloud: np.ndarray,
    depth_map: np.ndarray,
    frame_shape,
) -> Dict[str, Any]:

    """
    Analyze one MoveNet person + ZED point cloud to produce:
    - depth_z (meters): robust estimate from local neighborhood in stereo depth
    - x_m (3D X if available, otherwise 2D-based fallback)
    - z_m (3D Z)
    - facing / not-facing
    - eligibility flag
    """
    h, w, _ = frame_shape
    kps = person["keypoints"]
    instance_score = person["instance_score"]

    # COCO indices
    NOSE = 0
    L_SHOULDER = 5
    R_SHOULDER = 6

    u_l, v_l, s_l = kps[L_SHOULDER]
    u_r, v_r, s_r = kps[R_SHOULDER]
    u_n, v_n, s_n = kps[NOSE]

    # Basic visibility on key joints
    mean_kp = (s_l + s_r + s_n) / 3.0
    if s_l < MIN_KEYPOINT_SCORE or s_r < MIN_KEYPOINT_SCORE or s_n < MIN_KEYPOINT_SCORE:
        person.update(
            {
                "cx": 0,
                "cy": 0,
                "x_m": 0.0,
                "z_m": float("inf"),
                "depth_z": float("inf"),
                "facing": False,
                "mean_kp_score": mean_kp,
                "eligible": False,
            }
        )
        return person

    # Shoulder midpoint as torso center (2D)
    cx_f = (u_l + u_r) / 2.0
    cy_f = (v_l + v_r) / 2.0
    cx = int(np.clip(cx_f, 0, w - 1))
    cy = int(np.clip(cy_f, 0, h - 1))

    # ---- Robust depth/X from local stereo neighborhood ----
    # point_cloud: H x W x 4 [X,Y,Z,W]; Z is depth from stereo
    Z = point_cloud[:, :, 2]
    X = point_cloud[:, :, 0]

    # 9x9 patch around (cx, cy)
    R = 4  # radius
    y0 = max(0, cy - R)
    y1 = min(h, cy + R + 1)
    x0 = max(0, cx - R)
    x1 = min(w, cx + R + 1)

    patch_Z = Z[y0:y1, x0:x1]
    patch_X = X[y0:y1, x0:x1]

    valid = np.isfinite(patch_Z) & (patch_Z > 0.0)

    if valid.any():
        # Use median of valid depth to be robust to outliers
        depth_z = float(np.median(patch_Z[valid]))
        # Use X at those same valid locations
        x_m_3d = float(np.median(patch_X[valid]))
        depth_valid = True
        z_m = depth_z  # for consistency
    else:
        depth_valid = False
        depth_z = float("inf")
        z_m = float("inf")
        x_m_3d = 0.0

    # Horizontal X used for the eyes:
    #   - If depth is valid, use real 3D X from stereo
    #   - Otherwise, 2D image-based fallback
    if depth_valid and np.isfinite(x_m_3d):
        x_m = x_m_3d
    else:
        # 2D fallback: map pixel X to [-MAX_X_RANGE_M, +MAX_X_RANGE_M]
        norm_pix_x = (cx - (w / 2.0)) / (w / 2.0)  # -1 .. 1
        x_m = float(norm_pix_x * MAX_X_RANGE_M)

    # ---- Facing approximation from shoulders ----
    dx = float(u_r - u_l)
    dy = float(v_r - v_l)
    if abs(dx) < 10:
        slope = float("inf")
    else:
        slope = dy / dx
    shoulders_horizontal = abs(slope) < 0.5

    center_x_image = w / 2.0
    shoulders_center = (u_l + u_r) / 2.0
    symmetry_ok = abs(shoulders_center - center_x_image) < (w * 0.35)

    facing = shoulders_horizontal and symmetry_ok

    # Eligibility: visible, decent instance score, roughly facing
    eligible = (
        facing
        and (mean_kp >= MIN_MEAN_KP_SCORE)
        and (instance_score >= MIN_INSTANCE_SCORE)
    )

    person.update(
        {
            "cx": cx,
            "cy": cy,
            "x_m": x_m,
            "z_m": z_m,
            "depth_z": depth_z,
            "facing": facing,
            "mean_kp_score": mean_kp,
            "eligible": eligible,
        }
    )
    return person




# ----------------------------
# Eye Offsets & Rendering
# ----------------------------

def compute_pupil_offsets_for_slots(
    slot_persons: List[Optional[Dict[str, Any]]],
    previous_offsets: np.ndarray,
) -> np.ndarray:
    """
    Compute smoothed horizontal pupil offsets for each of the 3 eye pairs.
    """
    new_offsets = previous_offsets.copy().astype(np.float32)

    for i in range(3):
        p = slot_persons[i]
        if p is None:
            target = 0.0
        else:
            x_m = float(p.get("x_m", 0.0))
            norm_x = max(-1.0, min(1.0, x_m / MAX_X_RANGE_M))
            target = norm_x * MAX_PUPIL_OFFSET

        new_offsets[i] = (
            previous_offsets[i] * (1.0 - PUPIL_SMOOTHING_ALPHA)
            + target * PUPIL_SMOOTHING_ALPHA
        )

    new_offsets = np.clip(new_offsets, -MAX_PUPIL_OFFSET, MAX_PUPIL_OFFSET)
    return new_offsets


def render_eyes_canvas(pupil_offsets: np.ndarray) -> np.ndarray:
    """
    Render 3 pairs of eyes with given horizontal offsets (pixels) per pair.
    """
    canvas = np.zeros((EYE_CANVAS_HEIGHT, EYE_CANVAS_WIDTH, 3), dtype=np.uint8)
    center_y = EYE_CANVAS_HEIGHT // 2
    num_pairs = 3

    for i in range(num_pairs):
        pair_center_x = int((EYE_CANVAS_WIDTH * (2 * i + 1)) / (2 * num_pairs))
        left_center = (int(pair_center_x - EYE_HORIZONTAL_SPACING / 2), center_y)
        right_center = (int(pair_center_x + EYE_HORIZONTAL_SPACING / 2), center_y)

        # Sclera
        cv2.circle(canvas, left_center, EYE_SCLERA_RADIUS, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, right_center, EYE_SCLERA_RADIUS, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Pupils
        offset = float(pupil_offsets[i])
        left_pupil_center = (int(left_center[0] + offset), left_center[1])
        right_pupil_center = (int(right_center[0] + offset), right_center[1])

        cv2.circle(canvas, left_pupil_center, EYE_PUPIL_RADIUS, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, right_pupil_center, EYE_PUPIL_RADIUS, (0, 0, 0), -1, lineType=cv2.LINE_AA)

    return canvas


# ----------------------------
# Main
# ----------------------------

def main():
    print("Starting Padayani ZED + MoveNet MultiPose TFLite Prototype...")

    # 1. ZED init
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # For original ZED (ZED 1), NEURAL can be flaky. Use PERFORMANCE or QUALITY.
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    # Set a generous depth range; we’ll handle filtering ourselves.
    init_params.depth_minimum_distance = 0.3  # meters
    init_params.depth_maximum_distance = 10.0  # meters

    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_stabilization = 80
    init_params.enable_image_enhancement = True
    init_params.sdk_verbose = True



    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        sys.exit(1)

    cam_info = zed.get_camera_information()
    print("Camera model:", cam_info.camera_model)
    print("Serial:", cam_info.serial_number)

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_as_static = True
    zed.enable_positional_tracking(tracking_params)

    # No ZED body tracking (ZED 1 not supported).

    image_mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    depth_mat = sl.Mat()

    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    # 2. Load MoveNet MultiPose TFLite
    interpreter, input_details, output_details = load_movenet_multipose_tflite(MOVENET_TFLITE_PATH)
    print("MoveNet MultiPose TFLite loaded.")

    frame_idx = 0
    pupil_offsets = np.zeros(3, dtype=np.float32)

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            frame_idx += 1

            # ZED data
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZ)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)  # NEW

            image_bgra = image_mat.get_data()
            point_cloud = point_cloud_mat.get_data()       # H x W x 4 [X,Y,Z,W]
            depth_map = depth_mat.get_data()               # H x W float32 (meters)


            # For debugging: show raw ZED BGR image (no preprocessing)
            display_frame = cv2.cvtColor(image_bgra, None)
            h, w, _ = display_frame.shape

            # Run MoveNet MultiPose
            persons_raw = run_movenet_multipose(
                interpreter,
                input_details,
                output_details,
                display_frame,
            )

            persons_analyzed = []
            for p in persons_raw:
                pa = analyze_person(p, point_cloud, display_frame.shape)
                persons_analyzed.append(pa)

            # Filter + sort eligible persons by depth
            eligible_persons = [p for p in persons_analyzed if p["eligible"]]
            eligible_persons.sort(key=lambda p: p["depth_z"])

            # If no one passes the eligibility test but we DO have detections,
            # pick the best person by instance score so the eyes still do something.
            if not eligible_persons and persons_analyzed:
                best = max(persons_analyzed, key=lambda p: p["instance_score"])
                eligible_persons = [best]


            # Selection logic for slots (3 eye pairs)
            slot_persons: List[Optional[Dict[str, Any]]] = [None, None, None]

            if len(eligible_persons) == 1:
                slot_persons = [eligible_persons[0]] * 3
            elif 2 <= len(eligible_persons) <= 3:
                for i in range(len(eligible_persons)):
                    slot_persons[i] = eligible_persons[i]
            elif len(eligible_persons) > 3:
                closest_three = eligible_persons[:3]
                for i in range(3):
                    slot_persons[i] = closest_three[i]

            # Update pupil offsets
            pupil_offsets = compute_pupil_offsets_for_slots(slot_persons, pupil_offsets)

            # Mark selected persons
            selected_set = set(id(p) for p in slot_persons if p is not None)

            # Draw skeletons + labels
            for p in persons_analyzed:
                keypoints = p["keypoints"]
                depth_z = p["depth_z"]
                cx, cy = p["cx"], p["cy"]
                facing = p["facing"]
                mean_kp = p["mean_kp_score"]
                inst_score = p["instance_score"]
                eligible = p["eligible"]

                if id(p) in selected_set:
                    color = COLOR_SELECTED
                    status = "SELECTED"
                elif eligible:
                    color = COLOR_ELIGIBLE
                    status = "ELIGIBLE"
                else:
                    color = COLOR_IGNORED
                    status = "IGNORED"

                draw_pose_skeleton(display_frame, keypoints, color)

                facing_str = "FACING" if facing else "NOT FACING"
                if np.isfinite(depth_z):
                    label1 = f"{status} | {facing_str}"
                    label2 = f"{depth_z:.2f} m | inst {inst_score:.2f} | kp {mean_kp:.2f}"
                else:
                    label1 = f"{status} | {facing_str}"
                    label2 = f"Depth invalid | inst {inst_score:.2f} | kp {mean_kp:.2f}"

                if 0 <= cx < w and 0 <= cy < h:
                    cv2.putText(
                        display_frame, label1, (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        display_frame, label2, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA
                    )

            # Render eyes canvas
            eyes_canvas = render_eyes_canvas(pupil_offsets)

            # Show
            cv2.imshow("ZED Padayani Feed (MoveNet MultiPose TFLite)", display_frame)
            cv2.imshow("Padayani Eyes", eyes_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        print("Shutting down cleanly...")
        cv2.destroyAllWindows()
        try:
            zed.disable_positional_tracking()
        except Exception:
            pass
        zed.close()


if __name__ == "__main__":
    """
    Setup:

    1. Place movenet_multipose_lightning.tflite in the same folder as this script
       or update MOVENET_TFLITE_PATH above.
    2. Install deps in your venv:

           pip install numpy==1.26.4 opencv-python tensorflow

       (plus the ZED SDK Python bindings).
    3. Run:

           python padayani_zed_movenet_tflite.py
    """
    main()
