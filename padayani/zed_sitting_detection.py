import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import math
from collections import deque
import logging

# --- Parameters ---
CONFIDENCE_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.7  # Thigh/Torso length ratio. Lower means more likely to be sitting.
# New threshold for high-angle view. Checks vertical distance between knee and hip.
# Smaller value means knee and hip are more vertically aligned (as seen from above).
KNEE_HIP_ALIGNMENT_THRESHOLD = 0.4 # Normalized by torso length

# Hysteresis/Smoothing
PREDICTION_BUFFER_SIZE = 15
MAJORITY_VOTE_THRESHOLD = 10 # Must have at least this many votes to switch state

def setup_logging():
    """Sets up logging to a file."""
    logging.basicConfig(filename='debug_log_zed.txt', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', filemode='w')
    logging.info("--- Starting New ZED Debug Session with Detailed Heuristics ---")

def get_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    if p1 is None or p2 is None:
        return float('inf') # Use infinity to indicate missing data for distance calculations
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_midpoint(p1, p2):
    """Calculate the midpoint between two 2D points."""
    if p1 is None or p2 is None:
        return None
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def get_kp(keypoints, index):
    """Safely get a keypoint tuple (x, y, conf) if its confidence is above the threshold."""
    if index < len(keypoints) and keypoints[index][2] > CONFIDENCE_THRESHOLD:
        return keypoints[index].cpu().numpy()
    return None

def main():
    setup_logging()
    
    # 1. Load Model
    model = YOLO('yolov8n-pose.pt')

    # 2. Initialize ZED Camera
    print("Initializing ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        logging.error(f"Failed to open ZED camera: {err}")
        exit(1) # Replaced sys.exit(1)

    print("ZED camera opened successfully.")
    logging.info("ZED camera opened successfully.")

    image_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    # 3. Initialize Hysteresis Buffer
    predictions = deque(maxlen=PREDICTION_BUFFER_SIZE)
    confirmed_status = "Unknown"

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame_bgra = image_mat.get_data()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            
            frame = cv2.resize(frame, (640*2, 480*2))
            frame = cv2.flip(frame, 0)
            
            results = model(frame, verbose=False)
            annotated_frame = frame.copy()
            
            person_detected = False
            proposed_status = "Unknown" # Default to unknown initially
            
            # Debug values
            torso_len, thigh_len, ratio, norm_alignment = 0.0, 0.0, -1.0, -1.0 # Initialize as floats
            is_sitting_by_ratio_flag = False
            is_sitting_by_alignment_flag = False

            if results and len(results) > 0 and results[0].keypoints and results[0].boxes:
                if results[0].boxes[0].conf > CONFIDENCE_THRESHOLD:
                    person_detected = True
                    annotated_frame = results[0].plot()
                    keypoints = results[0].keypoints.data[0] # [x, y, conf]

                    # Keypoint definitions (using helper function)
                    l_sh = get_kp(keypoints, 5) # Left Shoulder
                    r_sh = get_kp(keypoints, 6) # Right Shoulder
                    l_hip = get_kp(keypoints, 11) # Left Hip
                    r_hip = get_kp(keypoints, 12) # Right Hip
                    l_knee = get_kp(keypoints, 13) # Left Knee
                    r_knee = get_kp(keypoints, 14) # Right Knee
                    
                    # --- Calculate Torso Length ---
                    shoulder_mid = get_midpoint(l_sh[:2] if l_sh is not None else None, r_sh[:2] if r_sh is not None else None)
                    hip_mid = get_midpoint(l_hip[:2] if l_hip is not None else None, r_hip[:2] if r_hip is not None else None)
                    torso_len = get_distance(shoulder_mid, hip_mid)

                    # --- Select Best Leg Data for Calculations ---
                    thigh_hip_kp_val, thigh_knee_kp_val = None, None # actual [x,y,conf] values for hip and knee
                    
                    l_knee_conf = l_knee[2] if l_knee is not None else 0
                    r_knee_conf = r_knee[2] if r_knee is not None else 0

                    if l_knee_conf > r_knee_conf and l_hip is not None:
                        thigh_hip_kp_val, thigh_knee_kp_val = l_hip, l_knee
                    elif r_knee_conf > 0 and r_hip is not None:
                        thigh_hip_kp_val, thigh_knee_kp_val = r_hip, r_knee
                    
                    # Calculate 2D thigh length (foreshortened view)
                    thigh_len = get_distance(thigh_hip_kp_val[:2] if thigh_hip_kp_val is not None else None, 
                                             thigh_knee_kp_val[:2] if thigh_knee_kp_val is not None else None)

                    # --- Apply Heuristics ---
                    is_valid_body = torso_len < float('inf') and torso_len > 0

                    if is_valid_body:
                        # Heuristic 1: Thigh/Torso Ratio (good for foreshortening)
                        if thigh_len < float('inf'):
                            ratio = thigh_len / torso_len
                            is_sitting_by_ratio_flag = (ratio < RATIO_THRESHOLD)
                        
                        # Heuristic 2: Knee-Hip Vertical Alignment (good for high angle)
                        if thigh_hip_kp_val is not None and thigh_knee_kp_val is not None:
                            vertical_knee_hip_dist = abs(thigh_knee_kp_val[1] - thigh_hip_kp_val[1])
                            norm_alignment = vertical_knee_hip_dist / torso_len
                            is_sitting_by_alignment_flag = (norm_alignment < KNEE_HIP_ALIGNMENT_THRESHOLD)

                    # --- Final Decision ---
                    if is_sitting_by_ratio_flag and is_sitting_by_alignment_flag:
                        proposed_status = "SITTING"
                    else:
                        proposed_status = "STANDING"


            predictions.append(proposed_status)
            sitting_votes = predictions.count("SITTING")
            standing_votes = predictions.count("STANDING")

            if sitting_votes >= MAJORITY_VOTE_THRESHOLD:
                confirmed_status = "SITTING"
            elif standing_votes >= MAJORITY_VOTE_THRESHOLD:
                confirmed_status = "STANDING"
            # Otherwise, keep the last confirmed status until a majority is reached


            # --- Logging (Detailed) ---
            log_msg = (
                f"Status: {confirmed_status} (Proposed: {proposed_status}), "
                f"Torso: {torso_len:.2f}, Thigh: {thigh_len:.2f}, Ratio: {ratio:.2f} (Thresh: < {RATIO_THRESHOLD:.2f}), "
                f"AlignDist: {norm_alignment:.2f} (Thresh: < {KNEE_HIP_ALIGNMENT_THRESHOLD:.2f}), "
                f"RatioFlag: {is_sitting_by_ratio_flag}, AlignFlag: {is_sitting_by_alignment_flag}, "
                f"Votes (Sit/Stand): {sitting_votes}/{standing_votes}"
            )
            logging.info(log_msg)

            # --- On-screen debug text remains for user convenience ---
            color = (0, 0, 255) if confirmed_status == "SITTING" else (0, 255, 0)
            if confirmed_status == "Unknown": color = (0, 255, 255)
            cv2.putText(annotated_frame, f"STATE: {confirmed_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            y_pos = 60
            debug_texts = [
                f"Torso Length: {torso_len:.2f}",
                f"Thigh Length (2D): {thigh_len:.2f}",
                f"Thigh/Torso Ratio: {ratio:.2f} (Thresh < {RATIO_THRESHOLD:.2f})",
                f"Knee/Hip Align (Norm): {norm_alignment:.2f} (Thresh < {KNEE_HIP_ALIGNMENT_THRESHOLD:.2f})",
                f"Ratio Flag: {is_sitting_by_ratio_flag}, Align Flag: {is_sitting_by_alignment_flag}",
                f"Votes (Sit/Stand): {sitting_votes}/{standing_votes} (of {PREDICTION_BUFFER_SIZE})"
            ]
            for text in debug_texts:
                cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 25

            if not person_detected:
                predictions.clear()
                confirmed_status = "Unknown"
                cv2.putText(annotated_frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('High-Angle Sitting Detection (ZED)', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        print("Shutting down ZED camera...")
        zed.close()
        cv2.destroyAllWindows()
        logging.info("--- ZED Debug Session Ended ---")

if __name__ == "__main__":
    main()
