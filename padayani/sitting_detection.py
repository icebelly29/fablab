import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import logging

# --- Parameters ---
CONFIDENCE_THRESHOLD = 0.5
RATIO_THRESHOLD = 0.7
DROP_THRESHOLD_FACTOR = 0.5

# Hysteresis/Smoothing
PREDICTION_BUFFER_SIZE = 15
MAJORITY_VOTE_THRESHOLD = 10

def setup_logging():
    """Sets up logging to a file."""
    logging.basicConfig(filename='debug_log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', filemode='w')
    logging.info("--- Starting New Debug Session ---")

def get_distance(p1, p2):
    """Calculate Euclidean distance between two keypoints."""
    if p1 is None or p2 is None:
        return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_midpoint(p1, p2):
    """Calculate the midpoint between two keypoints."""
    if p1 is None or p2 is None:
        return None
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def get_kp(keypoints, index):
    """Safely get a keypoint if its confidence is above the threshold."""
    if keypoints[index][2] > CONFIDENCE_THRESHOLD:
        return keypoints[index][:2].cpu().numpy()
    return None

def main():
    setup_logging()
    
    # 1. Load Model and Camera
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)

    # 2. Initialize Hysteresis Buffer
    predictions = deque(maxlen=PREDICTION_BUFFER_SIZE)
    confirmed_status = "Unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 0)
        
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        
        person_detected = False
        proposed_status = "Unknown"
        
        torso_len, thigh_len, ratio, vertical_drop = 0, 0, -1, -1

        if results and len(results) > 0 and results[0].keypoints and results[0].boxes:
            if results[0].boxes[0].conf > CONFIDENCE_THRESHOLD:
                person_detected = True
                annotated_frame = results[0].plot()
                keypoints = results[0].keypoints.data[0]

                l_sh, r_sh = get_kp(keypoints, 5), get_kp(keypoints, 6)
                l_hip, r_hip = get_kp(keypoints, 11), get_kp(keypoints, 12)
                l_knee, r_knee = get_kp(keypoints, 13), get_kp(keypoints, 14)

                shoulder_mid = get_midpoint(l_sh, r_sh)
                hip_mid = get_midpoint(l_hip, r_hip)
                torso_len = get_distance(shoulder_mid, hip_mid)

                thigh_hip_kp, thigh_knee_kp = None, None
                l_knee_conf = keypoints[13][2] if l_knee is not None else 0
                r_knee_conf = keypoints[14][2] if r_knee is not None else 0

                if l_knee_conf > r_knee_conf:
                    thigh_hip_kp, thigh_knee_kp = l_hip, l_knee
                elif r_knee_conf > 0:
                    thigh_hip_kp, thigh_knee_kp = r_hip, r_knee
                
                thigh_len = get_distance(thigh_hip_kp, thigh_knee_kp)

                if torso_len > 1 and thigh_len < float('inf'):
                    ratio = thigh_len / torso_len
                
                if thigh_hip_kp is not None and thigh_knee_kp is not None:
                    vertical_drop = thigh_knee_kp[1] - thigh_hip_kp[1]

                proposed_status = "STANDING"
                if torso_len > 1 and ratio > 0 and ratio < RATIO_THRESHOLD:
                    proposed_status = "SITTING"
                elif torso_len > 1 and vertical_drop > 0 and vertical_drop < torso_len * DROP_THRESHOLD_FACTOR:
                    proposed_status = "SITTING"

        predictions.append(proposed_status)
        sitting_votes = predictions.count("SITTING")
        standing_votes = predictions.count("STANDING")

        if sitting_votes >= MAJORITY_VOTE_THRESHOLD:
            confirmed_status = "SITTING"
        elif standing_votes >= MAJORITY_VOTE_THRESHOLD:
            confirmed_status = "STANDING"

        # --- Logging ---
        log_msg = (
            f"Status: {confirmed_status} (Proposed: {proposed_status}), "
            f"Torso: {torso_len:.2f}, Thigh: {thigh_len:.2f}, Ratio: {ratio:.2f} (Thresh: < {RATIO_THRESHOLD}), "
            f"Drop: {vertical_drop:.2f} (Thresh: < {torso_len * DROP_THRESHOLD_FACTOR:.2f}), "
            f"Votes (Sit/Stand): {sitting_votes}/{standing_votes}"
        )
        logging.info(log_msg)

        # On-screen debug text remains for user convenience
        color = (0, 0, 255) if confirmed_status == "SITTING" else (0, 255, 0)
        if confirmed_status == "Unknown": color = (0, 255, 255)
        cv2.putText(annotated_frame, f"STATE: {confirmed_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        y_pos = 60
        debug_texts = [
            f"Torso Length: {torso_len:.2f}",
            f"Thigh Length: {thigh_len:.2f}",
            f"Thigh/Torso Ratio: {ratio:.2f} (Thresh: < {RATIO_THRESHOLD})",
            f"Vertical Drop: {vertical_drop:.2f} (Thresh: < {torso_len * DROP_THRESHOLD_FACTOR:.2f})",
            f"Votes (Sit/Stand): {sitting_votes}/{standing_votes} (of {PREDICTION_BUFFER_SIZE})"
        ]
        for text in debug_texts:
            cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 25

        if not person_detected:
            predictions.clear()
            confirmed_status = "Unknown"

        cv2.imshow('High-Angle Sitting Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("--- Session Ended ---")

if __name__ == "__main__":
    main()
