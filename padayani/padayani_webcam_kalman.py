import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import os
import csv

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
MODEL_NAME = "yolov8n-pose.pt" # Using the specific .pt file
FRAME_DOWNSCALE = 1.0 # Factor to downscale frame for faster processing
INFERENCE_EVERY_N_FRAMES = 2 # Perform pose detection every N frames

# Region of Interest (ROI) for the bench
# These are placeholder values and should be adjusted for the specific camera setup.
# Assumes a 1280x720 frame, then scaled by FRAME_DOWNSCALE.
_downscale_factor = FRAME_DOWNSCALE
ROI_X_MIN = int(300 * _downscale_factor)
ROI_Y_MIN = int(480 * _downscale_factor)
ROI_X_MAX = int(980 * _downscale_factor)
ROI_Y_MAX = int(680 * _downscale_factor)

# Locking and Tracking Parameters
MIN_LOCK_SIT_TIME = 2.0 # Seconds a person must be validly sitting to be locked
ID_PERSISTENCE_TIMEOUT = 1.5 # Seconds to keep tracking a lost ID before unlocking
FACE_SYMMETRY_THRESHOLD = 30 # Max pixel difference for nose-shoulder symmetry

# Heuristic Parameters
SITTING_ASPECT_RATIO_THRESHOLD = 1.8 # Max bbox height/width ratio for sitting
KNEE_HIP_Y_TOLERANCE_RATIO = 0.1 # How far knees can be below hips (ratio of ROI height)

# Logging
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "locks.csv")

# ============================================================================
# KALMAN FILTER CLASS
# ============================================================================
class KalmanFilter:
    """A simple Kalman filter for 2D point tracking."""
    def __init__(self, dt=1.0, u_x=0, u_y=0, std_acc=1., std_meas=3.):
        self.dt = dt
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        # Process noise covariance
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]], dtype=np.float32) * std_acc**2
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * std_meas**2
        # State estimate and covariance
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

# ============================================================================
# VISUALIZATION CONSTANTS & FUNCTIONS
# ============================================================================
# COCO keypoint indices
SKELETON_EDGES = [
    (0, 1, (255, 128, 0)), (0, 2, (255, 128, 0)), (1, 3, (255, 128, 0)), (2, 4, (255, 128, 0)), # Head
    (5, 6, (0, 255, 0)),   # Torso-Shoulders
    (5, 7, (0, 128, 255)), (7, 9, (0, 128, 255)),   # Left Arm
    (6, 8, (255, 0, 128)), (8, 10, (255, 0, 128)),  # Right Arm
    (11, 12, (0, 255, 0)), # Torso-Hips
    (5, 11, (0, 255, 128)),(6, 12, (128, 255, 0)),  # Torso-Sides
    (11, 13, (0, 128, 255)), (13, 15, (0, 128, 255)),# Left Leg
    (12, 14, (255, 0, 128)), (14, 16, (255, 0, 128)) # Right Leg
]
KEYPOINT_COLORS = [ (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (0, 255, 0), (0, 255, 0), (0, 128, 255), (255, 0, 128), (0, 128, 255), (255, 0, 128), (0, 255, 0), (0, 255, 0), (0, 128, 255), (255, 0, 128), (0, 128, 255), (255, 0, 128) ]

def draw_skeleton(frame, kpts):
    """Draws colored skeleton edges and keypoints."""
    for start_idx, end_idx, color in SKELETON_EDGES:
        p1 = get_keypoint(kpts, start_idx)
        p2 = get_keypoint(kpts, end_idx)
        if p1 is not None and p2 is not None:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    for i in range(kpts.shape[1]):
        pt = get_keypoint(kpts, i)
        if pt is not None:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, KEYPOINT_COLORS[i], -1)

# ============================================================================
# HELPER & HEURISTIC FUNCTIONS
# ============================================================================
def get_keypoint(keypoints, index):
    """Safely get a keypoint by index."""
    if keypoints is None or keypoints.data.shape[1] <= index: return None
    pt = keypoints.data[0, index, :].cpu().numpy()
    if pt[2] > 0.3: return pt[:2]
    return None

def is_sitting_heuristic(box, kpts):
    """Checks if a person is likely sitting based on pose."""
    l_hip, r_hip = get_keypoint(kpts, 11), get_keypoint(kpts, 12)
    
    # 1. Hips must be visible and within the bench ROI.
    if l_hip is None and r_hip is None: return False
    
    hip_y = l_hip[1] if l_hip is not None else r_hip[1]
    hip_x = l_hip[0] if l_hip is not None else r_hip[0]
    if not (ROI_X_MIN < hip_x < ROI_X_MAX and ROI_Y_MIN < hip_y < ROI_Y_MAX):
        return False
        
    # 2. Bounding box aspect ratio should be like a sitting person.
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w == 0 or h / w > SITTING_ASPECT_RATIO_THRESHOLD:
        return False

    # 3. Knees should not be significantly lower than hips.
    l_knee, r_knee = get_keypoint(kpts, 13), get_keypoint(kpts, 14)
    if l_knee is not None and r_knee is not None:
        knee_y_avg = (l_knee[1] + r_knee[1]) / 2
        roi_height = ROI_Y_MAX - ROI_Y_MIN
        if (knee_y_avg - hip_y) > (roi_height * KNEE_HIP_Y_TOLERANCE_RATIO):
            return False

    return True

def is_facing_camera_heuristic(kpts):
    """Checks if a person is likely facing the camera using nose-shoulder symmetry."""
    nose, l_shoulder, r_shoulder = get_keypoint(kpts, 0), get_keypoint(kpts, 5), get_keypoint(kpts, 6)

    if nose is None or l_shoulder is None or r_shoulder is None: return False
    if l_shoulder[0] >= r_shoulder[0]: return False # Shoulders are crossed, invalid pose

    dist_l_nose = abs(nose[0] - l_shoulder[0])
    dist_r_nose = abs(r_shoulder[0] - nose[0])
    
    return abs(dist_l_nose - dist_r_nose) < FACE_SYMMETRY_THRESHOLD

def setup_logging():
    """Create log directory and file if they don't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "lock_start_time", "lock_end_time", "duration_seconds"])

def log_lock_event(start_time, end_time):
    """Append a lock event to the CSV file."""
    duration = (end_time - start_time).total_seconds()
    timestamp = datetime.datetime.now().isoformat()
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, start_time.isoformat(), end_time.isoformat(), f"{duration:.2f}"])

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    print("Initializing...")
    setup_logging()
    
    try:
        model = YOLO(MODEL_NAME)
        print(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return
        
    H, W, _ = frame.shape
    scaled_w, scaled_h = int(W * FRAME_DOWNSCALE), int(H * FRAME_DOWNSCALE)

    status = "SEARCHING"
    locked_person_id, lock_start_time = None, None
    last_seen_time, potential_candidates = {}, {}
    kf, frame_count, prev_time = None, 0, time.time()

    window_name = "Padayani Webcam Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, scaled_w, scaled_h)

    print("Initialization complete. Starting main loop...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_processed = cv2.resize(frame, (scaled_w, scaled_h))
        current_time = time.time()
        
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            results = model.track(frame_processed, persist=True, verbose=False, classes=[0])
            
            detected_ids_this_frame = set()
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes, ids, keypoints_list = results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu(), results[0].keypoints
                detected_ids_this_frame.update(ids.numpy().astype(int))

                if status == "SEARCHING":
                    valid_candidates_this_frame = set()
                    for i, (box, person_id) in enumerate(zip(boxes, ids)):
                        box = box.numpy().astype(int)
                        person_id = person_id.numpy().astype(int)
                        kpts = keypoints_list[i]
                        
                        if is_sitting_heuristic(box, kpts) and is_facing_camera_heuristic(kpts):
                            valid_candidates_this_frame.add(person_id)
                            if person_id not in potential_candidates:
                                potential_candidates[person_id] = current_time
                                print(f"[SEARCHING] New candidate {person_id} is sitting in ROI.")
                            
                            if current_time - potential_candidates[person_id] >= MIN_LOCK_SIT_TIME:
                                status, locked_person_id = "LOCKED", person_id
                                lock_start_time = datetime.datetime.now()
                                last_seen_time[locked_person_id] = current_time
                                
                                cx = int((box[0] + box[2]) / 2)
                                cy = int((box[1] + box[3]) / 2)
                                kf = KalmanFilter(dt=1/15.0)
                                kf.x[:2] = np.array([[cx], [cy]])
                                
                                potential_candidates.clear()
                                print(f"*** LOCKED ON ID: {locked_person_id} ***")
                                break
                    
                    stale_candidates = [pid for pid in potential_candidates if pid not in valid_candidates_this_frame]
                    for pid in stale_candidates:
                        del potential_candidates[pid]
                        print(f"[SEARCHING] Candidate {pid} no longer meets criteria.")

                if status == "LOCKED" and locked_person_id is not None:
                    if kf: kf.predict()
                    
                    if locked_person_id in detected_ids_this_frame:
                        idx = (ids == locked_person_id).nonzero(as_tuple=True)[0][0]
                        box = boxes[idx].numpy().astype(int)
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        
                        if kf: kf.update(np.array([[cx], [cy]]))
                        last_seen_time[locked_person_id] = current_time
                    else:
                        if current_time - last_seen_time.get(locked_person_id, 0) > ID_PERSISTENCE_TIMEOUT:
                            print(f"UNLOCK: Lost track of person {locked_person_id}.")
                            status = "UNLOCKING"
            
            if status == "UNLOCKING":
                print(f"Logging lock event for ID {locked_person_id}.")
                if lock_start_time: log_lock_event(lock_start_time, datetime.datetime.now())
                locked_person_id, lock_start_time, kf = None, None, None
                status = "SEARCHING"

        cv2.rectangle(frame_processed, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (0, 255, 255), 2)
        
        if 'results' in locals() and results and results[0].boxes and results[0].boxes.id is not None:
            boxes, ids, keypoints_list = results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu(), results[0].keypoints
            for box, person_id, kpts, conf in zip(boxes, ids, keypoints_list, results[0].boxes.conf.cpu().numpy()):
                person_id = int(person_id)
                box = [int(p) for p in box]
                box_color = (0, 255, 0)

                if person_id == locked_person_id:
                    box_color = (0, 0, 255)
                    if kf:
                        pred = kf.x
                        cv2.circle(frame_processed, (int(pred[0]), int(pred[1])), 8, (255, 0, 0), -1)
                
                if status == "SEARCHING" and person_id in potential_candidates:
                    elapsed = current_time - potential_candidates[person_id]
                    remaining = max(0, MIN_LOCK_SIT_TIME - elapsed)
                    timer_text = f"{remaining:.1f}s"
                    (w, h), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    text_pos = (box[0], box[1] - 35)
                    cv2.rectangle(frame_processed, (text_pos[0], text_pos[1] - h - 4), (text_pos[0] + w, text_pos[1] + 4), (0,0,0), -1)
                    cv2.putText(frame_processed, timer_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                cv2.rectangle(frame_processed, (box[0], box[1]), (box[2], box[3]), box_color, 2)
                cv2.putText(frame_processed, f"ID: {person_id} Conf: {conf:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                cv2.circle(frame_processed, (cx, cy), 5, (255, 255, 255), -1)
                draw_skeleton(frame_processed, kpts)

        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        status_text = f"Status: {status}" + (f" (ID: {locked_person_id})" if locked_person_id is not None else "")
        cv2.putText(frame_processed, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_processed, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, frame_processed)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27:
            if status == "LOCKED" and lock_start_time:
                log_lock_event(lock_start_time, datetime.datetime.now())
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()
