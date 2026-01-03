import cv2
from ultralytics import YOLO
import numpy as np

def is_person_sitting(keypoints, conf_threshold=0.5):
    """
    Analyzes keypoints to determine if a person is sitting.
    Returns:
        - True if the person is likely sitting.
        - False if the person is likely standing.
        - None if not enough keypoints are visible to make a determination.
    """
    try:
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_knee = keypoints[13]
        r_knee = keypoints[14]
    except IndexError:
        return None

    if (l_shoulder[2] < conf_threshold or r_shoulder[2] < conf_threshold or
            l_hip[2] < conf_threshold or r_hip[2] < conf_threshold or
            l_knee[2] < conf_threshold or r_knee[2] < conf_threshold):
        return None

    avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    avg_knee_y = (l_knee[1] + r_knee[1]) / 2
    torso_height = abs(avg_hip_y - avg_shoulder_y)
    
    if avg_hip_y > avg_knee_y - (torso_height * 0.2):
        return True
    else:
        return False

def draw_skeleton(frame, keypoints, confidence_threshold=0.5):
    """Draws the skeleton and keypoints on the frame."""
    skeleton_pairs = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    for i, j in skeleton_pairs:
        kp1, kp2 = keypoints[i], keypoints[j]
        if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
            cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (255, 255, 0), 2)

def main():
    """
    Main function to run webcam-based sitting detection with lock-on tracking.
    """
    BENCH_ROI = (200, 150, 610, 480)
    locked_target_id = None
    lost_target_frames = 0
    LOCK_GRACE_PERIOD = 10 # Wait for 10 frames before giving up the lock

    try:
        model = YOLO('yolo_nano/yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        return

    print("Starting webcam sitting detection...")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break
            
            frame = cv2.flip(frame, 0) # Mirror the frame

            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

            found_locked_target_in_frame = False
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                all_keypoints = results[0].keypoints.data.cpu().numpy()

                if locked_target_id is not None:
                    for person_box, person_id, person_keypoints in zip(boxes, ids, all_keypoints):
                        if person_id == locked_target_id:
                            found_locked_target_in_frame = True
                            lost_target_frames = 0 # Reset grace period counter
                            
                            sitting_status = is_person_sitting(person_keypoints)
                            status_text = "SITTING" if sitting_status else "STANDING" if sitting_status is False else "UNCERTAIN"
                            color = (0, 0, 255) if sitting_status else (255, 165, 0) if sitting_status is False else (0, 255, 255)
                            
                            label = f"ID: {person_id} - {status_text} [LOCKED]"
                            cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), color, 2)
                            cv2.putText(frame, label, (person_box[0], person_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            draw_skeleton(frame, person_keypoints)
                            
                            center_x = (person_box[0] + person_box[2]) // 2
                            center_y = (person_box[1] + person_box[3]) // 2
                            cv2.circle(frame, (center_x, center_y), 7, color, -1)
                            
                            break
                else:
                    for person_box, person_id, person_keypoints in zip(boxes, ids, all_keypoints):
                        center_x = (person_box[0] + person_box[2]) // 2
                        center_y = (person_box[1] + person_box[3]) // 2
                        if (BENCH_ROI[0] < center_x < BENCH_ROI[2] and 
                            BENCH_ROI[1] < center_y < BENCH_ROI[3]):
                            print(f"Acquired new target in ROI. Locking onto ID: {person_id}")
                            locked_target_id = person_id
                            break

            if locked_target_id is not None and not found_locked_target_in_frame:
                lost_target_frames += 1
                if lost_target_frames > LOCK_GRACE_PERIOD:
                    print(f"Lost track of locked target ID: {locked_target_id} after {LOCK_GRACE_PERIOD} frames. Searching for new target.")
                    locked_target_id = None
                    lost_target_frames = 0
            
            # --- Visualization ---
            cv2.rectangle(frame, (BENCH_ROI[0], BENCH_ROI[1]), (BENCH_ROI[2], BENCH_ROI[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Bench ROI", (BENCH_ROI[0], BENCH_ROI[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            status_display = f"LOCKED ON: {locked_target_id}" if locked_target_id else "SEARCHING IN ROI"
            cv2.putText(frame, status_display, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Webcam Sitting Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Closing script...")
        cap.release()
        cv2.destroyAllWindows()
        print("Script finished.")

if __name__ == "__main__":
    main()
