import cv2
from ultralytics import YOLO
from enum import Enum
import time
import random

# State machine enumeration
class State(Enum):
    """
    Defines the states for our sensor fusion logic.
    """
    IDLE = 1
    ACQUISITION = 2
    TRACKING = 3

class MockPIRSensor:
    """
    A placeholder class to simulate a PIR sensor, activated by user input.
    """
    def __init__(self):
        self._is_active = False
        self._last_activation_time = 0

    def trigger(self):
        """ Manually trigger the sensor. """
        self._is_active = True
        self._last_activation_time = time.time()
        print("PIR TRIGGERED (MANUAL)")

    def read(self) -> bool:
        """ Returns True if sensor is active. Deactivates after a short period. """
        if self._is_active and time.time() - self._last_activation_time > 2: # Active for 2 seconds
            self._is_active = False
        return self._is_active

def main():
    """
    Main function to run the sensor fusion sitting detection.
    """
    # --- Configuration ---
    state = State.IDLE
    pir_sensor = MockPIRSensor()
    tracked_person_id = None
    
    # --- IMPORTANT: Adjust these coordinates to define the bench area in your video feed ---
    # Format: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    BENCH_ROI = (150, 250, 550, 450)
    
    # --- YOLOv8 Model Setup ---
    # Using the pose estimation model found in your 'yolo_nano' directory
    try:
        model = YOLO('yolo_nano/yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure 'yolo_nano/yolov8n-pose.pt' exists.")
        return

    # --- Video Capture ---
    # Using a test video from your project. Change to 0 for webcam.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        return

    print("Starting Sensor Fusion State Machine...")
    print("Press 'q' to quit, 't' to manually trigger PIR sensor.")

    # --- Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Resetting to start.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # --- State Machine Logic ---
        if state == State.IDLE:
            if pir_sensor.read():
                print("State Change: IDLE -> ACQUISITION")
                state = State.ACQUISITION

            results = model.track(frame, persist=True, verbose=False, tracker="C:/Python/Python311/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml")

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                print(f"ACQUISITION: Detected {len(ids)} persons.")

                for box, person_id in zip(boxes, ids):
                    # Calculate center point of the detected person
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2
                    
                    print(f"  Person ID: {person_id}, BBox: {box}, Center: ({center_x}, {center_y})")
                    print(f"  BENCH_ROI: {BENCH_ROI}")

                    # If a person's center is inside the Bench ROI, lock them
                    if BENCH_ROI[0] < center_x < BENCH_ROI[2] and BENCH_ROI[1] < center_y < BENCH_ROI[3]:
                        tracked_person_id = person_id
                        state = State.TRACKING
                        print(f"State Change: ACQUISITION -> TRACKING (Locked on ID: {tracked_person_id})")
                        break # Lock onto the first person found and exit the loop
                    else:
                        print(f"  Person ID: {person_id} center NOT in BENCH_ROI.")
            else:
                print("ACQUISITION: No persons detected with IDs.")

        elif state == State.TRACKING:
            # Continue tracking the locked person
            results = model.track(frame, persist=True, verbose=False, tracker="C:/Python/Python311/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml")
            person_found_in_frame = False

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, person_id in zip(boxes, ids):
                    if person_id == tracked_person_id:
                        person_found_in_frame = True
                        # Draw bounding box for the tracked person
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {person_id} - SITTING", (box[0], box[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break
            
            # EXIT STATE: If the tracked person is lost, return to IDLE
            if not person_found_in_frame:
                print(f"State Change: TRACKING -> IDLE (Tracked ID {tracked_person_id} lost)")
                tracked_person_id = None
                state = State.IDLE

        # --- Visualization ---
        # Draw the Bench ROI
        cv2.rectangle(frame, (BENCH_ROI[0], BENCH_ROI[1]), (BENCH_ROI[2], BENCH_ROI[3]), (255, 128, 0), 2)
        cv2.putText(frame, "Bench ROI", (BENCH_ROI[0], BENCH_ROI[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        
        # Display current state text
        cv2.putText(frame, f"State: {state.name}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Sensor Fusion Sitting Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('t'): # 't' for trigger
            pir_sensor.trigger()
        
        time.sleep(0.01) # Small delay to ensure frame rendering and input capture

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == "__main__":
    main()
