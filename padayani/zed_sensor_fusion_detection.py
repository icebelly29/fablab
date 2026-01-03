import cv2
import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
from enum import Enum
import time
import random

# State machine enumeration (same as before)
class State(Enum):
    IDLE = 1
    ACQUISITION = 2
    TRACKING = 3

# Manual-trigger mock PIR sensor (same as before)
class MockPIRSensor:
    def __init__(self):
        self._triggered = False

    def trigger(self):
        self._triggered = True
        print("PIR TRIGGERED (MANUAL)")

    def read(self) -> bool:
        """
        Reads the sensor status. This is a single-shot read that consumes the trigger.
        """
        if self._triggered:
            self._triggered = False  # Consume the trigger
            return True
        return False

def main():
    """
    Main function to run the sensor fusion sitting detection with a ZED camera.
    """
    # --- ZED Camera Setup ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Not used, but good practice

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {err}")
        print("Please check that the ZED camera is connected and the ZED SDK is installed.")
        return

    runtime_parameters = sl.RuntimeParameters()
    image_mat = sl.Mat() # Mat to hold the ZED image data

    # --- Configuration ---
    state = State.IDLE
    pir_sensor = MockPIRSensor()
    tracked_person_id = None
    
    # --- IMPORTANT: Adjust these coordinates for your ZED camera's view ---
    BENCH_ROI = (250, 300, 1000, 650) # (x1, y1, x2, y2) for HD720 resolution
    
    # --- YOLOv8 Model Setup ---
    try:
        model = YOLO('yolo_nano/yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        zed.close()
        return

    print("Starting Sensor Fusion State Machine with ZED Camera...")
    print("Press 'q' to quit, 't' to manually trigger PIR sensor.")

    # --- Main Loop ---
    try:
        while True:
            # Grab a new frame from the ZED camera
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve the left camera image
                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                
                # Convert ZED's BGRA sl.Mat to a BGR numpy array for OpenCV and YOLO
                frame = image_mat.get_data()[:, :, :3].copy()
                frame = np.ascontiguousarray(frame, dtype=np.uint8)

                # --- State Machine Logic (identical to the previous script) ---
                if state == State.IDLE:
                    if pir_sensor.read():
                        print("State Change: IDLE -> ACQUISITION")
                        state = State.ACQUISITION

                elif state == State.ACQUISITION:
                    # Check for a new trigger to cancel acquisition
                    if pir_sensor.read():
                        print("State Change: ACQUISITION -> IDLE (Cancelled by new PIR trigger)")
                        state = State.IDLE
                        continue

                    results = model.track(frame, persist=True, verbose=False, tracker="C:/Python/Python311/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml")

                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        
                        for box, person_id in zip(boxes, ids):
                            center_x = (box[0] + box[2]) // 2
                            center_y = (box[1] + box[3]) // 2
                            
                            if BENCH_ROI[0] < center_x < BENCH_ROI[2] and BENCH_ROI[1] < center_y < BENCH_ROI[3]:
                                tracked_person_id = person_id
                                state = State.TRACKING
                                print(f"State Change: ACQUISITION -> TRACKING (Locked on ID: {tracked_person_id})")
                                break

                elif state == State.TRACKING:
                    results = model.track(frame, persist=True, verbose=False, tracker="C:/Python/Python311/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml")
                    person_found_in_frame = False

                    if results[0].boxes.id is not None:
                        for box, person_id in zip(results[0].boxes.xyxy.cpu().numpy().astype(int), results[0].boxes.id.cpu().numpy().astype(int)):
                            if person_id == tracked_person_id:
                                person_found_in_frame = True
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                cv2.putText(frame, f"ID: {person_id} - SITTING", (box[0], box[1] - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                break
                    
                    if not person_found_in_frame:
                        print(f"State Change: TRACKING -> IDLE (Tracked ID {tracked_person_id} lost)")
                        tracked_person_id = None
                        state = State.IDLE

                # --- Visualization ---
                cv2.rectangle(frame, (BENCH_ROI[0], BENCH_ROI[1]), (BENCH_ROI[2], BENCH_ROI[3]), (255, 128, 0), 2)
                cv2.putText(frame, "Bench ROI", (BENCH_ROI[0], BENCH_ROI[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                
                cv2.putText(frame, f"State: {state.name}", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                cv2.imshow('ZED Sensor Fusion Sitting Detection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('t'):
                    pir_sensor.trigger()
            
            else:
                # If grabbing a frame fails, wait briefly before trying again
                time.sleep(0.01)

    finally:
        # --- Cleanup ---
        print("Closing script...")
        zed.close()
        cv2.destroyAllWindows()
        print("Script finished.")

if __name__ == "__main__":
    main()
