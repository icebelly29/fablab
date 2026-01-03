import asyncio
import json
import cv2
import numpy as np
import pyzed.sl as sl
import websockets
import mediapipe as mp
import time
from collections import deque
from enum import Enum, auto
import random

# --- Configuration ---
ROWS = 8
COLS = 13
WEBSOCKET_URI = "ws://localhost:8080"
DEBUG_WINDOW_NAME = "Echo - Debug View"
INSTALLATION_WINDOW_NAME = "Echo - Installation View"

# --- Interaction Tuning ---
IDLE_TIMEOUT = 5.0  # seconds to wait before starting idle mode
DRAW_MODE_HOLD_TIME = 3.0  # seconds to hold open palm to enter draw mode
WIPE_TIME_THRESHOLD = 0.5  # max seconds for a wipe gesture
WIPE_DISTANCE_THRESHOLD = 0.4  # min normalized distance for a wipe

# --- Visualization Configuration ---
WITHER_COLOR = (50, 50, 50)
BLOOM_COLOR = (100, 255, 255)
CANVAS_SIZE = (600, 1000, 3)
FLOWER_RADIUS = 20
H_PADDING = (CANVAS_SIZE[1] - (COLS * FLOWER_RADIUS * 2)) // 2
V_PADDING = (CANVAS_SIZE[0] - (ROWS * FLOWER_RADIUS * 2)) // 2

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
FINGERTIP_IDS = [
    mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

class Mode(Enum):
    IDLE = auto()
    LISTENING = auto()
    DRAW_CHECK = auto()
    DRAW = auto()
    WAVE = auto()

# --- Helper Functions ---

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.05
    return kf

def is_palm_open(hand_landmarks):
    # Check if key finger tips are above their respective PIP joints (meaning extended)
    extended_fingers = 0
    # Index Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        extended_fingers += 1
    # Middle Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        extended_fingers += 1
    # Ring Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        extended_fingers += 1
    # Pinky Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        extended_fingers += 1
    
    # Palm is considered open if at least 3 of the 4 main fingers are extended
    return extended_fingers >= 3

def is_fist(hand_landmarks):
    # Check if key finger tips are below their respective PIP joints (meaning curled)
    curled_fingers = 0
    # Index Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        curled_fingers += 1
    # Middle Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        curled_fingers += 1
    # Ring Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        curled_fingers += 1
    # Pinky Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        curled_fingers += 1

    # Hand is considered a fist if at least 3 of the 4 main fingers are curled
    return curled_fingers >= 3


async def send_message_batch(websocket, cells, state):
    """Sends a batch of messages for a set of cells."""
    if len(cells) == 0: return
    tasks = [send_websocket_message(websocket, r, c, state) for r, c in cells]
    await asyncio.gather(*tasks)

async def send_websocket_message(websocket, row, col, state):
    message = json.dumps({"row": int(row), "col": int(col), "state": int(state)})
    # print(f"Sending: {message}") # Can be noisy, uncomment for debug
    await websocket.send(message)

# --- Visualization ---

def draw_ui(frame, grid_states, active_cell, hand_landmarks, mode):
    # Draw Installation View
    canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            offset = FLOWER_RADIUS if r % 2 == 1 else 0
            center_x = H_PADDING + (c * FLOWER_RADIUS * 2) + FLOWER_RADIUS + offset
            center_y = V_PADDING + (r * FLOWER_RADIUS * 2) + FLOWER_RADIUS
            color = BLOOM_COLOR if grid_states[r, c] == 1 else WITHER_COLOR
            cv2.circle(canvas, (center_x, center_y), FLOWER_RADIUS, color, -1)
            cv2.circle(canvas, (center_x, center_y), FLOWER_RADIUS, (100, 100, 100), 1)
    
    # Draw Debug View
    h, w, _ = frame.shape
    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if active_cell:
        r, c = active_cell
        cv2.rectangle(frame, (c * (w//COLS), r * (h//ROWS)), ((c+1) * (w//COLS), (r+1) * (h//ROWS)), (0, 255, 0), 2)
    
    cv2.putText(frame, f"MODE: {mode.name}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame, canvas

# --- Mode Logic ---

async def run_wind_pattern(websocket, grid_states, should_run):
    """Plays calm wind patterns (random flicker) in IDLE mode."""
    while True:
        try:
            await should_run.wait() # Wait until instructed to run

            num_flickers = random.randint(5, 15) # Flickering several flowers
            for _ in range(num_flickers):
                if not should_run.is_set(): break
                r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
                
                # Bloom
                await send_websocket_message(websocket, r, c, 1)
                grid_states[r, c] = 1
                await asyncio.sleep(random.uniform(0.1, 0.3)) # Short bloom time
                
                # Wither
                if not should_run.is_set(): break # Check again before extinguishing
                await send_websocket_message(websocket, r, c, 0)
                grid_states[r, c] = 0
                await asyncio.sleep(random.uniform(0.05, 0.15)) # Short pause

            if not should_run.is_set(): continue
            await asyncio.sleep(2) # Pause between flicker bursts

        except asyncio.CancelledError:
            print("Wind pattern task cancelled.")
            # Clear any remaining active flowers when cancelled
            for r in range(ROWS):
                for c in range(COLS):
                    if grid_states[r,c] == 1:
                        await send_websocket_message(websocket, r, c, 0)
                        grid_states[r,c] = 0
            break
        except Exception as e:
            print(f"Error in wind pattern task: {e}")
            break

async def run_wave_animation(websocket, grid_states, start_pos, end_pos):
    """Animates a wave based on a wipe gesture."""
    direction = np.array(end_pos) - np.array(start_pos)
    # Avoid division by zero if the gesture is a point
    norm = np.linalg.norm(direction)
    if norm == 0: return
    direction /= norm
    
    num_steps = 30
    for i in range(num_steps):
        wave_pos = i / (num_steps - 1)
        
        cells_to_bloom = set()
        for r in range(ROWS):
            for c in range(COLS):
                cell_norm_pos = np.array([(c + 0.5) / COLS, (r + 0.5) / ROWS])
                proj = np.dot(cell_norm_pos, direction)
                if abs(proj - wave_pos) < 0.05:
                    cells_to_bloom.add((r,c))
        
        await send_message_batch(websocket, cells_to_bloom, 1)
        for r_b, c_b in cells_to_bloom: grid_states[r_b, c_b] = 1
        await asyncio.sleep(0.02)

        await send_message_batch(websocket, cells_to_bloom, 0)
        for r_b, c_b in cells_to_bloom: grid_states[r_b, c_b] = 0

# --- Main Application ---
async def main():
    zed = sl.Camera()
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30, sdk_verbose=0)
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera."); return

    kf = create_kalman_filter()
    grid_states = np.zeros((ROWS, COLS), dtype=np.uint8)
    image = sl.Mat()
    
    # State Machine Variables
    current_mode = Mode.IDLE
    last_hand_time = time.time()
    mode_entry_time = 0
    hand_pos_history = deque(maxlen=10)
    wipe_gesture_data = None
    
    # Async task management
    idle_event = asyncio.Event()
    idle_task = None

    # Performance Optimization
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 2
    last_results = None

    try:
        async with websockets.connect(WEBSOCKET_URI, ping_interval=5, ping_timeout=20) as websocket:
            print(f"Connected. Starting in {current_mode.name} mode.")

            # Create and start the idle task. It will wait until its event is set.
            idle_task = asyncio.create_task(run_wind_pattern(websocket, grid_states, idle_event))
            if current_mode == Mode.IDLE:
                idle_event.set()

            while zed.is_opened():
                if zed.grab() != sl.ERROR_CODE.SUCCESS: continue
                
                frame_counter += 1
                
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                
                active_cell = None
                hand_detected = False
                
                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    last_results = results
                    hand_detected = results.multi_hand_landmarks is not None

                    if not hand_detected and current_mode != Mode.IDLE:
                        if time.time() - last_hand_time > IDLE_TIMEOUT:
                            print("Transition to IDLE")
                            current_mode = Mode.IDLE
                            idle_event.set()
                    
                    if hand_detected:
                        last_hand_time = time.time()
                        if current_mode == Mode.IDLE:
                            print("Transition to LISTENING")
                            current_mode = Mode.LISTENING
                            mode_entry_time = time.time()
                            idle_event.clear()
                            await send_message_batch(websocket, np.argwhere(grid_states == 1), 0)
                            grid_states.fill(0)
                else:
                    hand_detected = last_results.multi_hand_landmarks is not None if last_results else False

                if current_mode != Mode.IDLE and hand_detected and last_results and last_results.multi_hand_landmarks:
                    hand_landmarks = last_results.multi_hand_landmarks[0]
                    
                    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                        wrist_pos = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        hand_pos_history.append((time.time(), (wrist_pos.x, wrist_pos.y)))

                    if current_mode == Mode.LISTENING:
                        if is_palm_open(hand_landmarks):
                            current_mode = Mode.DRAW_CHECK
                            mode_entry_time = time.time()
                        
                        if len(hand_pos_history) == hand_pos_history.maxlen:
                            start_time, start_pos = hand_pos_history[0]
                            end_time, end_pos = hand_pos_history[-1]
                            if (end_time - start_time) < WIPE_TIME_THRESHOLD and np.linalg.norm(np.array(end_pos) - np.array(start_pos)) > WIPE_DISTANCE_THRESHOLD:
                                print("Wipe Detected! Transition to WAVE")
                                wipe_gesture_data = (start_pos, end_pos)
                                current_mode = Mode.WAVE
                                hand_pos_history.clear()

                    elif current_mode == Mode.DRAW_CHECK:
                        if not is_palm_open(hand_landmarks):
                            current_mode = Mode.LISTENING
                        elif time.time() - mode_entry_time > DRAW_MODE_HOLD_TIME:
                            print("Transition to DRAW")
                            current_mode = Mode.DRAW
                            await send_message_batch(websocket, np.argwhere(grid_states == 1), 0)
                            grid_states.fill(0)

                    elif current_mode == Mode.DRAW:
                        if is_fist(hand_landmarks):
                            print("Fist detected. Exiting DRAW mode.")
                            current_mode = Mode.LISTENING
                            await send_message_batch(websocket, np.argwhere(grid_states == 1), 0)
                            grid_states.fill(0)
                        else:
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            kf.predict()
                            corrected = kf.correct(np.array([index_tip.x, index_tip.y], dtype=np.float32).reshape(2,1))
                            r, c = min(ROWS-1, max(0, int(corrected[1,0]*ROWS))), min(COLS-1, max(0, int(corrected[0,0]*COLS)))
                            active_cell = (r, c)
                            if grid_states[r, c] == 0:
                                grid_states[r, c] = 1
                                await send_websocket_message(websocket, r, c, 1)
                                await asyncio.sleep(0) # Yield control to event loop

                    elif current_mode == Mode.WAVE:
                        await run_wave_animation(websocket, grid_states, wipe_gesture_data[0], wipe_gesture_data[1])
                        print("Wave finished. Transition to LISTENING")
                        current_mode = Mode.LISTENING

                debug_frame, install_frame = draw_ui(frame_bgr, grid_states, active_cell, 
                                                     last_results.multi_hand_landmarks[0] if last_results and last_results.multi_hand_landmarks else None, 
                                                     current_mode)
                cv2.imshow(DEBUG_WINDOW_NAME, debug_frame)
                cv2.imshow(INSTALLATION_WINDOW_NAME, install_frame)

                if cv2.waitKey(1) & 0xFF == 27: break
    
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"Connection to {WEBSOCKET_URI} failed: {e}")
    except KeyboardInterrupt:
        print("\nProgram terminated.")
    finally:
        print("Releasing resources...")
        if idle_task:
            idle_task.cancel()
        if zed.is_opened(): zed.close()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())