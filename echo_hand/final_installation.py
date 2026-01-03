import asyncio
import json
import cv2
import numpy as np
import pyzed.sl as sl
import websockets
import mediapipe as mp
import time
from enum import Enum, auto
from collections import deque

# --- Primary Configuration ---
ROWS, COLS = 8, 13
WEBSOCKET_URI = "ws://localhost:8080"

# --- Interaction Tuning ---
IDLE_TIMEOUT = 5.0
HOLD_TO_DRAW_TIME = 5.0
WIPE_VELOCITY_THRESHOLD = 0.8 # Normalized distance per second

# --- Window & Visualization ---
DEBUG_WINDOW = "Echo - Debug View"
INSTALLATION_WINDOW = "Echo - Installation View"
WITHER_COLOR, BLOOM_COLOR = (50, 50, 50), (100, 255, 255)
CANVAS_SIZE = (600, 1000, 3)
FLOWER_RADIUS = 20
H_PAD = (CANVAS_SIZE[1] - (COLS * FLOWER_RADIUS * 2)) // 2
V_PAD = (CANVAS_SIZE[0] - (ROWS * FLOWER_RADIUS * 2)) // 2

# --- State Machine ---
class Mode(Enum):
    IDLE = auto()
    INTERACTIVE = auto()
    WAVE = auto()
    DRAW = auto()

# --- MediaPipe & ZED Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- Helper Functions ---
async def send_message_batch(websocket, cells, action):
    if len(cells) == 0: return
    # print(f"Batch sending {len(cells)} '{action}' actions.")
    await asyncio.gather(*[send_websocket_message(websocket, r, c, action) for r, c in cells])

async def send_websocket_message(websocket, row, col, action):
    message = json.dumps({"row": int(row), "col": int(col), "action": action})
    await websocket.send(message)

def is_palm_open(hand_landmarks):
    extended_fingers = sum(1 for tip_id, pip_id in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
    ] if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y)
    return extended_fingers >= 3

def is_fist(hand_landmarks):
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

# --- Animation Logic (for background tasks) ---

async def run_idle_animation(websocket, grid_states, event):
    """Generates a calm, sine-wave based wind pattern for IDLE mode."""
    t = 0
    while True:
        try:
            await event.wait()
            t += 0.05
            cols_range, rows_range = np.arange(COLS), np.arange(ROWS)
            cols_mesh, rows_mesh = np.meshgrid(cols_range, rows_range)
            wave = np.sin(cols_mesh * 0.4 + t) + np.sin(rows_mesh * 0.6 + t * 0.8)
            
            to_bloom, to_wither = [], []
            for r in range(ROWS):
                for c in range(COLS):
                    new_state = 1 if wave[r, c] > 1.2 else 0
                    if grid_states[r, c] != new_state:
                        grid_states[r, c] = new_state
                        if new_state == 1:
                            to_bloom.append((r, c))
                        else:
                            to_wither.append((r, c))
            
            # Send changes in efficient batches
            if to_bloom: await send_message_batch(websocket, to_bloom, "bloom")
            if to_wither: await send_message_batch(websocket, to_wither, "wither")
            
            await asyncio.sleep(0.05) # Slower, more reasonable animation speed
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in idle animation: {e}")

async def run_wave_animation(websocket, grid_states, direction, event):
    try:
        await event.wait()
        cols = range(COLS) if direction == "right" else reversed(range(COLS))
        step_delay = 2.0 / COLS

        for c in cols:
            if not event.is_set(): break
            cells_to_act_on = [(r, c) for r in range(ROWS)]
            
            await send_message_batch(websocket, cells_to_act_on, "bloom")
            for r, c_ in cells_to_act_on: grid_states[r, c_] = 1
            await asyncio.sleep(step_delay)
            
            await send_message_batch(websocket, cells_to_act_on, "wither")
            for r, c_ in cells_to_act_on: grid_states[r, c_] = 0
    except asyncio.CancelledError:
        print("Wave animation task cancelled.")
    finally:
        event.clear()

# --- UI Drawing ---

def draw_ui(frame, grid_states, hand_landmarks, mode, hold_countdown=0):
    # Installation View
    canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            offset = FLOWER_RADIUS if r % 2 == 1 else 0
            center = (H_PAD + (c * 2 + 1) * FLOWER_RADIUS + offset, V_PAD + (r * 2 + 1) * FLOWER_RADIUS)
            color = BLOOM_COLOR if grid_states[r, c] else WITHER_COLOR
            cv2.circle(canvas, center, FLOWER_RADIUS, color, -1)

    # Debug View
    h, w, _ = frame.shape
    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if hold_countdown > 0:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            pos = (int(wrist.x * w), int(wrist.y * h))
            cv2.putText(frame, str(int(hold_countdown)), pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
    
    cv2.putText(frame, f"MODE: {mode.name}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return frame, canvas

# --- Main Application ---
async def main():
    zed = sl.Camera()
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30, sdk_verbose=0)
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera."); return

    grid_states = np.zeros((ROWS, COLS), dtype=np.uint8)
    image, kf = sl.Mat(), cv2.KalmanFilter(4, 2) # kf is not used, can be removed if not needed later
    
    # State & Performance Vars
    current_mode, last_hand_time = Mode.IDLE, time.time()
    hold_start_time = 0
    hand_pos_history = deque(maxlen=10)
    frame_counter, last_results = 0, None
    PROCESS_EVERY_N_FRAMES = 2
    
    # --- Async Task Management ---
    idle_event, wave_event = asyncio.Event(), asyncio.Event()
    idle_task, wave_task = None, None

    try:
        async with websockets.connect(WEBSOCKET_URI, ping_interval=5, ping_timeout=10) as websocket:
            print(f"Connected. Starting in {current_mode.name} mode.")
            idle_task = asyncio.create_task(run_idle_animation(websocket, grid_states, idle_event))
            if current_mode == Mode.IDLE: idle_event.set()

            while zed.is_opened():
                if zed.grab() != sl.ERROR_CODE.SUCCESS: continue
                frame_counter += 1
                
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                
                hand_landmarks = None
                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    results = hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    last_results = results
                if last_results and last_results.multi_hand_landmarks:
                    hand_landmarks = last_results.multi_hand_landmarks[0]

                # --- State Machine Transitions ---
                if not hand_landmarks:
                    if current_mode != Mode.IDLE and (time.time() - last_hand_time > IDLE_TIMEOUT):
                        print("Transition to IDLE")
                        current_mode = Mode.IDLE
                        if wave_task and not wave_task.done(): wave_task.cancel()
                        idle_event.set()
                else:
                    last_hand_time = time.time()
                    if current_mode == Mode.IDLE:
                        print("Transition to INTERACTIVE")
                        idle_event.clear()
                        current_mode = Mode.INTERACTIVE
                        await send_message_batch(websocket, np.argwhere(grid_states == 1), "wither")
                        grid_states.fill(0)
                
                # --- Mode-Specific Logic ---
                hold_countdown = 0
                if current_mode == Mode.INTERACTIVE and hand_landmarks:
                    wrist_pos_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    hand_pos_history.append((time.time(), wrist_pos_x))
                    
                    if len(hand_pos_history) == hand_pos_history.maxlen:
                        time_delta = hand_pos_history[-1][0] - hand_pos_history[0][0]
                        dist_delta = hand_pos_history[-1][1] - hand_pos_history[0][1]
                        if time_delta > 0 and abs(dist_delta) / time_delta > WIPE_VELOCITY_THRESHOLD:
                            print("Wipe Detected! Transition to WAVE")
                            current_mode = Mode.WAVE
                            wave_task = asyncio.create_task(run_wave_animation(websocket, grid_states, "right" if dist_delta > 0 else "left", wave_event))
                            wave_event.set()
                    
                    if is_palm_open(hand_landmarks):
                        if hold_start_time == 0: hold_start_time = time.time()
                        elapsed = time.time() - hold_start_time
                        hold_countdown = HOLD_TO_DRAW_TIME - elapsed + 1
                        if elapsed > HOLD_TO_DRAW_TIME:
                            print("Hold Detected! Transition to DRAW")
                            current_mode = Mode.DRAW
                            hold_start_time = 0
                            await send_message_batch(websocket, np.argwhere(grid_states == 1), "wither")
                            grid_states.fill(0)
                    else:
                        hold_start_time = 0
                
                elif hand_landmarks and current_mode == Mode.DRAW:
                    if is_fist(hand_landmarks):
                        print("Fist Detected! Exiting DRAW mode.")
                        current_mode = Mode.INTERACTIVE
                        await send_message_batch(websocket, np.argwhere(grid_states == 1), "wither")
                        grid_states.fill(0)
                    else:
                        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        r, c = min(ROWS-1, max(0, int(tip.y*ROWS))), min(COLS-1, max(0, int(tip.x*COLS)))
                        if grid_states[r, c] == 0:
                            grid_states[r, c] = 1
                            await send_websocket_message(websocket, r, c, "bloom")
                            await asyncio.sleep(0.001)

                elif current_mode == Mode.WAVE:
                    if not wave_event.is_set():
                        print("Wave finished. Transition to INTERACTIVE")
                        current_mode = Mode.INTERACTIVE

                # --- UI Update ---
                debug_frame, install_frame = draw_ui(frame_bgr, grid_states, hand_landmarks, current_mode, hold_countdown)
                cv2.imshow(DEBUG_WINDOW, debug_frame)
                cv2.imshow(INSTALLATION_WINDOW, install_frame)
                if cv2.waitKey(1) & 0xFF == 27: break
    
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"Connection failed: {e}")
    except KeyboardInterrupt:
        print("\nProgram terminated.")
    finally:
        print("Releasing resources...")
        if idle_task: idle_task.cancel()
        if wave_task: wave_task.cancel()
        try:
            if idle_task: await idle_task
            if wave_task: await wave_task
        except asyncio.CancelledError:
            pass # Expected
        if zed.is_opened(): zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
