
import asyncio
import json
import cv2
import numpy as np
import pyzed.sl as sl
import websockets
import mediapipe as mp

# --- Configuration ---
ROWS = 8
COLS = 13
WEBSOCKET_URI = "ws://localhost:8080"
DEBUG_WINDOW_NAME = "Echo - Debug View"
INSTALLATION_WINDOW_NAME = "Echo - Installation View"

# --- Visualization Configuration ---
WITHER_COLOR = (50, 50, 50)       # Dark Gray
BLOOM_COLOR = (100, 255, 255)     # Bright Yellow/Cyan
CANVAS_SIZE = (600, 1000, 3)
FLOWER_RADIUS = 20
H_PADDING = (CANVAS_SIZE[1] - (COLS * FLOWER_RADIUS * 2)) // 2
V_PADDING = (CANVAS_SIZE[0] - (ROWS * FLOWER_RADIUS * 2)) // 2


# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Kalman Filter ---
def create_kalman_filter():
    """Creates and configures a Kalman filter for 2D point tracking."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.05
    return kf

# --- WebSocket Communication ---
async def send_websocket_message(websocket, row, col, state):
    message = json.dumps({"row": row, "col": col, "state": state})
    print(f"Sending: {message}")
    await websocket.send(message)

# --- Visualization ---
def draw_debug_view(frame, active_cell, hand_landmarks):
    """Draws the grid, highlights the active cell, and shows hand landmarks."""
    h, w, _ = frame.shape
    cell_height = h // ROWS
    cell_width = w // COLS

    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if active_cell:
        row, col = active_cell
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (col * cell_width, row * cell_height),
            ((col + 1) * cell_width, (row + 1) * cell_height),
            (0, 255, 0), -1
        )
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    for i in range(1, ROWS):
        cv2.line(frame, (0, i * cell_height), (w, i * cell_height), (200, 200, 200), 1)
    for i in range(1, COLS):
        cv2.line(frame, (i * cell_width, 0), (i * cell_width, h), (200, 200, 200), 1)
    
    return frame

def draw_installation_view(grid_states):
    """Creates a visualization of the flower grid state."""
    canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            # Offset every other row for an argyle-like appearance
            offset = FLOWER_RADIUS if r % 2 == 1 else 0
            center_x = H_PADDING + (c * FLOWER_RADIUS * 2) + FLOWER_RADIUS + offset
            center_y = V_PADDING + (r * FLOWER_RADIUS * 2) + FLOWER_RADIUS

            color = BLOOM_COLOR if grid_states[r, c] == 1 else WITHER_COLOR
            cv2.circle(canvas, (center_x, center_y), FLOWER_RADIUS, color, -1)
            cv2.circle(canvas, (center_x, center_y), FLOWER_RADIUS, (100, 100, 100), 1) # Border

    return canvas

# --- Main Application Logic ---
async def main():
    # ZED Initialization
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 0

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera. Is it connected?")
        return

    # State, Filter, and Window Initialization
    kf = create_kalman_filter()
    last_cell = None
    grid_states = np.zeros((ROWS, COLS), dtype=np.uint8) # Master state tracker
    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    cv2.namedWindow(DEBUG_WINDOW_NAME)
    cv2.namedWindow(INSTALLATION_WINDOW_NAME)

    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print(f"Successfully connected to {WEBSOCKET_URI}")
            
            while zed.is_opened():
                if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue

                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_bgra = image.get_data()
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                results = hands.process(frame_rgb)
                current_cell = None
                active_landmarks = None

                if results.multi_hand_landmarks:
                    active_landmarks = results.multi_hand_landmarks[0]
                    index_tip = active_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    kf.predict()
                    measurement = np.array([index_tip.x, index_tip.y], dtype=np.float32)
                    corrected = kf.correct(measurement.reshape(2, 1))
                    smooth_x, smooth_y = corrected[0, 0], corrected[1, 0]

                    col = min(COLS - 1, max(0, int(smooth_x * COLS)))
                    row = min(ROWS - 1, max(0, int(smooth_y * ROWS)))
                    current_cell = (row, col)

                if current_cell != last_cell:
                    if last_cell is not None:
                        await send_websocket_message(websocket, last_cell[0], last_cell[1], 0)
                        grid_states[last_cell[0], last_cell[1]] = 0 # Update master state
                    
                    if current_cell is not None:
                        await send_websocket_message(websocket, current_cell[0], current_cell[1], 1)
                        grid_states[current_cell[0], current_cell[1]] = 1 # Update master state
                    
                    last_cell = current_cell
                
                # Visualization
                debug_frame = draw_debug_view(frame_bgr, current_cell, active_landmarks)
                installation_frame = draw_installation_view(grid_states)

                cv2.imshow(DEBUG_WINDOW_NAME, debug_frame)
                cv2.imshow(INSTALLATION_WINDOW_NAME, installation_frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    if last_cell is not None:
                        await send_websocket_message(websocket, last_cell[0], last_cell[1], 0)
                    break
    
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError):
        print(f"Connection to {WEBSOCKET_URI} failed. Is the server running?")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        print("Releasing resources...")
        hands.close()
        zed.close()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    asyncio.run(main())
