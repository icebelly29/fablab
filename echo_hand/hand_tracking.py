
import asyncio
import json
import cv2
import mediapipe as mp
import websockets

# --- Configuration ---
GRID_ROWS = 8
GRID_COLS = 13
WEBSOCKET_URI = "ws://localhost:8080"
WINDOW_NAME = "Echo - Interactive Art Installation"

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

async def send_websocket_message(websocket, row, col, action):
    """Formats and sends a JSON message to the WebSocket server."""
    message = json.dumps({"row": row, "col": col, "action": action})
    print(f"Sending: {message}")
    await websocket.send(message)

def draw_grid(frame, active_cell=None):
    """Draws the grid and highlights the active cell on the frame."""
    h, w, _ = frame.shape
    cell_height = h // GRID_ROWS
    cell_width = w // GRID_COLS

    # Draw grid lines
    for i in range(1, GRID_ROWS):
        cv2.line(frame, (0, i * cell_height), (w, i * cell_height), (255, 255, 255), 1)
    for i in range(1, GRID_COLS):
        cv2.line(frame, (i * cell_width, 0), (i * cell_width, h), (255, 255, 255), 1)

    # Highlight the active cell
    if active_cell:
        row, col = active_cell
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

async def main():
    """Main function to run the hand tracking and WebSocket communication."""
    last_cell = None
    active_cell = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print(f"Successfully connected to {WEBSOCKET_URI}")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and find hands
                results = hands.process(rgb_frame)

                current_cell = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Get the tip of the index finger (landmark 8)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Denormalize coordinates
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    # Map coordinates to grid
                    col = min(GRID_COLS - 1, max(0, int(index_finger_tip.x * GRID_COLS)))
                    row = min(GRID_ROWS - 1, max(0, int(index_finger_tip.y * GRID_ROWS)))
                    current_cell = (row, col)
                    active_cell = current_cell

                # --- State Change Logic ---
                if last_cell != current_cell:
                    # Wither the old cell if it exists
                    if last_cell is not None:
                        await send_websocket_message(websocket, last_cell[0], last_cell[1], "wither")
                    
                    # Bloom the new cell if it exists
                    if current_cell is not None:
                        await send_websocket_message(websocket, current_cell[0], current_cell[1], "bloom")
                    
                    last_cell = current_cell
                
                # If no hand is detected, ensure the last cell withers
                if not results.multi_hand_landmarks and last_cell is not None:
                    await send_websocket_message(websocket, last_cell[0], last_cell[1], "wither")
                    last_cell = None
                    active_cell = None


                # Draw the grid and display the frame
                display_frame = draw_grid(frame, active_cell)
                cv2.imshow(WINDOW_NAME, display_frame)

                if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                    break
    
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"Connection to {WEBSOCKET_URI} failed: {e}")
        print("Please ensure the WebSocket server is running.")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released.")

if __name__ == "__main__":
    # --- Installation Note ---
    # Please install the required libraries if you haven't already:
    # pip install opencv-python mediapipe websockets
    
    asyncio.run(main())
