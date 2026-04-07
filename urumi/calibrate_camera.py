import cv2
import numpy as np
import time

# Measurement constants
MARKER_ID = 0
MARKER_SIZE_MM = 19.0
TARGET_PPM = 10.0 # Force the flattened image to have exactly 10 pixels per mm

def get_aruco_components():
    """Compatibility wrapper for different OpenCV versions"""
    try:
        # OpenCV 4.7+
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return dictionary, parameters, detector
    except AttributeError:
        # OpenCV 4.6 and older
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        return dictionary, parameters, None

def calculate_pixels_per_mm(image):
    dictionary, parameters, detector = get_aruco_components()

    # Detect the markers
    if detector is not None:
        # OpenCV 4.7+
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        # OpenCV 4.6 and older
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    # Check if we detected any markers
    if ids is not None:
        for i in range(len(ids)):
            # Look for our specified marker ID
            if ids[i][0] == MARKER_ID:
                # The corners of the marker are in corners[i][0]
                # Order is Top-Left, Top-Right, Bottom-Right, Bottom-Left
                pts = corners[i][0]
                
                # Calculate the euclidean distance for all 4 sides of the square
                side_lengths = [
                    np.linalg.norm(pts[0] - pts[1]), # Top width
                    np.linalg.norm(pts[1] - pts[2]), # Right height
                    np.linalg.norm(pts[2] - pts[3]), # Bottom width
                    np.linalg.norm(pts[3] - pts[0])  # Left height
                ]
                
                # Average the side lengths to reduce measurement error
                avg_length_px = sum(side_lengths) / 4.0
                
                # Calculate Raw pixels per metric (PPM)
                pixels_per_mm = avg_length_px / MARKER_SIZE_MM
                
                # --- FLATTENING (HOMOGRAPHY) LOGIC ---
                # We know the marker is perfectly square. Let's force it to be a perfect square
                # in the image using TARGET_PPM to lock the pixel density across the whole plane.
                center = pts.mean(axis=0)
                half_w = (MARKER_SIZE_MM * TARGET_PPM) / 2.0
                
                dst_pts = np.array([
                    [center[0] - half_w, center[1] - half_w], # Top-Left
                    [center[0] + half_w, center[1] - half_w], # Top-Right
                    [center[0] + half_w, center[1] + half_w], # Bottom-Right
                    [center[0] - half_w, center[1] + half_w]  # Bottom-Left
                ], dtype=np.float32)

                # Compute perspective transform matrix
                homography_matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst_pts)
                
                return pixels_per_mm, pts, homography_matrix
    
    return None, None, None

def find_working_camera():
    """Iterate through V4L2 indices on Raspberry Pi to find the actual streaming node."""
    print("Auto-detecting available camera indices...")
    for index in range(10):  # Check 0 through 9
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            # Wait a tiny bit and try to pull a frame
            time.sleep(0.5)
            ret, _ = cap.read()
            if ret:
                print(f"-> Success! Valid camera stream found at index {index}")
                cap.release()
                return index
        cap.release()
    return -1

def main():
    print("Initializing camera feed...")
    
    # Auto-detect the right index
    cam_index = find_working_camera()
    
    if cam_index == -1:
        print("\nERROR: Could not find any standard V4L2 camera streams.")
        print("Because you are on a modern Raspberry Pi OS (Bullseye/Bookworm),")
        print("the traditional V4L2 driver is replaced by the 'libcamera' stack.")
        print("\nTo fix this without changing the code, run the script via the compatibility layer:")
        print("    libcamerify python calibrate_camera.py")
        print("\nAlternatively, ensure your ribbon cable is completely seated.")
        return
        
    # Enforcing the Video For Linux 2 (V4L2) backend (cv2.CAP_V4L2) and setting
    # an explicit resolution resolves this memory allocation pipeline failure.
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Warming up camera sensor, please wait 2 seconds...")
    time.sleep(2.0)

    print(f"Looking for 4x4 ArUco Marker ID: {MARKER_ID} ({MARKER_SIZE_MM}x{MARKER_SIZE_MM}mm)")
    print("Press 'q' to quit.")
    
    failed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            if failed_frames > 5:
                print("Failed to grab frames consistently. Check if the correct camera index is used.")
                break
            print("Missed frame, retrying...")
            time.sleep(0.5)
            continue
            
        # Reset counter on successful frame
        failed_frames = 0
            
        ppm, marker_corners, M = calculate_pixels_per_mm(frame)
        
        if ppm is not None:
            # 1. FLAT FLATTENED OUTPUT
            # Warp the entire frame using the homography matrix
            h, w = frame.shape[:2]
            flattened = cv2.warpPerspective(frame, M, (w, h))

            # Draw on original frame
            int_corners = np.int32(marker_corners)
            cv2.polylines(frame, [int_corners], True, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw PPM: {ppm:.2f} px/mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Showing Flattened Perspective...",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw grid/info on flattened frame
            cv2.putText(flattened, f"LOCKED PPM: {TARGET_PPM} px/mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Flattened (Bird's Eye)", flattened)
        else:
            cv2.putText(frame,f"Waiting for marker id {MARKER_ID}...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Destroy the flattened window if marker is lost to avoid stale frames
            try:
                cv2.destroyWindow("Flattened (Bird's Eye)")
            except:
                pass

        # Draw a central crosshair on the raw frame for physical alignment
        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        cross_size = 15
        cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 255), 1)  # Yellow horizontal
        cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 255), 1)  # Yellow vertical
        # Small center dot
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        # Show the primary window
        cv2.imshow("Raw Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nShutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
