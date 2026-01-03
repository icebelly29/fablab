import cv2
import numpy as np

def detect_edges(image):
    """Performs the edge detection pipeline."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def main():
    """Opens a camera feed and displays live edge detection."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Live camera feed initiated. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Get edge detection results
        edges = detect_edges(frame)

        # To display side-by-side, we need the frames to be the same type.
        # Canny returns a single-channel (grayscale) image, so we convert it to BGR.
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Horizontally stack the original and the edge-detected frames
        combined_display = np.hstack((frame, edges_bgr))

        # Display the resulting frame
        cv2.imshow('Live Edge Detection (Original vs. Edges)', combined_display)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")

if __name__ == '__main__':
    main()
