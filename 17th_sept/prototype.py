import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Eye pair positions (column layout)
EYE_POSITIONS = [
    (150, 100),
    (150, 250),
    (150, 400)
]

EYE_RADIUS = 40
PUPIL_RADIUS = 15
PUPIL_RANGE = 18  # how far pupils can move inside the eye


def draw_eye_pair(frame, center_x, center_y, dx, dy):
    # Left eye
    lx, ly = center_x - 40, center_y
    # Right eye
    rx, ry = center_x + 40, center_y

    # Draw white eyeballs
    cv2.circle(frame, (lx, ly), EYE_RADIUS, (255, 255, 255), -1)
    cv2.circle(frame, (rx, ry), EYE_RADIUS, (255, 255, 255), -1)

    # Compute pupil offsets
    px = int(dx * PUPIL_RANGE)
    py = int(dy * PUPIL_RANGE)

    # Draw pupils
    cv2.circle(frame, (lx + px, ly + py), PUPIL_RADIUS, (0, 0, 0), -1)
    cv2.circle(frame, (rx + px, ry + py), PUPIL_RADIUS, (0, 0, 0), -1)


def get_target_face(results, width, height):
    """Choose the largest face (closest person)."""
    best_face = None
    best_area = 0

    if not results.detections:
        return None

    for det in results.detections:
        box = det.location_data.relative_bounding_box
        x, y, w, h = box.xmin, box.ymin, box.width, box.height
        area = w * h

        if area > best_area:
            best_area = area
            best_face = (x, y, w, h)

    if best_face is None:
        return None

    x, y, w, h = best_face

    cx = int((x + w / 2) * width)
    cy = int((y + h / 2) * height)
    return cx, cy


def normalize_direction(cx, cy, width, height):
    """Convert target pixel location into a normalized direction dx, dy."""
    dx = (cx - width / 2) / (width / 2)
    dy = (cy - height / 2) / (height / 2)

    # clamp between -1 and 1
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))

    return dx, dy


def main():
    cap = cv2.VideoCapture(0)

    eye_window = np.zeros((500, 300, 3), dtype=np.uint8)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Mediapipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            target = get_target_face(results, w, h)

            if target is not None:
                cx, cy = target
                dx, dy = normalize_direction(cx, cy, w, h)

                # Draw debug point on camera feed
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            else:
                # no face → eyes go to neutral position
                dx, dy = 0, 0

            # Clear eyes window
            eye_window[:] = (50, 50, 50)

            # Draw all 3 pairs
            for (ex, ey) in EYE_POSITIONS:
                draw_eye_pair(eye_window, ex, ey, dx, dy)

            # Show camera feed
            cv2.imshow("Camera Feed", frame)

            # Show eyes
            cv2.imshow("Padayani Eyes", eye_window)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
