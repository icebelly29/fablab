import cv2
import numpy as np
import time
import random

detector_choice = "mediapipe"  # "haar", "mediapipe", "yolo"

if detector_choice == "haar":
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
elif detector_choice == "mediapipe":
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)
elif detector_choice == "yolo":
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.3

width, height = 800, 400
eye_radius = 60
pupil_radius = 20
left_eye_center = (int(width * 0.35), int(height * 0.5))
right_eye_center = (int(width * 0.65), int(height * 0.5))

cap = cv2.VideoCapture(0)

current_x, current_y = 0, 0
smooth_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    offset_x, offset_y = 0, 0
    face_found = False

    if detector_choice == "haar":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            cx = x + fw // 2
            cy = y + fh // 2
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            face_found = True

    elif detector_choice == "mediapipe":
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            fw = int(bbox.width * w)
            fh = int(bbox.height * h)
            cx = x + fw // 2
            cy = y + fh // 2
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            face_found = True

    elif detector_choice == "yolo":
        results = model(frame[..., ::-1])
        preds = results.pred[0]
        person_boxes = preds[preds[:, 5] == 0]
        if len(person_boxes) > 0:
            best = person_boxes[person_boxes[:, 4].argmax()]
            x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            face_found = True

    if face_found:
        nx = (cx / w) * 2 - 1
        ny = (cy / h) * 2 - 1
        max_offset = 25
        offset_x = int(nx * max_offset)
        offset_y = int(ny * max_offset)

    current_x = int(current_x + (offset_x - current_x) * smooth_factor)
    current_y = int(current_y + (offset_y - current_y) * smooth_factor)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.circle(canvas, left_eye_center, eye_radius, (0, 0, 0), 3)
    cv2.circle(canvas, right_eye_center, eye_radius, (0, 0, 0), 3)

    cv2.circle(canvas,
               (left_eye_center[0] + current_x, left_eye_center[1] + current_y),
               pupil_radius, (0, 0, 0), -1)
    cv2.circle(canvas,
               (right_eye_center[0] + current_x, right_eye_center[1] + current_y),
               pupil_radius, (0, 0, 0), -1)

    cv2.imshow("Robot Eye Simulation", canvas)
    cv2.imshow("Camera Feed + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
