from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv11n-pose model
model = YOLO('yolo11n-pose.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 0)
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
