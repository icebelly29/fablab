import cv2
import time
import base64
from vision import process_edges

# Fallback to OpenCV if PyTurboJPEG is not available
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    USE_TURBO = True
except ImportError:
    USE_TURBO = False

# Set this to True on your Raspberry Pi to use hardware acceleration
USE_PICAMERA2 = True

if USE_PICAMERA2:
    from picamera2 import Picamera2

def encode_jpeg_fast(frame):
    if USE_TURBO:
        # Hardware accelerated or SIMD optimized
        return base64.b64encode(jpeg.encode(frame, quality=70)).decode('utf-8')
    else:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(buffer).decode('utf-8')

def capture_loop(cmd_queue, telemetry_queue):
    if USE_PICAMERA2:
        picam2 = Picamera2()
        # Configure for 640x480, returning BGR array to match OpenCV expectations
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
    else:
        # Using generic V4L2/DirectShow backend for testing
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mm_per_px = 0.1 # This should come from calibration
    
    # Mock machine position for UI testing since we might not have a real gantry connected
    sim_x, sim_y = 475.0, 350.0 
    
    while True:
        if USE_PICAMERA2:
            try:
                frame = picam2.capture_array()
                ret = True
            except Exception:
                ret = False
        else:
            ret, frame = cap.read()

        if not ret:
            time.sleep(0.1)
            continue
            
        dx_mm, dy_mm, vis_frame = process_edges(frame, mm_per_px=mm_per_px)
        
        # Simulate movement based on the algorithm output
        sim_x += dx_mm
        sim_y += dy_mm
        
        b64_img = encode_jpeg_fast(vis_frame)
        
        data = {
            'type': 'telemetry',
            'x': sim_x,
            'y': sim_y,
            'dx': dx_mm,
            'dy': dy_mm,
            'image': b64_img
        }
        
        if telemetry_queue.full():
            try:
                telemetry_queue.get_nowait()
            except:
                pass
                
        telemetry_queue.put(data)
