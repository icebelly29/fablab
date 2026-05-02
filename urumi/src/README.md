# Vision-Guided Gantry System

A high-performance, real-time computer vision pipeline for a Raspberry Pi 4-based CNC gantry system. This software processes live camera feeds to autonomously trace object perimeters and provides a live HTML5 WebSockets Digital Twin UI.

## How It Works

The system is built on a hybrid concurrency architecture to guarantee low latency on the Raspberry Pi 4:

1. **Vision Worker (Multiprocessing):**
   Runs entirely on an isolated CPU core. It continually captures frames (via `camera.py`), identifies object edges (`vision.py` using Canny/Contours), calculates proportional cross-track errors to keep the camera centered, and computes the tangent vector to dictate the next G-code motion `(dx, dy)`.
2. **Main Controller (Asyncio):**
   Handles non-blocking I/O operations. It manages the `GrblController` to stream calculated `(dx, dy)` commands to the CNC without overflowing the 128-byte serial buffer. It concurrently runs a FastAPI server.
3. **Web UI Digital Twin:**
   FastAPI streams real-time telemetry and compressed image tiles to the browser via WebSockets at 10-20Hz. The frontend performs **Kinematic Stitching**: instead of heavy image processing, the UI maps raw frames onto an HTML5 Canvas using absolute machine coordinates.

## Project Structure

*   `main.py`: The entry point that orchestrates multiprocessing queues and asyncio loops.
*   `camera.py`: The video capture loop (simulated to work on Windows for testing, ready for Picamera2 on Pi).
*   `vision.py`: The mathematical core containing tangent vector extraction and closed-loop error correction.
*   `grbl.py`: Asyncio-based non-blocking serial communication driver.
*   `server.py`: FastAPI server bridging backend telemetry to the frontend via WebSockets.
*   `static/`: Contains `index.html` and `app.js` which render the 950x700mm live map.

## Setup & Running

### 1. Requirements

Modern operating systems (like Raspberry Pi OS Bookworm) enforce PEP 668, which requires installing Python packages inside a Virtual Environment to avoid breaking system packages.

Make sure your terminal is inside the `src` directory, then run:

```bash
# 1. Create a virtual environment named 'venv'
python3 -m venv venv

# 2. Activate the virtual environment
# On Linux/Raspberry Pi/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt
```

> **Note on PyTurboJPEG:** The requirements file includes `PyTurboJPEG` for hardware-accelerated JPEG encoding. If you get an error installing it, you can safely comment it out in `requirements.txt`; `camera.py` has a built-in fallback to standard OpenCV encoding.

### 2. Running on Windows (Simulation Mode)

You can test the entire UI and vision pipeline directly on your PC using your webcam:
```bash
python main.py
```
*   The script will attempt to connect to COM3 (this will fail gracefully and the server will still run if no machine is connected).
*   It uses `cv2.VideoCapture(0)` to read your webcam.
*   It injects simulated `(X, Y)` machine coordinates back to the UI based on the edge detection algorithm.
*   Open your browser and navigate to [http://localhost:8000](http://localhost:8000) to view the live Digital Twin UI.

### 3. Deploying to Raspberry Pi 4

Before running on the Pi, make the following hardware-specific adjustments:
1.  **Serial Port:** Open `main.py` and change the GRBL port from `'COM3'` to `'/dev/ttyUSB0'` (or your specific Pi serial port).
2.  **Camera Backend:** Open `camera.py`. For the PiHQ camera on modern Pi OS, consider replacing `cv2.VideoCapture(0)` with a `Picamera2` integration to ensure 30fps zero-copy memory access.
3.  **Run:** `python main.py`
