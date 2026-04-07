# Camera Calibration with ArUco Markers

A universal camera calibration sequence designed to dynamically calculate the **Pixels Per Metric (PPM)** value (specifically, Pixels per Millimeter) for any camera. This is ideal for use with a Raspberry Pi and standard webcams.

## New Features
- **Auto-Detect Camera Index**: Intelligently cycles through indices `0` to `9` to find the actual V4L2 streaming pixel node rather than crashing on dummy hardware encoders.
- **Bird's Eye Perspective Flattening**: Computes a homography matrix on the detected marker to warp the screen and create a mathematically perfect flat plane, locking the PPM identically to `10 px/1 mm` across the whole window.
- **Alignment Crosshair**: Overlays a persistent center crosshair on the raw feed to help you align your camera dead center physically.

## How It Works 

The script uses OpenCV to locate a predefined square marker in the physical world. Since we know the exact physical dimensions of this marker, the script calculates how many pixels that width occupies on the camera sensor, yielding an accurate ratio. 

### Input
- **Video Source**: A live video feed from auto-detected camera indices (defaults to `0`).
- **Physical Target**: An ArUco marker from the `DICT_4X4_50` dictionary with **ID: 0**.
- **Marker Dimensions**: The script expects the printed marker to be exactly **19mm x 19mm**.

### Output
- **Visual Overlay (Raw Feed)**: Opens a window displaying the live video feed. It draws a green bounding box over the detected marker, a center alignment crosshair, and displays the real-time "Pixels per mm" ratio on screen.
- **Visual Overlay (Flattened)**: A second "Bird's Eye" window opens whenever the marker is visible. This window cancels out perspective distortions by mathematically warping the image to guarantee that `1 mm` equals exactly `10 pixels` indiscriminately across the image plane.
- **Console Output**: Periodically prints the detected `Pixels per mm` value to the standard terminal output for easy logging.

---

## Setup & Running

### 1. Requirements
Ensure you have Python installed. The script depends on OpenCV (including its ArUco modules) and NumPy. You can install all dependencies via pip:

```bash
pip install opencv-python opencv-contrib-python numpy
```

### 2. Running the script
To execute the calibration feed, open your terminal in the same directory as the script and run:

```bash
python calibrate_camera.py
```
.0
> **IMPORTANT: For Raspberry Pi (Bullseye / Bookworm):**
> Modern Pi OS uses the `libcamera` backend, which standard OpenCV cannot interface with naturally using V4L2. If you get an error that the camera can't be opened, run the wrapper:
> ```bash
> libcamerify python calibrate_camera.py
> ```

### 3. Usage Steps
1. Once the script runs, a window labeled **Raw Camera Feed** will pop up. Align your camera using the center yellow crosshair.
2. Hold up or place your **19x19mm ArUco 4x4 ID:0 marker** in clear view of the camera.
3. A second window labeled **Flattened (Bird's Eye)** will appear when the marker locks.
4. Note down your raw PPM or verify the flattened view behavior.
5. Press the **`q`** key on your keyboard while the window is active to safely quit the utility and release the camera stream.

---

## Troubleshooting
- **Error: Could not find any standard V4L2 camera streams.**
  If you are running a Raspberry Pi OS built on Debian Bullseye or Bookworm, make sure you prefix the command with `libcamerify`.
- **Marker not detecting or flattening window shakes wildly?**
  Ensure there is decent lighting in the room and the marker has strong black-and-white contrast. Keep in mind that doing a full-frame homography on extremely small ArUco markers amplifies tiny read errors. For rock-solid flattening, standard practices use 4 markers spanning the very edge of the workbed.
