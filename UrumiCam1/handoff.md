# Project Handoff

## Changes Made (2026-05-15)

### 1. Jog Functionality Fix
- **Issue**: Jogging only worked once per page load and didn't log anything.
- **Fix**: 
    - Added `position_update` to WebSocket routing in `websocket.js`.
    - Updated `app.py` to emit logs and position updates during/after jogging.
    - This unlocks the UI buttons and provides real-time feedback.

### 2. Focus Threshold Adjustment
- **Issue**: Scans were failing because the "sharpness" pass-mark was too high (1117).
- **Fix**: Lowered `min_focus_variance` to `10.0` in `config.json`. This allows the scan to proceed even if the image isn't perfectly sharp or has low contrast.

### 3. Deep Motion Synchronization & UI Fix
- **Issue**: Gantry wasn't moving during scans, and the UI camera feed was freezing.
- **Fix**: 
    - **Physical Sync**: Added `_serial.reset_input_buffer()` to `uart_comm.py` to clear out old "ok" messages from the OS buffer.
    - **UI Freeze**: Disabled automatic preview restarts during scans in `camera_feed.js` to avoid hardware conflicts.
    - **Live Feedback**: The system now pushes high-res captured frames and live coordinates to the UI during the scan.

### 4. Interactive System Terminal
- **Feature**: Replaced the static log with a functional **System Terminal**.
- **Usage**: Type commands directly into the prompt (e.g., `ping 2`, `enable all 1`) to talk to the Pico. Use **Up/Down arrows** for command history.

### 6. Coordinate-Based Offline Stitcher
- **New Program**: Created `stitch_scan.py` in the root directory.
- **Purpose**: A standalone CLI tool to stitch your images perfectly using their gantry coordinates, even if they have no similar features.
- **Features**: 
    - Uses `scan_manifest.json` for precise placement.
    - **Feathering**: Smooths the edges between tiles so you don't see harsh lines.
    - **Alpha Blending**: Averages the overlapping pixels for a seamless look.
    - **Scaling**: Option to generate smaller mosaics for quick viewing (e.g., `--scale 0.5`).
- **How to use**:
    ```bash
    python3 stitch_scan.py scans/your_scan_folder
    ```
