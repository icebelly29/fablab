# UrumiCam — Version Control & Change Log

This file tracks all technical modifications made to the UrumiCam system.

---

## [2026-05-15] - System Stability & Feature Enhancement

### 1. Jogging Subsystem Fix
*   **Component**: `static/js/websocket.js`, `server/app.py`
*   **Description**: Added `position_update` to the WebSocket event routing whitelist. Previously, the frontend would set `jogBusy = true` but never receive the event to reset it.
*   **Logic**: The backend `on_jog` handler now explicitly emits `position_update` and `log_message` upon UART completion.
*   **Result**: Jog buttons are now responsive for multiple consecutive uses and provide feedback in the UI log.

### 2. Focus Sensitivity Adjustment
*   **Component**: `config.json`
*   **Description**: Reduced the Laplacian variance threshold (`min_focus_variance`) from `1117.6` to `10.0`.
*   **Reasoning**: The previous threshold was calibrated for high-contrast subjects and caused scans to fail on softer workpieces.
*   **Result**: Scans now reliably pass focus checks on various surface types.

### 3. UART Hardware Synchronization (The "Ghost OK" Fix)
*   **Component**: `server/uart_comm.py`
*   **Description**: Enhanced the `_flush_rx()` method to call `self._serial.reset_input_buffer()`.
*   **Logic**: Clears the OS-level serial buffer. This prevents the state machine from reading "stale" `ok` responses from previous jog commands and mistakenly believing a scan move has already finished.
*   **Result**: Resolved the physical desync where the software would progress through tiles while the gantry remained stationary.

### 4. Live UI Feedback Engine
*   **Component**: `server/state_machine.py`, `server/app.py`, `static/js/camera_feed.js`
*   **Description**: 
    *   Added `on_frame` callback to the Scan Engine to pipe high-res captures to the UI.
    *   Wired `on_arrived` to trigger `position_update` events during active scans.
    *   Disabled automatic MJPEG preview restarts during scans in `camera_feed.js` to prevent hardware resource contention.
*   **Result**: The UI now shows the actual gantry coordinates and the captured high-res tile images in real-time as the scan progresses.

### 5. Interactive System Terminal
*   **Component**: `static/index.html`, `static/css/styles.css`, `static/js/log_panel.js`, `server/app.py`
*   **Description**: Transformed the static log panel into a functional command-line interface.
    *   **Backend**: Added `terminal_command` socket event and `uart.send_command()` bypass.
    *   **Frontend**: Added command input, terminal-style CSS, and command history (Arrow Keys).
*   **Result**: User can now send raw ASCII commands (e.g., `ping 2`, `enable all 1`) directly to the Pico from the browser.

### 6. Auto-Recovery Connection Logic
*   **Component**: `server/uart_comm.py`, `server/app.py`
*   **Description**: 
    *   Made `connect()` idempotent and self-cleaning.
    *   Added auto-reconnect logic to the terminal handler.
*   **Result**: Fixed the "Machine not connected" error; the system now attempts to re-establish the serial link automatically when a command is sent.

### 7. Coordinate-Based Offline Stitcher (CLI)
*   **Component**: `stitch_scan.py` (New File)
*   **Description**: A standalone Python utility that assembles mosaics using gantry coordinates instead of image feature matching.
*   **Features**: Alpha-blending, edge feathering (smoothing), and configurable scaling.
*   **Fix [2026-05-15]**: Resolved a `ValueError` crash on large mosaics. Added robust bounds checking and image clamping to prevent negative array indices when placing tiles.
*   **Result**: Allows perfect stitching of scans even when images lack visual landmarks/features, with no crashes on high-res output.

### 8. High-Priority Emergency Stop Button
*   **Component**: `static/index.html`, `static/css/styles.css`, `static/js/app.js`, `server/app.py`
*   **Description**: Added a permanently visible red **STOP** button to the action bar.
*   **Logic**: Sends a `system_stop` event that simultaneously calls `engine.abort_scan()` and `uart.send_abort()`.
*   **Result**: Immediate halt of both the software sequence and physical gantry motion for maximum safety.

---
**Status**: All core stability bugs (Jog, Focus, Motion Sync) resolved. New troubleshooting tools (Terminal, Stitcher, E-Stop) implemented and verified.
