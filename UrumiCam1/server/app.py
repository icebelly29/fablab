"""
============================================================================
                    URUMICAM — FLASK APPLICATION
============================================================================

Flask + Flask-SocketIO server. Serves the web UI and provides real-time
WebSocket communication for scan control and monitoring.

Usage:
    python app.py                  # Production (Pi 4 hardware)
    python app.py --mock           # Development (mock hardware)
    python app.py --mock --port 5000

============================================================================
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import threading
from pathlib import Path

from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import Config
from server.uart_comm import UARTComm
from server.gpio_gate import GPIOGate
from server.camera import CameraController
from server.roi_detector import ROIDetector
from server.tile_planner import TilePlanner
from server.quality import FocusChecker
from server.stitcher import MosaicStitcher
from server.scan_io import ScanIO
from server.calibration import CalibrationManager
from server.state_machine import ScanEngine

# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urumicam.app")

# ── Argument Parsing ───────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="UrumiCam Scanner Server")
parser.add_argument("--mock", action="store_true", help="Run with mock hardware")
parser.add_argument("--port", type=int, default=5000, help="Server port")
parser.add_argument("--host", default="0.0.0.0", help="Server host")
args = parser.parse_args()

MOCK_MODE = args.mock

# ── Flask App ──────────────────────────────────────────────────────────────

static_dir = str(Path(__file__).resolve().parent.parent / "static")
app = Flask(__name__, static_folder=static_dir, static_url_path="")
app.config["SECRET_KEY"] = "urumicam-scanner-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Initialize Subsystems ─────────────────────────────────────────────────

config = Config()
uart = UARTComm(
    port=config.get("uart_port", "/dev/ttyACM0"),
    baudrate=config.get("uart_baudrate", 115200),
    mock=MOCK_MODE,
)
uart.configure(
    motor_x_addr=config.get("motor_x_addr", 3),
    motor_y_addr=config.get("motor_y_addr", 2),
    steps_per_mm_x=config.get("steps_per_mm_x", 160.0),
    steps_per_mm_y=config.get("steps_per_mm_y", 160.0),
)
gpio = GPIOGate(
    pin=config.get("gpio_quiescence_pin", 17),
    mock=MOCK_MODE,
    dwell_s=config.get("quiescence_dwell_s", 0.4),
)
camera = CameraController(config=config, mock=MOCK_MODE)
roi_detector = ROIDetector(config)
tile_planner = TilePlanner(config)
focus_checker = FocusChecker(config)
stitcher = MosaicStitcher(config)
scan_io = ScanIO(base_dir=config.get("scan_output_dir", "scans"))
calibration = CalibrationManager(config, camera, uart, gpio, focus_checker)

engine = ScanEngine(
    config, uart, gpio, camera,
    roi_detector, tile_planner, focus_checker,
    stitcher, scan_io
)

# ── Wire Engine Events to WebSocket ───────────────────────────────────────

def emit_state_change(new_state, old_state):
    socketio.emit("state_change", {
        "state": new_state,
        "previous": old_state,
        "timestamp": time.strftime("%H:%M:%S"),
    })

def emit_tile_update(tile_data):
    socketio.emit("tile_update", tile_data)

def emit_log(message, level="info"):
    socketio.emit("log_message", {
        "message": message,
        "level": level,
        "timestamp": time.strftime("%H:%M:%S"),
    })

def emit_progress(completed, total):
    socketio.emit("scan_progress", {
        "completed": completed,
        "total": total,
    })

def emit_scan_complete(data):
    socketio.emit("scan_complete", data)

def emit_roi_detected(data):
    # Convert contours to serializable format
    roi_payload = {
        "success": data["success"],
        "rois": data["rois"],
        "rois_px": data["rois_px"],
        "method": data["method"],
    }
    socketio.emit("roi_overlay", roi_payload)

def emit_error(error_type, detail):
    socketio.emit("error", {"type": error_type, "detail": detail})

def emit_captured_frame(frame):
    import cv2
    import base64
    # Downscale for UI performance
    h, w = frame.shape[:2]
    preview = cv2.resize(frame, (640, int(640 * h / w)))
    _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf).decode("ascii")
    socketio.emit("camera_frame", {"data": b64})

def emit_position_update(x, y):
    # Convert absolute steps back to mm for the UI
    x_mm = x / config.get("steps_per_mm_x", 160.0)
    y_mm = y / config.get("steps_per_mm_y", 160.0)
    socketio.emit("position_update", {"x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3)})

engine.on_state_change = emit_state_change
engine.on_tile_update = emit_tile_update
engine.on_log = emit_log
engine.on_progress = emit_progress
engine.on_scan_complete = emit_scan_complete
engine.on_roi_detected = emit_roi_detected
engine.on_error = emit_error
engine.on_frame = emit_captured_frame

uart.on_arrived = lambda x, y: [emit_position_update(x, y), engine._on_arrived(x, y)]

# ── HTTP Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(static_dir, path)

@app.route("/api/status")
def api_status():
    status = engine.get_status()
    # Add UART diagnostics
    status['uart'] = {
        'connected': uart.is_connected,
        'port': uart.port,
        'baudrate': uart.baudrate,
        'recent_messages': uart.get_log(5)
    }
    return jsonify(status)

@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(config.to_dict())

@app.route("/api/config", methods=["POST"])
def api_set_config():
    updates = request.json
    config.update(updates)
    return jsonify({"status": "ok"})

# ── WebSocket Events ───────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("[WS] Client connected")
    emit("state_change", {
        "state": engine.state.value,
        "previous": None,
        "timestamp": time.strftime("%H:%M:%S"),
    })
    emit("log_message", {
        "message": f"Connected to UrumiCam {'(MOCK)' if MOCK_MODE else ''}",
        "level": "success",
        "timestamp": time.strftime("%H:%M:%S"),
    })

@socketio.on("scan_start")
def on_scan_start(data):
    folder_name = data.get("folder_name", time.strftime("scan_%Y%m%d_%H%M%S"))
    roi_box = data.get("roi_box")
    logger.info(f"[WS] Scan start: {folder_name} with ROI {roi_box}")
    camera.stop_preview()
    engine.start_scan(folder_name, roi_box)

@socketio.on("scan_abort")
def on_scan_abort(data=None):
    logger.info("[WS] Scan abort")
    engine.abort_scan()

@socketio.on("system_stop")
def on_system_stop(data=None):
    """Emergency stop: halts scan engine and sends stop command to hardware."""
    logger.warning("[WS] EMERGENCY STOP REQUESTED")
    engine.abort_scan()
    uart.send_abort()
    emit_log("EMERGENCY STOP EXECUTED", "error")

@socketio.on("roi_confirm")
def on_roi_confirm(data):
    logger.info("[WS] ROI confirmed")
    engine.confirm_roi(data)

@socketio.on("roi_rescan")
def on_roi_rescan(data=None):
    logger.info("[WS] ROI rescan requested")
    engine.start_scan(engine.scan_dir.name if engine.scan_dir else "rescan")

@socketio.on("retry_failed")
def on_retry_failed(data=None):
    logger.info("[WS] Retry failed tiles")
    engine.retry_failed_tiles()

@socketio.on("scan_reset")
def on_scan_reset(data=None):
    logger.info("[WS] Scan reset")
    engine.reset()

@socketio.on("jog")
def on_jog(data):
    """Jog the gantry by a step in a given direction."""
    if not uart.is_connected:
        emit_log("Not connected to machine", "error")
        return

    direction = data.get("dir", "")
    step_mm   = float(data.get("step_mm", 1.0))
    sps       = int(config.get("move_sps", 800))
    spmx      = config.get("steps_per_mm_x", 160.0)
    spmy      = config.get("steps_per_mm_y", 160.0)

    dx_steps = 0
    dy_steps = 0
    if   direction == "x+": dx_steps =  int(step_mm * spmx)
    elif direction == "x-": dx_steps = -int(step_mm * spmx)
    elif direction == "y+": dy_steps =  int(step_mm * spmy)
    elif direction == "y-": dy_steps = -int(step_mm * spmy)
    else:
        emit_log(f"Unknown jog direction: {direction}", "error")
        return

    logger.info(f"[WS] Jog {direction} {step_mm}mm ({dx_steps},{dy_steps} steps)")
    emit_log(f"Jogging {direction} ({step_mm} mm)...")

    def _on_jog_done(pos_mm):
        logger.info(f"[WS] Jog complete: position now {pos_mm['x_mm']:.2f}, {pos_mm['y_mm']:.2f}")
        emit_log(f"Jog complete: ({pos_mm['x_mm']:.2f}, {pos_mm['y_mm']:.2f}) mm", "success")
        socketio.emit("position_update", pos_mm)

    def _on_jog_failed():
        logger.error("[WS] Jog failed — Pico not responding")
        emit_log("Jog failed — Pico not responding", "error")
        # Emit current position to unlock the UI jogBusy flag
        socketio.emit("position_update", uart.get_position_mm())

    uart.send_jog(dx_steps, dy_steps, sps=sps, on_complete=_on_jog_done, on_failed=_on_jog_failed)

@socketio.on("terminal_command")
def on_terminal_command(data):
    cmd = data.get("command", "").strip()
    if not cmd:
        return

    # Special handling for 'connect' command
    if cmd.lower() == "connect":
        if uart.is_connected:
            emit_log("Already connected to machine", "info")
        else:
            emit_log("Attempting to connect to machine...", "info")
            if uart.connect():
                emit_log("Connected to machine!", "success")
            else:
                emit_log("Connection failed. Check port and permissions.", "error")
        return

    # Auto-connect attempt if disconnected
    if not uart.is_connected:
        logger.info("[WS] Terminal command received while disconnected — attempting auto-connect")
        if not uart.connect():
            emit_log("Not connected to machine (and auto-connect failed)", "error")
            return
        emit_log("Auto-connected to machine", "success")

    uart.send_command(cmd)

@socketio.on("reset_position")
def on_reset_position(data=None):
    """Reset the tracked position to (0, 0) without moving."""
    logger.info("[WS] Position reset to (0,0)")
    uart.reset_position()
    socketio.emit("position_update", {"x_mm": 0.0, "y_mm": 0.0})

@socketio.on("config_update")
def on_config_update(data):
    logger.info(f"[WS] Config update: {list(data.keys())}")
    config.update(data)
    emit("log_message", {
        "message": "Configuration updated",
        "level": "success",
        "timestamp": time.strftime("%H:%M:%S"),
    })

@socketio.on("calibrate")
def on_calibrate(data):
    cal_type = data.get("type", "")
    logger.info(f"[WS] Calibration: {cal_type}")

    def progress(msg):
        emit_log(f"[CAL] {msg}", "info")

    if cal_type == "pixels_per_step":
        result = calibration.calibrate_pixels_per_step(
            step_count=data.get("step_count", 1000),
            callback=progress
        )
    elif cal_type == "tile_fov":
        result = calibration.calibrate_tile_fov(
            known_width_mm=data.get("width_mm", 10),
            known_height_mm=data.get("height_mm", 7.5),
            callback=progress
        )
    elif cal_type == "focus_baseline":
        result = calibration.calibrate_focus_baseline(callback=progress)
    elif cal_type == "quiescence":
        result = calibration.calibrate_quiescence(callback=progress)
    else:
        result = {"error": f"Unknown calibration type: {cal_type}"}

    emit("calibration_result", result)

# ── Camera Preview Stream ─────────────────────────────────────────────────

@socketio.on("start_preview")
def on_start_preview(data=None):
    camera.start_preview()
    # Start emitting frames
    def stream_frames():
        while camera._preview_running:
            frame = camera.get_preview_frame()
            if frame:
                b64 = base64.b64encode(frame).decode("ascii")
                socketio.emit("camera_frame", {"data": b64})
            socketio.sleep(0.066)  # ~15 FPS
    socketio.start_background_task(stream_frames)

@socketio.on("stop_preview")
def on_stop_preview(data=None):
    camera.stop_preview()

# ── Mock Mode Helpers ──────────────────────────────────────────────────────

if MOCK_MODE:
    def mock_scan_simulation():
        """Simulate Pico 2 responses in mock mode."""
        original_send_move = uart.send_move_to

        def mock_send_move(x, y):
            original_send_move(x, y)
            # Simulate arrival after a delay
            def delayed_arrival():
                time.sleep(0.3)  # Simulate motion time
                uart.mock_inject(f"ACK_ARRIVED {int(x)} {int(y)}")
                time.sleep(0.1)
                gpio.mock_set_quiescent(True)
                time.sleep(0.5)
                gpio.mock_set_quiescent(False)
            threading.Thread(target=delayed_arrival, daemon=True).start()

        uart.send_move_to = mock_send_move

    mock_scan_simulation()

# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "MOCK" if MOCK_MODE else "PRODUCTION"
    logger.info(f"╔════════════════════════════════════════╗")
    logger.info(f"║   UrumiCam Scanner Server ({mode:10s}) ║")
    logger.info(f"╠════════════════════════════════════════╣")
    logger.info(f"║   Port: {args.port:<31}║")
    logger.info(f"║   Host: {args.host:<31}║")
    logger.info(f"╚════════════════════════════════════════╝")

    # Connect UART
    uart.connect()

    # Start Flask-SocketIO
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
