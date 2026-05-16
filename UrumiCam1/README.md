# UrumiCam — Macro Gantry Scanner

A production-grade stop-and-shoot raster scanning system for gantry-based cutting platforms. Captures high-resolution tile images and stitches them into a full-bed mosaic.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 4                      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │ Flask +   │  │ OpenCV   │  │ Web UI            │ │
│  │ SocketIO  │  │ Vision   │  │ (Browser)         │ │
│  │ Server    │  │ Pipeline │  │                   │ │
│  └────┬──────┘  └────┬─────┘  └───────────────────┘ │
│       │              │                               │
│  ┌────┴──────────────┴──────┐                       │
│  │    State Machine Engine   │                       │
│  └──────────┬───────────────┘                       │
│             │ USB Serial (/dev/ttyACM0)              │
└─────────────┼──────────────────────────────────────┘
              │ ASCII text commands
              │ ("move", "stop", "enable", "ping")
┌─────────────┼──────────────────────────────────────┐
│  ┌──────────┴───────────┐                          │
│  │  Raspberry Pi Pico 2  │  (Urumi-Fw)             │
│  │  Core 0: USB parser   │                          │
│  │  Core 1: RS485 master │                          │
│  └──────────┬────────────┘                          │
│             │ RS485 (115200 baud)                    │
│    ┌────────┴────────┐                              │
│    │ ATtiny3224 nodes │  (one per stepper axis)     │
│    │ Node X, Node Y   │                              │
│    └─────────────────┘                              │
└─────────────────────────────────────────────────────┘
```

> **Note:** The Pico 2 runs the existing **Urumi-Fw** firmware unmodified.
> UrumiCam talks to it via USB serial using the native ASCII command protocol.

## Quick Start

### Development (Windows — Mock Mode)

```bash
cd UrumiCam
pip install -r requirements.txt
python server/app.py --mock --port 5000
```

Open `http://localhost:5000` in your browser.

### Production (Raspberry Pi 4)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3-venv rpicam-apps

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Flash the Pico 2 with Urumi-Fw (PlatformIO)
#    See: https://github.com/your-org/Urumi-Fw
#    Connect Pico via USB, then:
#    pio run -e RPiPico2 --target upload

# 3. Run server (Pico must be connected via USB before starting)
# Note: Ensure venv is active (source venv/bin/activate) before running
python server/app.py
```

## Project Structure

```
UrumiCam/
├── server/                 # Python backend (Pi 4)
│   ├── app.py              # Flask + SocketIO server
│   ├── state_machine.py    # Core scan state machine
│   ├── uart_comm.py        # USB serial → Pico 2 communication
│   ├── gpio_gate.py        # Software quiescence gate
│   ├── camera.py           # rpicam-based capture
│   ├── roi_detector.py     # Contrast ROI detection
│   ├── tile_planner.py     # Tile grid computation
│   ├── quality.py          # Focus quality (Laplacian variance)
│   ├── stitcher.py         # Mosaic stitching (ORB + fallback)
│   ├── scan_io.py          # File/folder management
│   ├── calibration.py      # Calibration routines
│   └── config.py           # Configuration management
│
├── static/                 # Web UI
│   ├── index.html          # Two-panel layout
│   ├── css/styles.css      # Premium dark theme
│   └── js/                 # UI modules
│       ├── app.js           # Main controller
│       ├── websocket.js     # Socket.IO client
│       ├── tile_grid.js     # Canvas tile renderer
│       ├── camera_feed.js   # MJPEG camera display
│       ├── state_display.js # State machine display
│       ├── log_panel.js     # Scrolling log
│       └── calibration_ui.js # Settings panel
│
├── config.json             # Persistent configuration
├── requirements.txt        # Python dependencies
└── scans/                  # Scan output (runtime)
```

## Communication Protocol

### Pi 4 → Pico 2 (USB Serial, 115200 baud, ASCII)

The Pico 2 runs **Urumi-Fw** and accepts plain-text commands via its USB serial port (`/dev/ttyACM0`):

| Command | Format | Description |
|---------|--------|-------------|
| `move` | `move <count> <addr...> <steps...> <sps...>` | Queue synchronized multi-axis move |
| `stop` | `stop` | Emergency stop — clears all buffers |
| `enable` | `enable <addr\|all> <0\|1>` | Enable/disable motor driver |
| `ping` | `ping <addr>` | Check if RS485 node is alive |

**Responses from Pico:**

| Response | Meaning |
|----------|---------|
| `ok` | Move segment queued successfully |
| `nope` | Ring buffer full — retry |
| `ready` | Buffer drained past low watermark — safe to resume |
| `Ping response: OK` | Node is alive |
| `Ping response: TIMEOUT` | Node not responding |

**Move command examples:**
```
# Move X axis (node 3) forward 1600 steps at 800 sps
move 1 3 1600 800

# Move X (node 3) and Y (node 2) simultaneously
move 2 3 2 1600 800 800 800

# Emergency stop
stop

# Enable all motors on startup
enable all 1
```

### RS485 Bus (Pico 2 → ATtiny3224 motor nodes)

Binary framing — handled entirely by Urumi-Fw. UrumiCam does not speak this protocol directly.

```
Frame: [SIZE] [ADDR] [CMD] [PAYLOAD...] [CRC16 LE]
SIZE = ((bytes_after_size) << 1) | 1   ← LSB always 1
```

Node addresses: **X = node 3, Y = node 2** (configured in `config.json`)

## Scan Workflow

### ROI Definition (Manual — 2-Point Jogging)
1. Jog camera to top-left corner of workpiece using existing CNC controller
2. Enter coordinates in UrumiCam UI → **Top-Left X/Y**
3. Jog to bottom-right corner
4. Enter coordinates → **Bottom-Right X/Y**
5. Click **Start Scan**

### Scan States

`IDLE` → `PLAN` → `TARGETING` → `SETTLING` → `CAPTURING` → `PROCESSING` → `TILE_COMPLETE / TILE_FAILED` → `STITCH` → `COMPLETE`

## Configuration

Edit `config.json` or use the Settings panel in the UI. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_fov_x_mm` | 10.0 | Tile field of view width (mm) |
| `tile_fov_y_mm` | 7.5 | Tile field of view height (mm) |
| `overlap_fraction` | 0.28 | Tile overlap (28%) |
| `max_focus_retries` | 3 | Focus failure retry count |
| `steps_per_mm_x` | 160 | X axis steps per mm |
| `steps_per_mm_y` | 160 | Y axis steps per mm |
| `motor_x_addr` | 3 | RS485 node address for X motor |
| `motor_y_addr` | 2 | RS485 node address for Y motor |

## License

Private — Fablab ICC
