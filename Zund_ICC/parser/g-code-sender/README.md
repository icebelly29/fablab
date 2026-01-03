# 🚀 Zünd ICC G-Code Sender: Startup Guide

This directory contains the React frontend and the Node.js bridge for the Zünd ICC system.

## 🛠 Prerequisites

- **Node.js**: v18 or newer recommended.
- **Hardware**: ESP32 connected via USB (default port is `COM8`).
- **Dependencies**: Run `npm install` in this directory before starting.

## 🏃 How to Run

You need to run **two separate services** concurrently.

### 1. Start the WebSocket-to-Serial Bridge
This handles communication between your browser and the hardware, and performs SVG-to-G-code conversion.

```powershell
# In terminal 1
node websocket-bridge.cjs
```
> **Note**: If your ESP32 is not on `COM8`, update the `SERIAL_PORT_PATH` constant in `websocket-bridge.cjs`.

### 2. Start the Frontend (UI)
This launches the web interface where you can load SVGs and control the machine.

```powershell
# In terminal 2
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🎨 SVG to G-Code Conversion Features

The system uses a custom `SvgConverter.cjs` engine with several advanced features:

- **Biarc Approximation**: Converts complex SVG paths and curves into smooth circular arcs (`G2`/`G3` commands) instead of tiny linear segments. This results in:
    - Smaller G-code file sizes.
    - Smoother machine motion.
    - Higher precision for curved shapes.
- **Configurable Parameters**:
    - **Feed Rate**: Control cutting speed.
    - **Safe Z / Cut Z**: Define travel and cutting heights.
    - **Tool Commands**: Custom `M3`/`M5` (or others) for tool engagement.
- **Real-time Preview**: View the generated G-code path before sending it to the machine.

## 📁 Project Structure

- `websocket-bridge.cjs`: The Node.js server bridging WebSockets and Serial.
- `SvgConverter.cjs`: The core conversion logic (Node.js version).
- `src/utils/SvgConverter.js`: Browser-side version of the converter.
- `src/components/GCodePreview.jsx`: Visualizes the toolpath.
- `src/App.jsx`: Main UI application logic.