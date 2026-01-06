# ESP32 UI Deployment Guide

This guide explains how to update the React-based user interface for the **Cutter (WiFi)** captive portal.

## Prerequisites

1.  **Node.js**: Installed on your system.
2.  **Arduino IDE**: With ESP32 board support installed.
3.  **ESP32 Tools**: Specifically `mklittlefs` and `esptool` (usually handled by Arduino IDE).

## Quick Summary

The deployment process consists of three main steps:
1.  **Build**: Compile the React app.
2.  **Prepare**: Move the compiled files to the firmware's `data` folder.
3.  **Upload**: Flash the filesystem to the ESP32.

---

## Detailed Steps

### 1. Build the Frontend

Open a terminal in `parser/g-code-sender` and run:

```powershell
npm run build:esp32
```

This creates a `dist-esp32` folder containing the optimized HTML, CSS, and JS files.

### 2. Prepare the Firmware Data

1.  Navigate to `parser/esp32_host_firmware`.
2.  Delete the contents of the `data` folder.
3.  Copy everything from `parser/g-code-sender/dist-esp32` into `data`.
4.  **Important**: Rename `index_esp32.html` to `index.html`.

### 3. Upload to ESP32

#### Option A: Using Arduino IDE (Recommended for simple updates)
1.  Open `esp32_host_firmware.ino` in Arduino IDE.
2.  Ensure your board is selected and connected.
3.  Go to **Tools > ESP32 Sketch Data Upload**.
    *   *Note: If you don't see this option, you may need to install the "Arduino ESP32 Filesystem Uploader" plugin.*

#### Option B: Using Command Line (PowerShell)
If you don't have the plugin, use this PowerShell script (update paths as needed):

```powershell
# Paths (Update these to match your system)
$esptool = "$env:LOCALAPPDATA\Arduino15\packages\esp32\tools\esptool_py\5.1.0\esptool.exe"
$mklittlefs = "$env:LOCALAPPDATA\Arduino15\packages\esp32\tools\mklittlefs\4.0.2-db0513a\mklittlefs.exe"
$dataDir = ".\data"
$outputBin = ".\web_site.bin"
$port = "COM15" # Change to your COM port

# 1. Create Image
& $mklittlefs -c $dataDir -p 256 -b 4096 -s 0x160000 $outputBin

# 2. Upload (Hold BOOT button if needed)
& $esptool --chip esp32 --port $port --baud 921600 --before default-reset --after hard-reset write-flash -z --flash-mode dio --flash-freq 80m --flash-size 4MB 0x290000 $outputBin
```

---

## Troubleshooting

*   **"Port Busy"**: Close Arduino Serial Monitor or any other terminal using the port.
*   **"Timed Out" / "Failed to connect"**: Hold the **BOOT** button on the ESP32 while the "Connecting..." text appears in the console.
*   **White Screen / 404**: Ensure you renamed `index_esp32.html` to `index.html`.
