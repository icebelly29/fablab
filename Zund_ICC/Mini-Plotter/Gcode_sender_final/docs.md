# Mini-Plotter ESP32 Host Documentation

## 1. Project Overview
This project turns an **ESP32** into a standalone wireless host for your Mini-Plotter. It broadcasts a WiFi Access Point named **"Cutter-Link"** and serves a modern web dashboard stored on its internal filesystem (LittleFS). Users can connect to this dashboard to convert SVG files and stream G-code commands to the machine wirelessly.

---

## 2. Quick Start Guide
1.  **Power On** the ESP32.
2.  **Connect** your PC or Phone to the WiFi network:
    *   **SSID:** `Cutter-Link`
    *   **Password:** *(None / Open Network)*
3.  **Open Browser** and navigate to `http://192.168.4.1` (or wait for the "Sign in to network" prompt).
4.  **Load Files** (G-code or SVG) via the "Load File" button or Drag & Drop.
5.  **Click "Start Cutting"**.

---

## 3. Web Interface Features
*   **Console:**
    *   View real-time logs and G-code transactions.
    *   **Clear Console:** Click the "Clear" button in the header or type `clear` / `/clear` in the input box to wipe the log.
    *   **Manual Input:** Type raw G-code (e.g., `G1 X10 Y10`) and hit Enter or Run.
*   **Workspace:**
    *   **G-Code Preview:** Visualizes the cut path (Blue = Cut, Gray = Move).
    *   **SVG Preview:** Shows the original vector image.
    *   **Editor:** Allows manual editing of the generated G-code before sending.

---

## 4. File Structure

### Root Directory (`Gcode_sender_final/`)
**Source Files (Frontend):** Edit these files to change the website.
*   `index.html`: Main UI structure.
*   `script.js`: Frontend logic (WebSocket, File Parsing, UI events).
*   `styles.css`: Visual styling.
*   `SvgConverter.js`: Library for converting SVG paths to G-code.

### Firmware Directory (`esp32_host_firmware/`)
**Device Code (Backend):**
*   `esp32_host_firmware.ino`: Main C++ firmware. Handles WiFi, WebServer, and Serial/WebSocket bridging.
*   `platformio.ini`: Build configuration for PlatformIO.
*   **`data/` (CRITICAL):** This folder contains the files served to the browser.
    *   **Note:** The ESP32 does *not* read the root files directly. It reads this `data/` folder.
    *   **Sync:** You must copy updated files from Root to `data/` before uploading.

---

## 5. How to Upload (Developer Guide)

You need to perform **two** upload steps: one for the code (Firmware) and one for the web files (Filesystem).

### Prerequisites
*   **VS Code** with the **PlatformIO IDE** extension installed.
*   *Alternatively:* The `pio` command-line tool.

### Step 1: Sync Web Files
If you have modified `index.html` or `script.js` in the root folder:
1.  **Copy** the modified files (`index.html`, `script.js`, `styles.css`, `SvgConverter.js`).
2.  **Paste** them into `esp32_host_firmware/data/`, replacing the existing files.

### Step 2: Upload Filesystem (LittleFS)
This uploads the contents of the `data/` folder to the ESP32's internal storage.
*   **VS Code (GUI):**
    1.  Click the **PlatformIO Alien Icon** (Left Sidebar).
    2.  Expand **Project Tasks** > **esp32dev** > **Platform**.
    3.  Click **Upload Filesystem Image**.
*   **Terminal (CLI):**
    ```bash
    cd esp32_host_firmware
    pio run --target uploadfs
    ```
    *(Note: This might require closing the Serial Monitor if it's open)*

### Step 3: Upload Firmware
This uploads the C++ logic (`.ino` file). You only need to do this if you changed the Arduino code.
*   **VS Code (GUI):**
    1.  PlatformIO Sidebar > **Project Tasks** > **esp32dev** > **General**.
    2.  Click **Upload**.
*   **Terminal (CLI):**
    ```bash
    cd esp32_host_firmware
    pio run --target upload
    ```

---

## 6. Troubleshooting

### "VFS / LittleFS Errors" (File Not Found)
*   **Symptom:** Serial Monitor shows errors about `connecttest.txt` or missing files.
*   **Cause:** This is normal "noise" from devices checking for internet connectivity (Captive Portal checks).
*   **Solution:** Ignore them. The firmware is designed to catch these and redirect them safely.

### Changes Not Showing on Website
*   **Symptom:** You edited `index.html` but the ESP32 still shows the old version.
*   **Cause:** You likely forgot to **Sync** (Step 1) or **Upload Filesystem** (Step 2).
*   **Solution:** Copy the files to `data/` again and run `Upload Filesystem Image`.

### WiFi Connection Issues
*   **Reset:** Press the `EN` (Reset) button on the ESP32.
*   **Forget Network:** Forget "Cutter-Link" on your device and reconnect to ensure you aren't using cached credentials.