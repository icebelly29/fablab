# Mini-Plotter Host Firmware (ESP32)

This project turns an ESP32 into a Wi-Fi Host for the Mini-Plotter. It serves a web-based control interface (stored in `data/`) and relays G-code commands to the plotter via Serial (UART).

## Features
- **Wi-Fi Access Point:** Creates a network named "Cutter-Link".
- **Web Interface:** Full G-code sender, SVG converter, and visualizer running in the browser.
- **WebSocket Control:** Real-time bi-directional communication.
- **File System (LittleFS):** Stores the web app directly on the ESP32's flash memory.

## Prerequisites

1.  **Hardware:**
    *   ESP32 Development Board (e.g., ESP32 DevKit V1).
    *   USB Cable (Micro-USB or USB-C, depending on board).
    *   Computer with VS Code installed.

2.  **Software:**
    *   **VS Code** with **PlatformIO IDE Extension**.
    *   *Optional:* **PlatformIO Core (CLI)** if using the terminal directly.

## Setup & Installation (VS Code GUI)

1.  **Open the Project:**
    *   Open VS Code.
    *   Go to `File` > `Open Folder` and select the `esp32_host_firmware` folder.
    *   Wait for PlatformIO to initialize.

2.  **Connect ESP32:**
    *   Plug your ESP32 into the computer via USB.

3.  **Upload Firmware (The C++ Code):**
    *   Click the **PlatformIO Alien Icon** in the left sidebar.
    *   Under `Project Tasks` > `esp32dev` > `General`, click **Upload**.
    *   *Note: If it fails with "Connecting...", hold the **BOOT** button on the ESP32 until the upload starts.*

4.  **Upload Filesystem (The Web App):**
    *   This uploads the contents of the `data/` folder (HTML, JS, CSS) to the ESP32.
    *   In the PlatformIO sidebar, go to `Project Tasks` > `esp32dev` > `Platform`.
    *   Click **Upload Filesystem Image**.
    *   *Note: Again, hold the **BOOT** button if it gets stuck at "Connecting...".*

## CLI Commands (Alternative)

If you prefer using the terminal (Command Line):

1.  **Open Terminal:** Ensure you are inside the `esp32_host_firmware` directory.

2.  **Upload Firmware:**
    ```bash
    pio run --target upload
    ```
    *(Hold BOOT button if you see "Connecting..." and it hangs)*

3.  **Upload Filesystem:**
    ```bash
    pio run --target uploadfs
    ```

4.  **Monitor Serial Output:**
    ```bash
    pio device monitor
    ```

## Usage

1.  **Connect to Wi-Fi:**
    *   On your phone or laptop, look for the Wi-Fi network **"Cutter-Link"**.
    *   Password (if requried)-: **Qatar2026**
    *   *Note: These credentials are set in `esp32_host_firmware.ino`.*

2.  **Open the Dashboard:**
    *   A "Sign in to network" popup might appear (Captive Portal). Click it to open the app.
    *   If not, open a browser and go to `http://192.168.4.1`.

3.  **Start Cutting:**
    *   **Connect:** Click the red "Disconnected" badge in the top right.
    *   **Load File:** Drag & drop a G-code or SVG file.
    *   **Run:** Click "Start Cutting".

## Troubleshooting

*   **"Wrong boot mode detected"**: This means the ESP32 didn't enter download mode.
    *   **Solution:** Hold the `BOOT` button on the board when the terminal says "Connecting...". Release it once the progress bar appears.
*   **"LittleFS Mount Failed"**: The filesystem is corrupt or empty.
    *   **Solution:** Run the **Upload Filesystem Image** task again.