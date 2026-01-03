# Zund ICC G-Code Parser and Sender

This directory contains the components needed to send G-code commands from a web-based user interface to an ESP32 microcontroller, which can then be used to control a machine like the Zund cutter.

The system is composed of three main parts:
1.  **ESP32 G-code Receiver**: An Arduino sketch that runs on an ESP32 to receive and acknowledge G-code commands.
2.  **WebSocket-to-Serial Bridge**: A Node.js script that acts as a middleman, relaying messages between the web UI and the ESP32's serial port.
3.  **G-code Sender Frontend**: A React-based web application that provides the user interface for loading, editing, converting, and sending G-code.

---

## How it Works

The communication flow is designed for reliable, line-by-line G-code execution:

1.  **Connection**: The user opens the web UI and clicks "Connect
2.  ." This establishes a WebSocket connection to the WebSocket-to-Serial Bridge.
3.  **G-code Sending**: The user clicks "Send G-code".
4.  **UI to Bridge**: The React frontend sends a s
5.  
6.  
7.  
8.  ingle line of G-code to the bridge via WebSocket.
9.  **Bridge to ESP32**: The Node.js bridge receives the WebSocket message and writes the G-cod
10. e line to the connected serial port.
11. **ESP32 Execution**: The ESP32 receives the command, processes it (in this case, just prints it), and sends back an `ok` string over serial to acknowledge completion.
12. **ESP32 to Bridge**: The bridge reads the `ok` acknowledgement from the serial port.
13. **Bridge to UI**: The bridge sends an `ack` message back to the React frontend via WebSocket.
14. **Loop**: The frontend, upon receiving the `ack`, knows it's safe to send the next line of G-code. This process repeats until all lines have been sent and acknowledged.

The frontend also supports loading SVG files, which are sent to the bridge for conversion into G-code and then loaded into the UI.

---

## Components

### 1. ESP32 G-code Receiver (`esp32_gcode_receiver.ino`)

A simple Arduino sketch for an ESP32 board.
- **Functionality**: Listens for incoming data on the serial port (`115200` baud rate). It reads each line, prints it to its own serial monitor for debugging, and sends back an `ok\n` acknowledgement to signal that it's ready for the next command.
- **Purpose**: To act as the machine-level controller that receives instructions.

### 2. WebSocket-to-Serial Bridge (`g-code-sender/websocket-bridge.cjs`)

A Node.js script that links the web frontend with the physical hardware.
- **Functionality**:
    - Starts a WebSocket server on `ws://localhost:8080`.
    - Connects to the ESP32 via a specified serial port (hardcoded as `COM8`).
    - Relays `gcode` messages from the frontend to the ESP32's serial port.
    - Listens for data from the ESP32 and relays it back to the frontend. An `ok` from the ESP32 is translated into an `ack` message for the UI.
    - Handles SVG-to-G-code conversion using the `svg-to-gcode` library.
- **Purpose**: To bridge the gap between web technologies (WebSockets) and hardware communication (USB/Serial).

### 3. G-code Sender Frontend (`g-code-sender/`)

A React application built with Vite.
- **Functionality**:
    - Provides a text area to view, edit, or paste G-code.
    - Allows users to load G-code from local files (`.gcode`, `.nc`, etc.).
    - Allows users to load an SVG file, which is then converted to G-code by the bridge.
    - Manages the line-by-line sending protocol with the bridge.
    - Displays the connection status and messages from the ESP32.
- **Purpose**: To provide a user-friendly interface for controlling the G-code sending process.

---

## Prerequisites

- **Hardware**: An ESP32 board.
- **Software**:
    - [Node.js and npm](https://nodejs.org/en/): Required to run the bridge and the frontend development server.
    - [Arduino IDE](https://www.arduino.cc/en/software) or [Arduino CLI](https://arduino.github.io/arduino-cli/): To compile and upload the sketch to the ESP32.
    - A modern web browser.

---

## How to Run

### Step 1: Flash the ESP32

1.  Open `parser/esp32_gcode_receiver.ino` in the Arduino IDE.
2.  Connect your ESP32 to your computer.
3.  In the IDE, select your ESP32 board from the `Tools > Board` menu.
4.  Select the correct COM port from `Tools > Port`.
5.  Click the "Upload" button to flash the sketch to the board.
    - You can open the Serial Monitor (set to 115200 baud) to see the "ESP32 G-code Receiver Ready" message. **Important**: Close the Serial Monitor before proceeding to the next step, as it will lock the serial port.

### Step 2: Set up the Backend Bridge and Frontend

1.  Open a terminal and navigate to the frontend directory:
    ```sh
    cd parser/g-code-sender
    ```
2.  Install the necessary Node.js dependencies:
    ```sh
    npm install
    ```

### Step 3: Run the System

You will need to run the bridge and the frontend in **two separate terminals**.

#### **Terminal 1: Start the WebSocket-to-Serial Bridge**

1.  **IMPORTANT**: Before starting, check the `websocket-bridge.cjs` file. The serial port is hardcoded.
    ```javascript
    const SERIAL_PORT_PATH = 'COM8'; // <-- CHANGE THIS to your ESP32's port if different
    ```
    Update this value to match the port your ESP32 is connected to.
2.  In the `parser/g-code-sender` directory, run the bridge:
    ```sh
    node websocket-bridge.cjs
    ```
    You should see messages indicating that the WebSocket server has started and it's trying to open the serial port.

#### **Terminal 2: Start the Frontend Application**

1.  In the `parser/g-code-sender` directory, run the Vite development server:
    ```sh
    npm run dev
    ```
2.  The terminal will output a local URL (e.g., `http://localhost:5173`). Open this URL in your web browser.

### Step 4: Send G-Code

1.  In the web UI, click the **"Connect to Bridge"** button. The status should change to "Connected".
2.  Use the default G-code in the text area or load your own file.
3.  Click the **"Send G-Code"** button.
4.  You will see the status update as each line is sent and acknowledged. You can also monitor the output in both the bridge and ESP32 serial monitor (if you have a separate way of viewing it) to see the communication in real-time.
