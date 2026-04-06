# Mini-Plotter: Deep Dive Analysis

This document provides a highly detailed, step-by-step analysis of the data pipeline for the Mini-Plotter project, from SVG file ingestion to the final G-code output. The analysis is based on the JavaScript codebase and the observed behavior of the ESP32 firmware.

## 1. File Ingestion and Pre-processing (`FileHandler.js`)

The pipeline begins when the user uploads an SVG file.

1.  **File Read**: The `handleFile` function is triggered. It first reads the uploaded file as a raw text string using `file.text()`.
2.  **XML Parsing**: The raw string is parsed into a navigable Document Object Model (DOM) tree using `const doc = new DOMParser().parseFromString(text, 'image/svg+xml');`.
3.  **Dimension Extraction**: The code attempts to determine the "real world" size of the SVG.
    *   It queries the root `<svg>` element for `width`, `height`, and `viewBox` attributes.
    *   The `parseToMM` helper function is used to convert attribute values (e.g., `"10in"`, `"100px"`, `"20cm"`) into a normalized millimeter value. It uses standard conversion factors (e.g., `val * 25.4` for inches, `val * 0.264583` for pixels). If no unit is found, it assumes millimeters.
4.  **Scaling Calculation**:
    *   **Objective**: To fit the SVG artwork onto the physical machine bed (defined as 230x310mm) while respecting its original aspect ratio.
    *   `vbW` and `vbH` are determined from the `viewBox` attribute. If `viewBox` is absent, it falls back to the parsed `width` and `height`.
    *   An initial `scale` is computed to reconcile the unitless `viewBox` dimensions with the dimensioned `width`/`height` attributes: `scale = w_mm / vbW`.
    *   **Auto-Fit**: The code calculates the current scaled dimensions (`currentW`, `currentH`). If either exceeds the bed size (minus a 10mm margin), a `fitScale` factor (`Math.min(scaleW, scaleH)`) is calculated and multiplied into the main `scale` variable.
5.  **Offset Calculation**:
    *   The goal is to center the scaled drawing on the bed.
    *   `offsetX = (bedW - finalW) / 2`
    *   `offsetY = (bedH - finalH) / 2`
    *   **Origin Correction**: The SVG `viewBox` can have a non-zero origin (e.g., `viewBox="-10 -10 100 100"`). The code compensates for this by subtracting the scaled `viewBox` origin: `finalOffsetX = offsetX - (vbMinX * scale)`.
    *   **Y-Axis Flip Correction**: Crucially, for the Y-axis flip (where SVG's top-left origin is mapped to CNC's bottom-left), the offset calculation is inverted. Instead of shifting the origin down, it's shifted up by the full height of the SVG. This is the logic that was implemented:
        `finalOffsetY = offsetY + (vbMinY + vbH) * scale;`
6.  **Converter Instantiation**: A new `SvgConverter` object is created, passing in the calculated `scale`, `finalOffsetX`, `finalOffsetY`, and the crucial `flipY: true` flag.

## 2. G-Code Conversion (`SvgConverter.js`)

This is the core engine where the geometric processing occurs.

### Stage 2.1: Path Tokenization

1.  **Entry Point**: `convert()` receives the raw SVG string. It finds all relevant geometric elements (`path`, `rect`, `circle`, etc.).
2.  **Element Normalization**: Shapes like `<rect>` and `<circle>` are converted into an equivalent `<path>` data format. For example, a `<rect>` becomes a series of `L` (LineTo) commands.
3.  **Tokenization**: The `parsePathData` method takes the path's `d` attribute string. This string is fed into a Regular Expression:
    ```javascript
    /([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/g
    ```
    This regex precisely splits the string into an array of command letters and numeric parameters. For example, `M10-20.5` becomes `['M', '10', '-20.5']`.

### Stage 2.2: Command Parsing

The `parseTokens` method functions as a state machine to build structured command objects.

1.  It iterates through the token array.
2.  If the token is a letter, it sets the `lastCommand` state.
3.  If the token is a number, it assumes it's an argument for the `lastCommand`. This handles SVG's implicit command syntax (e.g., `L 10 10 20 20` is the same as `L 10 10 L 20 20`).
4.  A `switch` statement determines how many numeric tokens to "eat" for each command type (`M` takes 2, `C` takes 6, etc.).
5.  The output is an array of objects, e.g., `{ type: 'C', args: [10, 10, 20, 20, 30, 30] }`.

### Stage 2.3: Curve Linearization (The Core Algorithm)

This process is handled by `flattenBezier` and the recursive `subdivideBezier` method.

1.  **Curve Representation**: A `CubicBezier` object is created with 4 `Vector2` points: P0, P1, P2, P3.
2.  **The `sample(t)` Method**: This method is the mathematical heart. It implements the explicit polynomial form of the Bezier equation:
    $B(t) = (1-t)^3P_0 + 3(1-t)^2tP_1 + 3(1-t)t^2P_2 + t^3P_3$
    It calculates the exact (x, y) coordinate on the curve for a given parameter `t` (from 0.0 to 1.0).
3.  **Recursive Subdivision**:
    *   `subdivideBezier(gcode, bezier, t0, t1)` is called, initially with `t0=0` and `t1=1`.
    *   **Step A**: It finds the start and end points of the current segment: `p0 = bezier.sample(t0)` and `p1 = bezier.sample(t1)`.
    *   **Step B**: It finds the temporal midpoint: `midT = (t0 + t1) / 2`.
    *   **Step C**: It calculates two different midpoints:
        *   `pMidActual`: The true point on the curve, found by `bezier.sample(midT)`.
        *   `pMidLinear`: The point on a straight line connecting `p0` and `p1`, found by `p0.add(p1.sub(p0).mul(0.5))`.
    *   **Step D (The Check)**: It calculates the Euclidean distance between these two points: `dist = pMidActual.dist(pMidLinear)`.
    *   **Step E (Termination Condition)**:
        *   If `dist < this.tolerance` (e.g., 0.05mm), the curve segment is considered "flat enough". A `G1` G-code command is emitted to the end point `p1` using `emitLinear`, and the recursion for this branch stops.
        *   If `dist >= this.tolerance`, the curve is too bent. The function calls itself twice:
            1.  `subdivideBezier(gcode, bezier, t0, midT)`
            2.  `subdivideBezier(gcode, bezier, midT, t1)`
        This process continues until all segments are flat enough.

### Stage 2.4: Coordinate Transformation & G-Code Formatting

1.  **The `transform(p)` Method**: Every single coordinate (`Vector2` object `p`) that will be turned into G-code is first passed through this function.
    *   `const x = (p.x * this.scale) + this.offsetX;`
    *   `let y = (p.y * this.scale);`
    *   `if (this.flipY) { y = -y; }` (The Y-axis inversion)
    *   `y += this.offsetY;`
    *   `return { x, y };`
2.  **G-code Generation (and Tangential Knife Logic)**:
    *   `emitLinear` does more than just format strings; it calculates the tool's rotational angle (`A` axis) required to perform the cut:
        *   It computes the raw angle using `Math.atan2(dy, dx) * 180 / Math.PI`.
        *   The angle is then mathematically wrapped to strictly fall between `0` and `360` degrees (`((angle % 360) + 360) % 360`).
        *   It calculates the shortest rotational distance (difference) to the new angle.
        *   If the difference exceeds the `angleThreshold` (indicating a sharp corner), `emitLinear` prepends commands to: Lift the knife (`G0 Z... A...`), orient the knife to the new targeted angle, and plunge it into the material (`G1 Z... A... F...`).
    *   Finally, it formats the main cut move into a `G1` command string: `G1 X${x.toFixed(decimals)} Y${y.toFixed(decimals)} Z... A${targetA.toFixed(2)} F${this.feedRate}`.
    *   The main `generateGcode` loop handles `M` (Move) commands by calling `transform` to move the knife while keeping it lifted.

## 3. Network Communication (`Connection.js`)

1.  **Connection**: The UI initiates a WebSocket connection to the ESP32's IP address on port 81 (`ws://<IP>:81`).
2.  **Sending Data**: When a G-code command is ready to be sent, the `send(cmd)` method is called.
    *   It wraps the command in a JSON structure: `{"type":"gcode","data":"G1 X10 Y10"}`.
    *   This JSON string is sent over the WebSocket.
3.  **Flow Control**: The system uses an acknowledgement (`ack`) mechanism.
    *   After the ESP32 receives and processes a command, it sends back a message: `{"type":"ack"}`.
    *   The `handleMessage` function in `Connection.js` receives this. If `msg.type === 'ack'`, it triggers the `onAck` callback.
    *   The main job processing script (`script.js`) uses this `onAck` signal to send the next line of G-code from its queue, preventing the ESP32's buffer from overflowing.

## 4. Firmware (`esp32_host_firmware.ino`)

The firmware's role is simple but critical: it is a **WiFi-to-Serial Bridge**.

1.  **WebSocket Server**: It runs a `WebSocketsServer` on port 81.
2.  **Event Handler**: The `onWebSocketEvent` function is the entry point for incoming data.
3.  **Message Parsing**:
    *   When an event of type `WStype_TEXT` occurs, the `uint8_t* payload` is converted into an Arduino `String`.
    *   It performs simple string manipulation to check for and extract the G-code data:
        *   `msg.indexOf("\"type\":\"gcode\"")`
        *   `dataStart = msg.indexOf("\"data\":\"")`
        *   `gcode = msg.substring(dataStart, dataEnd)`
    *   This is a lightweight manual JSON parsing method; it does not use a full JSON library.
4.  **Serial Transmission**: The extracted `gcode` string is immediately written to the hardware serial port via `Serial.println(gcode)`. This sends the command to the external CNC controller (e.g., a GRBL board) connected to the ESP32's TX/RX pins.
5.  **Acknowledgement**: Immediately after sending the data to the serial port, it sends the `ack` message back to the client to signal it's ready for the next command: `webSocket.sendTXT(num, "{\"type\":\"ack\"}");`.
