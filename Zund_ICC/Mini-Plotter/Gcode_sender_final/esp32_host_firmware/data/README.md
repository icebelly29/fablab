# WebSerial Trajectory Sender

This application is a browser-based frontend for controlling a CNC/Plotter machine (specifically targeting a Raspberry Pi Pico). It converts SVG vector graphics into highly detailed, segmented trajectory data (`xyz X Y Z Vx Vy Vz Angle`) and streams it over USB via the WebSerial API.

## Features

- **WebSerial Communication**: Directly talk to your Raspberry Pi Pico over USB from your browser. No extra drivers or backend servers needed.
- **Segmented Trajectory Generation**: Instead of generic `G1` commands, paths are subdivided into exact, equal-length segments (user-configurable).
- **Bezier Kinematics & Normalized Speed**: Calculates the pure directional velocity (`Vx`, `Vy`) for every single point and normalizes it to ensure the machine travels at a perfectly consistent user-defined Feedrate (Cutting Speed), regardless of curve lengths.
- **Tangential Knife Support**: Automatically calculates rotational angles for every segment and inserts Z-lift/plunge sequences for sharp corners based on an angle threshold.
- **Relative Z-Axis Control**: The trajectory engine outputs Relative Step Changes for the Z-axis (e.g. `6400` to plunge, `-6400` to lift, `0` for no change), ensuring clean delta moves for the Pico firmware.
- **Safe Retract on Restart**: If a job is interrupted, the UI remembers the machine's exact last known location. Upon restarting, it dynamically injects a pure vertical lift command to safely retract the pen *before* moving to the job's starting point, completely preventing diagonal drag crashes.
- **Auto-Retry Polling**: Fully handles Pico hardware buffer limits. If the firmware replies with `nope` (buffer full), the UI caches the exact line and automatically polls the connection with a 50ms delay until the Pico is ready (`ok`), ensuring massive files stream seamlessly without hanging.

---

## Setup & How to Run

Because this application uses the **WebSerial API**, modern browsers enforce strict security rules:
**You cannot double-click the `index.html` file to run it.** It must be served over a secure context (`https://` or `localhost`).

### Running Locally (Recommended)

You need to start a simple local web server in this folder. 

**Using Node.js:**
1. Open your terminal/command prompt.
2. Navigate to this folder (`data/`).
3. Run: `npx serve`
4. Open your web browser and go to the `localhost` URL provided in the terminal.

---

## How to Use

1. **Connect the Machine:**
   - Plug your Raspberry Pi Pico into your computer via USB.
   - Click the **Connect Serial** button in the top right.
   - A browser popup will appear. Select the COM port corresponding to your Pico and click "Connect".

2. **Machine Configuration (⚙️ Settings):**
   - Click the **Settings** button in the toolbar to open the Machine Configuration Modal.
   - Adjust the **Segment Length (mm)** to change the resolution of your trajectory points.
   - Dial in your **Cutting Speed (mm/sec)**. The UI will normalize all velocity vectors and apply this target feedrate globally.
   - Configure your stepper hardware: **Motor Steps per Rev**, **Microstepping**, and **mm per Revolution**. The UI will dynamically calculate the exact Steps/mm multiplier for the trajectory outputs.

3. **Load a File:**
   - Drag and drop an `.svg` file onto the "Trajectory Preview" window, or click **Load File**.
   - The application automatically scales the vector to fit your machine bed (230x310mm), flips the Y-axis (to match CNC standard coordinates), and converts it into pure integer CSV trajectory data prefixed with `xyz`.

4. **Review & Start Cutting:**
   - **Trajectory Preview Tab:** See exactly what the machine will draw. The visualizer parses the relative Z outputs to draw blue paths for Pen-Down moves and grey dashed paths for Pen-Up travels.
   - Once ready, click **Start**. The app streams the file line-by-line, perfectly coordinating with the Pico's `ok` and `nope` signals!

---

## The Output Format

The engine generates data mapped strictly to **Steps** and pure **Integers** (no decimals) to optimize firmware parsing. Every line follows this format:

`xyz X Y Z Vx Vy Vz Angle`

- **xyz**: The required prefix identifier.
- **X, Y**: The absolute target coordinates in Steps.
- **Z**: The *Relative* change in Z Steps (e.g. `6400` to plunge, `-6400` to lift).
- **Vx, Vy, Vz**: The normalized velocity vector in Steps/sec.
- **Angle**: The absolute Tangential Knife angle in degrees.

*(Example: `xyz 1024 1024 6400 1200 400 0 45`)*
