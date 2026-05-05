---
title: Software
image: https://t4.ftcdn.net/jpg/02/67/52/49/360_F_267524919_wXbVQHR189pLVU06eQ85GGLnJMq2eJFR.jpg

authors:
    -   name: Muhammed Midlaj N
        designation: Software Developer
        avatar: https://midlajn.github.io/PortFolio/images/profile.webp
        github: https://github.com/MidlajN
        linkedin: https://linkedin.com/in/johndoe

    -   name: Nikhil Nair
        designation: Software Engineer (Tech Fellow)
        avatar: https://avatars.githubusercontent.com/u/68722494?v=4
        github: https://github.com/icebelly29
        linkedin: https://www.linkedin.com/in/nikhilnair29

sections:
    -   title: "Electronics"
        link: "./electronics"
    -   title: "Mechanical"
        link: "./mechanical"
    -   title: "End effector"
        link: "./end-effector"
    -   title: "Software"
        link: "./software"
---

# Urumi Interface

Urumi Interface is a Frontend Application, developed using the Next.js framework. The application facilitates seamless communication with connected CNC (Computer Numerical Control) machines through the WebSerial API. This interface aims to offer a user-friendly experience for monitoring and controlling CNC operations directly from a web browser.

![Alt text](./media/urumi.png)

## Key Features

- **Real-time Communication**: Enables direct and efficient communication with CNC machines using the WebSerial API.
- **User-friendly Interface**: Designed with a clean and intuitive interface to ensure ease of use for operators.
- **Cross-Platform Accessibility**: Being a web application, it can be accessed from any device with a web browser, providing flexibility and convenience.
- **Next.js Framework**: Utilizes the powerful features of Next.js, including server-side rendering and static site generation, to ensure optimal performance and SEO.

## Technical Details

### Framework

The frontend application is built using the **Next.js** framework. Next.js is a popular React framework that offers several benefits, including:

- **Server-Side Rendering (SSR)**: Improves performance by rendering pages on the server.
- **Static Site Generation (SSG)**: Allows pre-rendering of pages at build time, which enhances load times and SEO.
- **API Routes**: Simplifies the creation of backend APIs within the same project.

### WebSerial API

The **WebSerial API** is a modern browser API that provides direct access to serial ports. This is crucial for enabling real-time communication between the web application and CNC machines. Key capabilities include:

- **Port Access**: Open and close connections to serial ports.
- **Data Transfer**: Send and receive data to/from the CNC machine.
- **Event Handling**: Handle serial port events to manage data streams and connection states.

### Communication Workflow

1. **Port Connection**: The user selects the appropriate serial port connected to the CNC machine.
2. **Data Exchange**: The application sends commands to and receives responses from the CNC machine via the WebSerial API.
3. **Real-time Monitoring**: The interface updates in real-time based on the data received from the CNC machine, allowing users to monitor operations seamlessly.

## Installation and Setup

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/your-repo/machine-interface.git
   ```
2. **Install Dependencies**:
   ```bash
   cd machine-interface
   npm install
   ```
3. **Run the Development Server**:
   ```bash
   npm run dev
   ```
4. **Access the Application**: Open your browser and navigate to `http://localhost:3000`.

## Usage Instructions

1. **Launch the Application**: Open the application in a web browser.
2. **Manipulate SVG**: Manipulate the Svg figure to Execute using the import or editor options
2. **Connect to CNC Machine**: Select the appropriate serial port from the list provided by the WebSerial API.
3. **Control Operations**: Use the provided interface to send commands and monitor the CNC machine's status.
4. **Disconnect**: Properly close the connection to the serial port when finished.

<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
 padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://scribehow.com/embed/Kochund_Interface__lsUAf8H9TsuqLawyu54OFg" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>

You can find a detailed User Instruction [Here](https://scribehow.com/shared/Kochund_Interface__lsUAf8H9TsuqLawyu54OFg)

## Security Considerations

- **Permissions**: The WebSerial API requires user consent to access serial ports, ensuring secure and controlled access.

## Conclusion

The Machine Interface Frontend Application provides a robust and user-friendly solution for interacting with CNC machines through a web browser. By leveraging Next.js and the WebSerial API, this application combines performance, accessibility, and real-time capabilities, making it an invaluable tool for CNC machine operators.

---

## WebSerial Trajectory Sender - Urumi V2.0

The Urumi Interface now includes a fully custom, browser-based **WebSerial Trajectory Sender** — a major evolution beyond simple G-code streaming. Instead of sending generic movement commands, the application converts SVG vector graphics into highly detailed, segmented trajectory data and streams it directly to the Raspberry Pi Pico over USB, with no backend server or extra drivers required.

## Demo Video
<video width="640" height="480" controls>
  <source src="./media/software/cutter.mp4" type="video/mp4">
</video>

### Key Capabilities

- **Segmented Trajectory Generation**: SVG paths are subdivided into exact, equal-length segments (user-configurable resolution). Each segment carries precise position and velocity data.
- **Bezier Kinematics & Normalized Speed**: The engine calculates the pure directional velocity (`Vx`, `Vy`) for every single point using Bezier curve derivatives and normalizes it, ensuring the machine travels at a perfectly consistent user-defined feedrate regardless of curve shape or length.
- **Tangential Knife Support**: Automatically calculates rotational angles for every segment and inserts Z-lift/plunge sequences for sharp corners based on a configurable angle threshold.
- **Fully Relative Coordinate System**: All axes (X, Y, Z, and Angle) output as **relative step changes (deltas)**. This ensures clean delta moves for the Pico firmware and avoids sending redundant zero-velocity data.
- **Safe Retract on Restart**: If a job is interrupted, the UI remembers the machine's last known position. On restart, it injects a vertical lift command before homing, completely preventing diagonal drag crashes.
- **Smart Buffer Management (`nope`/`ready` Handshake)**: Fully handles Pico hardware buffer limits. When the firmware replies with `nope` (buffer full), the UI pauses and waits for a `ready` signal before resuming — ensuring massive files stream seamlessly without flooding the serial port.

### Output Format

The trajectory engine generates data strictly as **Steps** and **pure Integers** (no decimals) to optimize Pico firmware parsing. Every line follows this format:

```
xyz X Y Z Vx Vy Vz Angle
```

| Field | Description |
|-------|-------------|
| `xyz` | Required prefix identifier |
| `X, Y` | Relative change in target coordinates (Steps) |
| `Z` | Relative Z change — e.g. `6400` to plunge, `-6400` to lift |
| `Vx, Vy, Vz` | Normalized velocity vector (Steps/sec) |
| `Angle` | Relative tangential knife angle change (Steps, from Steps/Degree) |

> **Example line:** `xyz 1024 1024 6400 1200 400 0 45`

The very first line sent is always `enable all 1` to engage the stepper drivers.

### Machine Configuration

The **Settings modal (⚙️)** lets you configure:

- **Segment Length (mm)** — trajectory resolution
- **Cutting Speed (mm/sec)** — globally normalized feedrate
- **Motor Steps per Rev, Microstepping, mm per Revolution** — X/Y axis scaling
- **Z-Axis Steps per MM** — independent Z scaling
- **Rotary Steps per Degree** — independent knife/rotary axis scaling

### Running the Interface Locally

Because the WebSerial API requires a secure context, the app **cannot** be opened by double-clicking `index.html`. Serve it locally instead:

```bash
# Navigate to the data/ folder
npx serve
```

Then open the `localhost` URL shown in your terminal.

---

## Camera Vision System *(In Progress)*

A computer vision pipeline is currently being developed and integrated into the Urumi project, targeting a **Raspberry Pi 4** as the processing unit. The goal is to give the machine spatial awareness — enabling it to scan workpieces, detect geometry, and self-calibrate its coordinate system using visual feedback.

> **Status:** Actively under development. Core modules are functional; full system integration is ongoing.

### What's Been Built So Far

#### Image Stitching
A kinematic-aware image stitching pipeline has been implemented that captures multiple overlapping frames as the gantry moves across the workpiece. The frames are registered and merged into a single, high-resolution composite image of the full work area — effectively turning the camera into a large-format digital scanner.

#### ArUco Marker — Pixel-per-Metric Calibration
An **ArUco marker** of known physical dimensions is placed within the camera's field of view to establish a reliable **pixels-per-millimeter scale factor**. The pipeline:
1. Detects the ArUco marker using OpenCV's `aruco` module.
2. Measures the marker's pixel dimensions.
3. Computes the `pixel_per_metric` ratio from the known real-world size.

This scale factor is then applied globally, allowing all downstream measurements to be expressed in real-world millimeters rather than raw pixels.

![aruco ppm](./media/software/image(11).png)

#### Edge Detection
An edge detection stage has been implemented to identify workpiece boundaries and geometry within the captured frames. This feeds into the path-planning layer, letting the machine reason about the shape and extent of the material being processed.

#### Centering Motion
A closed-loop **centering motion** routine has been developed to align the camera (and by extension, the tool head) with a detected feature. The pipeline:
1. Detects a target feature (e.g., ArUco marker center or workpiece edge).
2. Computes the pixel offset between the detected feature and the image center.
3. Converts the pixel offset to real-world millimeters using the `pixel_per_metric` calibration.
4. Issues corrective movement commands to the gantry to bring the target into center frame.

This loop runs iteratively until the offset falls within an acceptable tolerance, achieving precise visual alignment without manual intervention.

![centring](./media/software/image(12).png)

### Planned Next Steps

- Full integration of stitching + edge detection + centering into a single orchestrated pipeline
- Real-time feedback loop between the vision system and the motion controller
- Automated workpiece boundary extraction to generate cut paths directly from camera input

<Sections :sections="$frontmatter.sections" />
