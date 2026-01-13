# SVG to G-Code Converter: Logic & Architecture

## 1. System Overview

The **SvgConverter** is a specialized module designed to translate high-level vector graphics (SVG) into low-level machine instructions (G-code). This process enables a standard 3-axis CNC machine, which typically understands only simple linear movements, to reproduce complex designs created in software like Adobe Illustrator or Inkscape.

The core challenge this system solves is **Linearization**: converting mathematical curves (Bezier curves) into a series of precise linear segments that approximate the original shape.

---

## 2. The Conversion Pipeline

The conversion process executes in four distinct stages:

### Stage 1: Parsing
The system reads the raw SVG data. It identifies geometric primitives (Paths, Rectangles, Circles) and extracts their coordinate data.
*   **Input**: `<path d="M 0 0 L 10 10" />`
*   **Action**: Tokenizes the string into a command list: `[MoveTo(0,0), LineTo(10,10)]`.

### Stage 2: Normalization
All shapes are converted into a standardized format.
*   **Rectangles** are converted into 4 connected line segments.
*   **Circles** are converted into 4 connected Cubic Bezier curves.
*   **Relative Coordinates** (e.g., "move 10mm right") are converted to **Absolute Coordinates** (e.g., "move to X=50").

### Stage 3: Linearization (Curve Flattening)
This is the most critical step. CNC machines generally cannot execute a "Curve" command directly. The system uses an **Adaptive Subdivision Algorithm** to approximate curves.

**The Logic:**
1.  The algorithm analyzes a section of a curve.
2.  It compares the curved path to a straight line connecting the start and end points.
3.  **Deviation Check**: If the curve deviates from the straight line by more than the allowed `tolerance` (e.g., 0.05mm), the curve is split in half.
4.  This process repeats recursively until every segment is "flat" enough to be drawn as a straight line without visible loss of quality.

### Stage 4: Generation
The finalized list of linear segments is formatted into G-code strings.
*   **Rapid Moves (G0)**: Used when the pen needs to move to a new start point without drawing.
*   **Linear Cuts (G1)**: Used for all drawing movements.

---

## 3. Supported SVG Commands

The converter interprets the standard SVG path command set.

| Command | Name | Description | G-Code Output |
| :--- | :--- | :--- | :--- |
| **M / m** | **Move** | Lifts the pen and moves to a specific coordinate. | `G0 X... Y...` |
| **L / l** | **Line** | Lowers the pen and draws a straight line to a coordinate. | `G1 X... Y...` |
| **H / V** | **Horizontal / Vertical** | Optimized commands for perfectly straight lines. | `G1 X... Y...` |
| **C / c** | **Cubic Bezier** | A complex curve defined by 4 control points. | Series of `G1` segments |
| **S / s** | **Smooth Cubic** | A curve that continues smoothly from the previous one. | Series of `G1` segments |
| **Q / q** | **Quadratic Bezier** | A simpler curve defined by 3 control points. | Series of `G1` segments |
| **Z / z** | **Close Path** | Draws a straight line back to the shape's starting point. | `G1 X... Y...` |

> **Note**: The **Arc (A)** command is currently approximated as a linear segment to the endpoint. For best results, convert Arcs to Paths in your design software before exporting.

---

## 4. Configuration

The converter behavior can be tuned using the following parameters:

*   **`feedRate` (Default: 300)**:
    The speed at which the machine moves while drawing (in mm/minute).
    *   *Higher* = Faster completion.
    *   *Lower* = Better quality, especially with thick mediums.

*   **`tolerance` (Default: 0.05)**:
    The maximum allowed error (deviation) when flattening curves.
    *   *Lower (e.g., 0.01)* = Extremely smooth curves, but generates very large G-code files.
    *   *Higher (e.g., 0.5)* = Smaller files, but curves may appear "blocky" or faceted.

*   **`scale` (Default: 1.0)**:
    Global scaling factor.
    *   Use `1.0` for 1:1 scale (mm to mm).
    *   Use `25.4` to convert inches to millimeters.

*   **`flipY` (Default: false)**:
    Controls vertical orientation.
    *   *true* = Flips the Y-axis (useful for correcting SVG coordinates where Y+ is down vs. CNC where Y+ is up).
    *   *false* = Standard SVG orientation.

---

## 5. Developer Guide (Code Structure)

For developers maintaining `SvgConverter.js`, the class structure is organized as follows:

*   **`Vector2` Class**:
    Handles all 2D math (addition, subtraction, distance calculation). This ensures code readability by abstracting vector math.

*   **`CubicBezier` Class**:
    Represents a curve. Contains the `sample(t)` method, which returns the X,Y coordinate at a specific percentage (t) of the curve.

*   **`SvgConverter` Class**:
    *   **`convert(svgString)`**: The entry point. Manages the DOM parsing and overall flow.
    *   **`transform(vector)`**: Helper method that applies scaling, Y-axis flipping (if enabled), and offsets to every coordinate point before G-code generation.
    *   **`parsePathData(d)`**: Implements the state machine that reads SVG path strings.
    *   **`flattenBezier(bezier)`**: The recursive function that implements the Adaptive Subdivision logic described in Stage 3.
    *   **`generateGcode(commands)`**: Formats the final numerical data into standard G-code syntax.
