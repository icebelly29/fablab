# Bezier Kinematics Explainer

This folder contains a utility for calculating the kinematic state (Position, Velocity, and Acceleration) of a **3D Cubic Bezier Curve**.

## Interactive Visualizer

Since Markdown cannot run code, I have included a standalone HTML tool in this folder to visualize these concepts interactively (similar to cubic-bezier.com).

**[>> Click here to open `visualizer.html` <<](./visualizer.html)**

*(Note: If clicking the link doesn't work directly in your editor, simply open the `visualizer.html` file in your web browser.)*

### Features of the Visualizer
*   **Drag & Drop**: Move the 4 control points ($P_0, P_1, P_2, P_3$) to reshape the curve.
*   **Real-time Physics**: See the Velocity (orange) and Acceleration (blue) vectors attached to the moving point.
*   **Live Data**: Watch the X/Y kinematics values update instantly as you interact.

---

## Input Parameters

The function `getBezierKinematics(t, points)` requires two specific inputs:

1.  **`t` (The Parameter/Time)**
    *   **Type:** `Number`
    *   **Range:** `0.0` to `1.0`.
    *   **Description:** Represents the normalized progress along the curve. 
        *   `0.0` = The very start of the curve ($P_0$).
        *   `0.5` = The "middle" of the curve in terms of the math (not necessarily arc-length).
        *   `1.0` = The very end of the curve ($P_3$).

2.  **`points` (Control Points)**
    *   **Type:** `Array` of 4 `Arrays`.
    *   **Structure:** `[[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]`.
    *   **Roles:**
        *   **$P_0$ (Start Point):** The curve starts exactly here.
        *   **$P_1$ (First Control):** Pulls the curve away from $P_0$. The line segment $P_0 \to P_1$ defines the initial direction.
        *   **$P_2$ (Second Control):** Pulls the curve toward $P_3$. The line segment $P_2 \to P_3$ defines the final direction.
        *   **$P_3$ (End Point):** The curve ends exactly here.

---

## Output Variable Breakdown

The output is a single array `[X, Vx, Ax, Y, Vy, Ay, Z, Vz, Az]`.

### The Coordinates (Position)
*   **X, Y, Z**: These are the spatial coordinates of the point at time $t$. 
*   **Units**: Same as your input control points (e.g., mm, inches, or pixels).

### The Velocities (First Derivative)
*   **Vx, Vy, Vz**: The rate of change of the position relative to $t$.
*   **Meaning**: If you imagine $t$ moving from 0 to 1 in exactly 1 second, these values represent your speed in units/second.
*   **Use Case**: Essential for feedrate calculation and ensuring the plotter moves at a constant speed.

### The Accelerations (Second Derivative)
*   **Ax, Ay, Az**: The rate of change of the velocity relative to $t$.
*   **Meaning**: How hard the "motor" is pushing to change direction or speed.
*   **Use Case**: Used to prevent "jerk" or mechanical vibration by ensuring acceleration stays within your motor's physical limits.

---

## Math Formulas

The script treats the Bezier parameter $t$ as the independent variable:
The position is the standard Cubic Bezier equation:
$$B(t) = (1-t)^3P_0 + 3(1-t)^2tP_1 + 3(1-t)t^2P_2 + t^3P_3$$

### 2. Velocity $V(t)$ (First Derivative)
Velocity is the rate of change of position. It tells you the direction and speed at any point $t$:
$$B'(t) = 3(1-t)^2(P_1-P_0) + 6(1-t)t(P_2-P_1) + 3t^2(P_3-P_2)$$

### 3. Acceleration $A(t)$ (Second Derivative)
Acceleration is the rate of change of velocity. It is useful for smooth motion planning and ensuring motors don't jerk:
$$B''(t) = 6(1-t)(P_2-2P_1+P_0) + 6t(P_3-2P_2+P_1)$$

---

## Output Format

The function `getBezierKinematics(t, points)` returns a flat array of 9 numbers:

`[ X, Vx, Ax, Y, Vy, Ay, Z, Vz, Az ]`

| Value | Description |
| :--- | :--- |
| **X, Y, Z** | Instantaneous coordinates at $t$ |
| **Vx, Vy, Vz** | Velocity components (speed/direction) |
| **Ax, Ay, Az** | Acceleration components |

## Usage

1. Define 4 control points as `[[x,y,z], [x,y,z], [x,y,z], [x,y,z]]`.
2. Choose a $t$ value between `0.0` (start) and `1.0` (end).
3. Call the function to get the kinematic state.

```javascript
const points = [[0,0,0], [10,10,0], [20,10,0], [30,0,0]];
const state = getBezierKinematics(0.5, points);
// state = [x, vx, ax, y, vy, ay, z, vz, az]
```
