// bezier_kinematics.js

/**
 * Calculates the kinematic properties (Position, Velocity, Acceleration) 
 * for a 3D Cubic Bezier curve at a specific parameter t.
 * 
 * @param {number} t - The parameter t (0 <= t <= 1)
 * @param {Array<Array<number>>} points - Array of 4 control points [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
 * @returns {Array<number>} - [X, Vx, Ax, Y, Vy, Ay, Z, Vz, Az]
 */
function getBezierKinematics(t, points) {
    if (points.length !== 4) {
        throw new Error("Cubic Bezier requires exactly 4 control points.");
    }

    // Helper for 3D Cubic Bezier Position
    // B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    const pos = (p0, p1, p2, p3, t) => {
        const u = 1 - t;
        return (u ** 3) * p0 + 
               3 * (u ** 2) * t * p1 + 
               3 * u * (t ** 2) * p2 + 
               (t ** 3) * p3;
    };

    // Helper for 3D Cubic Bezier Velocity (First Derivative)
    // B'(t) = 3(1-t)^2*(P1-P0) + 6(1-t)*t*(P2-P1) + 3t^2*(P3-P2)
    const vel = (p0, p1, p2, p3, t) => {
        const u = 1 - t;
        return 3 * (u ** 2) * (p1 - p0) + 
               6 * u * t * (p2 - p1) + 
               3 * (t ** 2) * (p3 - p2);
    };

    // Helper for 3D Cubic Bezier Acceleration (Second Derivative)
    // B''(t) = 6(1-t)*(P2-2P1+P0) + 6t*(P3-2P2+P1)
    const acc = (p0, p1, p2, p3, t) => {
        const u = 1 - t;
        return 6 * u * (p2 - 2 * p1 + p0) + 
               6 * t * (p3 - 2 * p2 + p1);
    };

    const [P0, P1, P2, P3] = points;

    // Calculate components
    const x = pos(P0[0], P1[0], P2[0], P3[0], t);
    const vx = vel(P0[0], P1[0], P2[0], P3[0], t);
    const ax = acc(P0[0], P1[0], P2[0], P3[0], t);

    const y = pos(P0[1], P1[1], P2[1], P3[1], t);
    const vy = vel(P0[1], P1[1], P2[1], P3[1], t);
    const ay = acc(P0[1], P1[1], P2[1], P3[1], t);

    const z = pos(P0[2], P1[2], P2[2], P3[2], t);
    const vz = vel(P0[2], P1[2], P2[2], P3[2], t);
    const az = acc(P0[2], P1[2], P2[2], P3[2], t);

    return [x, vx, ax, y, vy, ay, z, vz, az];
}

// --- Usage Example ---

// Define 4 Control Points (P0, P1, P2, P3)
// Each point is [x, y, z]
const controlPoints = [
    [0, 0, 0],    // Start
    [10, 20, 5],  // Control Point 1
    [20, 20, 5],  // Control Point 2
    [30, 0, 0]    // End
];

const t = 0.5; // Evaluate at mid-curve

const result = getBezierKinematics(t, controlPoints);

console.log(`t = ${t}`);
console.log("Result [x, vx, ax, y, vy, ay, z, vz, az]:");
console.log(result);

// Optional: Detailed Log
console.log("\nDetailed Breakdown:");
console.log(`X: Pos=${result[0].toFixed(2)}, Vel=${result[1].toFixed(2)}, Acc=${result[2].toFixed(2)}`);
console.log(`Y: Pos=${result[3].toFixed(2)}, Vel=${result[4].toFixed(2)}, Acc=${result[5].toFixed(2)}`);
console.log(`Z: Pos=${result[6].toFixed(2)}, Vel=${result[7].toFixed(2)}, Acc=${result[8].toFixed(2)}`);
