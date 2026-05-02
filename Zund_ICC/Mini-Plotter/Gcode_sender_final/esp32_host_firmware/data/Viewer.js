/**
 * ============================================================================
 *                       G-CODE VISUALIZER (CANVAS)
 * ============================================================================
 * 
 * This module draws the "Map" of what the machine is going to do. It takes text
 * (G-code) and turns it into a picture on the screen.
 * 
 * THE BIG MATH PROBLEM (Coordinate Mapping):
 * 1. Machine World:
 *    - Origin (0,0) is at the BOTTOM-Left.
 *    - +Y goes UP.
 *    - Units are in Millimeters (mm).
 * 
 * 2. Computer Screen World (HTML Canvas):
 *    - Origin (0,0) is at the TOP-Left.
 *    - +Y goes DOWN.
 *    - Units are in Pixels (px).
 * 
 * HOW WE SOLVE IT:
 * We create two mapping functions 'mapX' and 'mapY' that act as translators.
 * [Machine X,Y] ---> [Scale] ---> [Flip Y] ---> [Offset] ---> [Screen Pixels]
 * 
 * VISUAL CUES:
 * - Solid Blue Lines: G1 (Cutting/Pen Down) - The machine is working.
 * - Dashed Grey Lines: G0 (Travel/Pen Up) - The machine is moving to a new spot.
 * - Dashed Box: Represents the physical size of the machine bed (230x310mm).
 * ============================================================================
 */

/**
 * @file Viewer.js
 * @description VISUALIZER
 * 
 * This module draws the G-code path on the HTML5 Canvas. 
 * 
 * CHALLENGE:
 * - Machine coordinates (Standard Cartesian): (0,0) is Bottom-Left. Y increases UP. 
 * - Computer Screen coordinates (Canvas): (0,0) is Top-Left. Y increases DOWN. 
 * 
 * We have to "map" (convert) every point from Machine Space to Screen Space.
 */

/**
 * Renders the G-code path.
 * @param {string} gcode - The raw G-code string.
 * @param {string} canvasId - HTML ID of the <canvas> element.
 * @param {string} containerId - HTML ID of the parent div (for sizing).
 */
export function renderGCode(gcode, canvasId = 'gcodeCanvas', containerId = 'canvasContainer', stepsPerMM = 1.0) {
    console.log("Viewer: Rendering G-Code with Sampling Points (v2)"); // Debug log to confirm update
    const canvas = document.getElementById(canvasId);
    const container = document.getElementById(containerId);
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');

    // --- 1. Setup Dimensions ---
    const bedW = 230; // Machine Width (mm)
    const bedH = 310; // Machine Height (mm)

    // Make the canvas match the size of its container div
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // --- 2. Parse G-Code ---
    // We need to turn text lines ("G1 X10 Y20") into number objects ({x:10, y:20}).
    const lines = gcode.split('\n');
    const paths = [];
    let cur = { x: 0, y: 0 }; // Current pen position (starts at 0,0)
    let isPenDown = false; // Track pen state based on relative Z changes

    lines.forEach(line => {
        // Remove comments (text after ';') and whitespace
        line = line.split(';')[0].trim().toUpperCase();
        if (!line) return;

        // --- NEW: Trajectory Format ---
        // Format: X Y Z Vx Vy Vz Angle (Space or Comma separated)
        if (!line.startsWith('G') && !line.startsWith('M') && (line.includes(',') || line.includes(' '))) {
            if (line.startsWith('X Y Z') || line.startsWith('XYZ X Y Z')) return; // Skip Header

            const parts = line.split(/[\s,]+/);
            if (parts.length > 0 && parts[0].toUpperCase() === 'XYZ') {
                parts.shift(); // Remove the "xyz" prefix
            }

            if (parts.length >= 3) {
                const x = parseFloat(parts[0]) / stepsPerMM;
                const y = parseFloat(parts[1]) / stepsPerMM;
                const zVal = parseFloat(parts[2]); // Relative Z change
                
                if (zVal > 0) isPenDown = true;  // Positive Z means moving Down
                else if (zVal < 0) isPenDown = false; // Negative Z means moving Up
                
                const isMove = !isPenDown;
                const next = { x, y };
                
                paths.push({
                    type: isMove ? 'move' : 'cut',
                    from: { ...cur },
                    to: { ...next }
                });
                cur = next;
            }
            return;
        }

        // --- LEGACY: G-Code Format ---
        const isMove = line.startsWith('G0') || line.startsWith('G1');
        if (isMove) {
            // Use Regex to find numbers after X and Y
            const xMatch = line.match(/X([-+]?\d*\.?\d+)/);
            const yMatch = line.match(/Y([-+]?\d*\.?\d+)/);
            
            const next = { ...cur };
            if (xMatch) next.x = parseFloat(xMatch[1]) / stepsPerMM;
            if (yMatch) next.y = parseFloat(yMatch[1]) / stepsPerMM;

            paths.push({
                type: line.startsWith('G0') ? 'move' : 'cut',
                from: { ...cur },
                to: { ...next }
            });
            cur = next; // Update current position
        }
    });

    // --- 3. Calculate Scale & Offset ---
    // We want the machine bed to fit nicely in the window with some padding.
    const padding = 40; // px
    const availableW = canvas.width - padding * 2;
    const availableH = canvas.height - padding * 2;
    
    // Calculate how much we need to shrink/grow 1mm to equal 1 pixel.
    const scaleX = availableW / bedW;
    const scaleY = availableH / bedH;
    const scale = Math.min(scaleX, scaleY); // Use smallest scale to fit both dimensions

    // Calculate margins to center the bed in the window
    const offsetX = (canvas.width - bedW * scale) / 2;
    const offsetY = (canvas.height - bedH * scale) / 2;

    // --- 4. Coordinate Mapper Functions ---
    // Converts Machine X (mm) to Canvas X (px)
    const mapX = (x) => x * scale + offsetX;
    
    // Converts Machine Y (mm) to Canvas Y (px)
    // Note the subtraction! Canvas Y=0 is top, Machine Y=0 is bottom.
    const mapY = (y) => canvas.height - (y * scale + offsetY); 

    // --- 5. Draw! ---
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear screen

    // Draw Bed Border
    ctx.setLineDash([10, 5]); // Dashed line
    ctx.strokeStyle = '#cbd5e1'; // Light grey
    ctx.lineWidth = 1;
    
    const bedX_canvas = mapX(0);
    const bedY_canvas = mapY(bedH); // Top-Left of bed in canvas coords
    
    ctx.strokeRect(bedX_canvas, bedY_canvas, bedW * scale, bedH * scale);
    
    // Draw Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px ui-monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`0,0 (BL)`, mapX(0), mapY(0) + 15); // Label Origin
    ctx.textAlign = 'right';
    ctx.fillText(`${bedW}x${bedH}mm`, mapX(bedW), mapY(bedH) - 5); // Label Size

    // Draw The Path
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    paths.forEach(p => {
        const startX = mapX(p.from.x);
        const startY = mapY(p.from.y);
        const endX = mapX(p.to.x);
        const endY = mapY(p.to.y);

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        
        if (p.type === 'move') {
            // G0: Rapid Move (Pen Up) -> Grey Dashed Line
            ctx.strokeStyle = '#d1d5db'; 
            ctx.setLineDash([5, 5]);
            ctx.stroke();
        } else {
            // G1: Cut Move (Pen Down) -> Blue Solid Line
            ctx.strokeStyle = '#3b82f6'; 
            ctx.setLineDash([]);
            ctx.stroke();

            // VISUALIZE SAMPLING POINTS
            // Draw a small dot at the end of every cut segment
            // Color: Bright Orange, Opaque
            ctx.fillStyle = '#ff6600'; 
            ctx.beginPath();
            ctx.arc(endX, endY, 3.0, 0, 2 * Math.PI); // Large 3px radius (6px wide)
            ctx.fill();
        }
    });

    // Empty State
    if (paths.length === 0) {
        ctx.fillStyle = '#9ca3af';
        ctx.textAlign = 'center';
        ctx.setLineDash([]);
        ctx.fillText("No paths found", canvas.width/2, canvas.height/2);
    }
}