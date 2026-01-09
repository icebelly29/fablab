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
export function renderGCode(gcode, canvasId = 'gcodeCanvas', containerId = 'canvasContainer') {
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

    lines.forEach(line => {
        // Remove comments (text after ';') and whitespace
        line = line.split(';')[0].trim().toUpperCase();
        if (!line) return;

        // We only care about Move commands (G0 = Rapid, G1 = Cut)
        const isMove = line.startsWith('G0') || line.startsWith('G1');
        if (isMove) {
            // Use Regex to find numbers after X and Y
            const xMatch = line.match(/X([-+]?\d*\.?\d+)/);
            const yMatch = line.match(/Y([-+]?\d*\.?\d+)/);
            
            const next = { ...cur };
            if (xMatch) next.x = parseFloat(xMatch[1]);
            if (yMatch) next.y = parseFloat(yMatch[1]);

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
        ctx.beginPath();
        ctx.moveTo(mapX(p.from.x), mapY(p.from.y));
        ctx.lineTo(mapX(p.to.x), mapY(p.to.y));
        
        if (p.type === 'move') {
            // G0: Rapid Move (Pen Up) -> Grey Dashed Line
            ctx.strokeStyle = '#d1d5db'; 
            ctx.setLineDash([5, 5]);
        } else {
            // G1: Cut Move (Pen Down) -> Blue Solid Line
            ctx.strokeStyle = '#3b82f6'; 
            ctx.setLineDash([]);
        }
        ctx.stroke();
    });

    // Empty State
    if (paths.length === 0) {
        ctx.fillStyle = '#9ca3af';
        ctx.textAlign = 'center';
        ctx.setLineDash([]);
        ctx.fillText("No paths found", canvas.width/2, canvas.height/2);
    }
}