/**
 * ============================================================================
 *                       FILE IMPORT & PROCESSING
 * ============================================================================
 * 
 * This module acts as the "Receptionist" for files. It decides what to do with
 * files dropped onto the page or selected by the user.
 * 
 * PIPELINE:
 * 1. Detection: Is it a .gcode file or an .svg file?
 * 
 * 2. G-CODE Path (Simple):
 *    - Just reads the text content.
 *    - Sends it directly to the machine.
 * 
 * 3. SVG Path (Complex):
 *    We must prepare the vector graphic for the physical constraints of the plotter.
 *    
 *    A. Unit Normalization:
 *       SVGs can be in pixels, inches, cm, mm, etc. We try to convert everything
 *       to millimeters (mm) to match the machine.
 *       
 *    B. Scaling (Auto-Fit):
 *       If the drawing is 500mm wide but the bed is only 230mm, we automatically
 *       shrink the drawing to fit safely within the margins.
 *       
 *    C. Centering:
 *       We calculate the offsets needed to place the drawing exactly in the
 *       middle of the bed.
 *       
 *    D. Coordinate Flip:
 *       Computer screens have (0,0) at the Top-Left.
 *       CNC machines usually have (0,0) at the Bottom-Left.
 *       We have to mathematically flip the Y-axis so the drawing isn't upside down.
 * 
 *    E. Conversion:
 *       Finally, we pass all these parameters to 'SvgConverter.js' to get the G-code.
 * ============================================================================
 */

/**
 * @file FileHandler.js
 * @description FILE IMPORT LOGIC
 * 
 * Handles loading files from the computer.
 * - If it's a G-Code file: Just load the text.
 * - If it's an SVG file: We have to do a lot of math to convert it to G-code.
 */

import SvgConverter from './SvgConverter.js';
import { log } from './Console.js';

/**
 * Process an uploaded file (SVG or GCode).
 * 
 * @param {File} file - The file object from Input or Drag/Drop.
 * @param {Function} onGCodeReady - Callback to save the new GCode.
 * @param {Function} onSwitchTab - Callback to change the view.
 */
export async function handleFile(file, onGCodeReady, onSwitchTab) {
    if (!file) return;
    log(`Loading ${file.name}...`, 'info');
    
    try {
        const text = await file.text();

        // --- CASE 1: SVG FILE ---
        if (file.name.toLowerCase().endsWith('.svg')) {
            
            // 1. Parse the XML
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, 'image/svg+xml');
            const svg = doc.querySelector('svg');

            // 2. Show the raw SVG in the "SVG Preview" tab
            if (svg) {
                // Force it to fit the preview window
                svg.style.width = '100%';
                svg.style.height = '100%';
                
                const svgPreview = document.getElementById('svgPreview');
                svgPreview.innerHTML = ''; 
                svgPreview.appendChild(svg);
            }

            // 3. Determine Dimensions (Complex!)
            // SVGs can use mm, cm, in, px, or no units at all.
            // We try to find the "Real World" size of the drawing.
            const bedW = 230;
            const bedH = 310;
            let w_mm = 0, h_mm = 0;
            let viewbox = [0, 0, 0, 0];

            if(svg) {
                const wAttr = svg.getAttribute('width');
                const hAttr = svg.getAttribute('height');
                const vbAttr = svg.getAttribute('viewBox');

                if (vbAttr) {
                    viewbox = vbAttr.split(/[ ,]+/).map(parseFloat);
                }
                
                // Helper to convert strings like "10in" to mm
                const parseToMM = (str) => {
                    if (!str) return 0;
                    const val = parseFloat(str);
                    if (isNaN(val)) return 0;
                    if (str.endsWith('mm')) return val;
                    if (str.endsWith('cm')) return val * 10;
                    if (str.endsWith('in')) return val * 25.4;
                    if (str.endsWith('pt')) return val * (25.4 / 72);
                    if (str.endsWith('pc')) return val * (25.4 / 6);
                    if (str.endsWith('px')) return val * 0.264583;
                    return val; // Assume mm if no unit provided
                };

                w_mm = parseToMM(wAttr);
                h_mm = parseToMM(hAttr);

                // Fallback: If width/height are missing, use ViewBox width/height
                if (w_mm === 0 && viewbox.length === 4) w_mm = viewbox[2];
                if (h_mm === 0 && viewbox.length === 4) h_mm = viewbox[3];
            }

            // --- SCALING LOGIC ---
            
            // 1. Base Dimensions from ViewBox or Attributes
            let vbW = viewbox.length === 4 ? viewbox[2] : w_mm;
            let vbH = viewbox.length === 4 ? viewbox[3] : h_mm;
            
            if (vbW === 0) vbW = w_mm;
            if (vbH === 0) vbH = h_mm;

            // 2. Calculate initial scale (Unit Conversion)
            // e.g., if ViewBox is 100 wide but Width is 100mm, scale is 1.
            let scale = (vbW > 0) ? (w_mm / vbW) : 1.0;

            const margin = 10; // 10mm safety margin
            
            let currentW = vbW * scale;
            let currentH = vbH * scale;

            // 3. Auto-Fit (Scale Down)
            // If the drawing is bigger than the bed, shrink it to fit.
            if (currentW > (bedW - margin) || currentH > (bedH - margin)) {
                const scaleW = (bedW - margin) / currentW;
                const scaleH = (bedH - margin) / currentH;
                const fitScale = Math.min(scaleW, scaleH);
                scale *= fitScale;
                log(`Scaled down to fit bed (${(fitScale * 100).toFixed(0)}%)`, 'info');
            }

            // 4. Centering
            // We want the drawing right in the middle of the bed.
            const finalW = vbW * scale;
            const finalH = vbH * scale;

            const offsetX = (bedW - finalW) / 2;
            const offsetY = (bedH - finalH) / 2;
            const vbMinX = viewbox.length === 4 ? viewbox[0] : 0;
            const vbMinY = viewbox.length === 4 ? viewbox[1] : 0;

            // Calculate final offset to shift the origin
            const finalOffsetX = offsetX - (vbMinX * scale);
            // Flip Y correction: We are flipping Y, so we need to shift the "bottom" (max Y in SVG) to the "bottom" offset.
            // Original: offsetY - (vbMinY * scale) (for non-flipped)
            // Flipped: offsetY + (vbMinY + vbH) * scale
            // Wait, (vbMinY + vbH) is the max Y coordinate in SVG space.
            // When flipped, it becomes negative.
            // So we add it back? Let's stick to the requested formula: offsetY + (vbMinY + vbH) * scale
            const finalOffsetY = offsetY + (vbMinY + vbH) * scale;

            try {
                // Run the conversion!
                const converter = new SvgConverter({
                    feedRate: 1000, 
                    scale: scale,
                    offsetX: finalOffsetX,
                    offsetY: finalOffsetY,
                    flipY: true
                });
                const gcode = converter.convert(text);
                
                onGCodeReady(gcode);
                log(`Converted (Size: ${finalW.toFixed(1)}x${finalH.toFixed(1)}mm)`, 'success');
                onSwitchTab('gcode-preview');
            } catch (err) {
                log(`Conversion Error: ${err.message}`, 'error');
            }

        } else {
            // --- CASE 2: G-CODE FILE ---
            // Simple: just read the text and use it.
            onGCodeReady(text);
            log('G-Code loaded.', 'success');
            onSwitchTab('gcode-preview');
        }
    } catch (err) {
        log(`File Read Error: ${err.message}`, 'error');
    }
}