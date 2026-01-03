import { useRef, useEffect } from 'react';

const GCodePreview = ({ gcode }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Parse G-code to find bounds and paths
    const lines = gcode.split('\n');
    const path = [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    // State machine
    let curX = 0, curY = 0;
    
    // Initial point
    path.push({ type: 'move', x: 0, y: 0 });
    minX = 0; minY = 0; maxX = 0; maxY = 0;

    lines.forEach(line => {
      line = line.trim().toUpperCase();
      // Remove comments
      const commentIdx = line.indexOf(';');
      if (commentIdx !== -1) line = line.substring(0, commentIdx).trim();
      if (!line) return;

      const tokens = line.split(/\s+/);
      const cmd = tokens[0];
      
      const getVal = (char) => {
        const token = tokens.find(t => t.startsWith(char));
        return token ? parseFloat(token.substring(1)) : null;
      };

      if (cmd === 'G0' || cmd === 'G1') {
        const x = getVal('X');
        const y = getVal('Y');
        
        let newX = curX;
        let newY = curY;
        
        if (x !== null) newX = x;
        if (y !== null) newY = y;
        
        if (cmd === 'G0') {
            path.push({ type: 'move', x: newX, y: newY });
        } else {
            path.push({ type: 'line', x: newX, y: newY });
        }
        
        curX = newX;
        curY = newY;
        
        // Update bounds
        minX = Math.min(minX, curX);
        minY = Math.min(minY, curY);
        maxX = Math.max(maxX, curX);
        maxY = Math.max(maxY, curY);
      } else if (cmd === 'G2' || cmd === 'G3') {
          // Arcs
          const x = getVal('X');
          const y = getVal('Y');
          const i = getVal('I') || 0;
          const j = getVal('J') || 0;
          // R is not supported yet for simplicity, usually I/J are generated
          
          let endX = (x !== null) ? x : curX;
          let endY = (y !== null) ? y : curY;
          
          path.push({
              type: 'arc',
              x: endX,
              y: endY,
              cx: curX + i,
              cy: curY + j,
              cw: cmd === 'G2'
          });
          
          // Rough bounds for arcs (just using end points and center, effectively bounding box of arc is tricky)
          // Just using endpoints is "good enough" for simple preview unless arc is huge loop
          minX = Math.min(minX, endX, curX + i);
          minY = Math.min(minY, endY, curY + j);
          maxX = Math.max(maxX, endX, curX + i);
          maxY = Math.max(maxY, endY, curY + j);
          
          curX = endX;
          curY = endY;
      }
    });

    // Determine scale and offset
    const padding = 20;
    const width = maxX - minX;
    const height = maxY - minY;
    
    // Avoid division by zero
    if (width === 0 && height === 0) return;

    const scaleX = (canvas.width - 2 * padding) / width;
    const scaleY = (canvas.height - 2 * padding) / height;
    const scale = Math.min(scaleX, scaleY);
    
    const offsetX = padding - minX * scale + (canvas.width - 2*padding - width * scale)/2;
    const offsetY = padding - minY * scale + (canvas.height - 2*padding - height * scale)/2;
    // Note: G-code Y is usually up, Canvas Y is down. We might need to flip Y.
    // Let's flip Y for correct visual representation.
    // In G-code, +Y is UP. In Canvas, +Y is DOWN.
    // Transform: canvasY = canvasHeight - (gcodeY * scale + offsetY) ?? 
    // Let's just do a simple flip transform.
    
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    // Draw
    ctx.lineWidth = 1;
    ctx.lineCap = 'round';
    
    const toCanvas = (x, y) => {
        // Standard mapping:
        // cx = (x - minX) * scale + centered_offset_x
        // cy = canvas.height - ((y - minY) * scale + centered_offset_y)
        
        // Let's recompute centering with flip in mind
        const drawnW = width * scale;
        const drawnH = height * scale;
        
        const startX = (canvas.width - drawnW) / 2;
        const startY = (canvas.height - drawnH) / 2; // This is the "top" in canvas for the "top" of bounding box
        
        const cx = startX + (x - minX) * scale;
        const cy = canvas.height - (startY + (y - minY) * scale); // Flip Y
        
        return { x: cx, y: cy };
    };

    // Draw origin
    const origin = toCanvas(0, 0);
    ctx.beginPath();
    ctx.strokeStyle = '#ccc';
    ctx.moveTo(origin.x - 10, origin.y);
    ctx.lineTo(origin.x + 10, origin.y);
    ctx.moveTo(origin.x, origin.y - 10);
    ctx.lineTo(origin.x, origin.y + 10);
    ctx.stroke();

    // Draw path
    ctx.beginPath();
    ctx.strokeStyle = '#007bff';
    
    // Move to start
    let startP = toCanvas(0,0); // Default start
    if (path.length > 0 && path[0].type === 'move') {
        startP = toCanvas(path[0].x, path[0].y);
    }
    ctx.moveTo(startP.x, startP.y);
    
    let lastX = 0; // G-code coords
    let lastY = 0;

    path.forEach(p => {
        const pt = toCanvas(p.x, p.y);
        
        if (p.type === 'move') {
            ctx.moveTo(pt.x, pt.y);
            // Optionally render G0 as dashed line?
            // For now, simple moveTo (pen up)
        } else if (p.type === 'line') {
            ctx.lineTo(pt.x, pt.y);
        } else if (p.type === 'arc') {
            // Context arc is: arc(x, y, radius, startAngle, endAngle, counterclockwise)
            // We have center (cx, cy), current point (lastX, lastY), end point (p.x, p.y)
            // Calculate radius, startAngle, endAngle
            
            const radius = Math.sqrt(Math.pow(p.cx - lastX, 2) + Math.pow(p.cy - lastY, 2)) * scale;
            const center = toCanvas(p.cx, p.cy);
            
            // Angles in canvas space (Y flipped!)
            // Vector from center to start (lastX, lastY)
            // But wait, toCanvas flips Y. 
            // Standard atan2(y, x) works in standard cartesian. 
            // If we use standard math on G-code coords, we get standard angles.
            // When drawing on canvas with Y flip, 'counterclockwise' argument might need inverting.
            
            const startAngle = Math.atan2(lastY - p.cy, lastX - p.cx);
            const endAngle = Math.atan2(p.y - p.cy, p.x - p.cx);
            
            // ctx.arc draws based on screen coords.
            // If we use the transformed center and radius, we are in screen coords.
            // Screen Y is down.
            // G2 (CW in Gcode) -> CW in screen (if Y flipped, does orientation preserve? Yes, mirrored Y mirrors rotation direction)
            // G-code +Y Up. CW is -angle.
            // Screen +Y Down. CW is +angle.
            // A flip in Y inverts the meaning of CW/CCW.
            // So G2 (CW) becomes CCW in screen coords?
            // Let's visualize: 12 o'clock to 3 o'clock.
            // Gcode: (0,1) to (1,0). Center (0,0). CW.
            // Screen (Y flip): (0, H-1) [bottom] to (1, H) [bottom-right? No, H-0]. 
            // (0, -1) to (1, 0) relative to center. 
            // This is actually CCW on screen.
            // So G2 (CW) -> true (counterclockwise for ctx.arc?)
            // ctx.arc 'anticlockwise' param: true = anti-clockwise.
            // So G2 -> true. G3 -> false.
            
            ctx.arc(center.x, center.y, radius, -startAngle, -endAngle, p.cw); 
            // Note: atan2 returns angle in standard math (CCW from +X).
            // Canvas angles are CW from +X. So we negate the angles?
            // Actually, because of Y-flip, the "visual" angle is inverted.
            
            // Let's stick to MoveTo/LineTo for arcs for robustness if this is flaky, 
            // but let's try.
        }
        
        lastX = p.x;
        lastY = p.y;
    });
    
    ctx.stroke();

  }, [gcode]);

  return (
    <div className="gcode-preview">
      <h3>Preview</h3>
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={400} 
        style={{ border: '1px solid #ddd', background: '#fafafa', borderRadius: '4px' }}
      />
    </div>
  );
};

export default GCodePreview;
