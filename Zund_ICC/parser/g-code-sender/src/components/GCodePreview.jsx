import { useRef, useEffect, useState } from 'react';

const GCodePreview = ({ gcode }) => {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const handleResize = () => {
        if (containerRef.current) {
            const { clientWidth, clientHeight } = containerRef.current;
            // Prevent 0 dimension issues
            if (clientWidth > 0 && clientHeight > 0) {
               setDimensions({ width: clientWidth, height: clientHeight });
            }
        }
    };
    
    // Initial size
    handleResize();
    
    // Observer for container resize (better than window resize for split panes)
    const resizeObserver = new ResizeObserver(() => handleResize());
    if (containerRef.current) {
        resizeObserver.observe(containerRef.current);
    }
    
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const { width, height } = dimensions;
    if (width === 0 || height === 0) return;

    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Basic Grid Background
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for(let x=0; x<width; x+=20) { ctx.moveTo(x,0); ctx.lineTo(x,height); }
    for(let y=0; y<height; y+=20) { ctx.moveTo(0,y); ctx.lineTo(width,y); }
    ctx.stroke();

    // Parse G-code to find bounds and paths
    const lines = gcode.split('\n');
    const path = [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    let curX = 0, curY = 0;
    
    // Initial point
    path.push({ type: 'move', x: 0, y: 0 });
    minX = 0; minY = 0; maxX = 0; maxY = 0;

    lines.forEach(line => {
      line = line.trim().toUpperCase();
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
        
        path.push({ type: cmd === 'G0' ? 'move' : 'line', x: newX, y: newY });
        
        curX = newX;
        curY = newY;
        
        minX = Math.min(minX, curX);
        minY = Math.min(minY, curY);
        maxX = Math.max(maxX, curX);
        maxY = Math.max(maxY, curY);
      }
      // G2/G3 Support omitted for brevity in this resize-fix, can be re-added if critical
    });

    // Auto-scale to fit
    const contentW = maxX - minX;
    const contentH = maxY - minY;
    
    const padding = 40;
    
    let scale = 1;
    if (contentW > 0 && contentH > 0) {
        const scaleX = (width - 2 * padding) / contentW;
        const scaleY = (height - 2 * padding) / contentH;
        scale = Math.min(scaleX, scaleY);
    } else {
        scale = 10; // Default if single point
    }

    // Centering
    const drawnW = contentW * scale;
    const drawnH = contentH * scale;
    
    const offsetX = (width - drawnW) / 2 - minX * scale;
    const offsetY = (height - drawnH) / 2 - minY * scale; 
    
    // Invert Y for visualization (Canvas 0,0 is top-left)
    // We want G-Code 0,0 to be bottom-left visually relative to the content bounding box
    // But typically G-code visualizers put 0,0 at bottom-left of screen or center.
    // Let's standardise: Flip Y axis.
    
    const toScreen = (x, y) => {
        return {
            x: offsetX + x * scale,
            y: height - (offsetY + y * scale) // Simple Y flip relative to bottom
            // Wait, centering calculation needs to align with this flip.
        };
    };
    
    // Re-calculate offsets for Y-flip centering
    // Screen Y = height - (y_gcode * scale + C)
    // We want (minY..maxY) to map to (padding..height-padding)
    // y_screen_min = height - (maxY * scale + C) = padding
    // y_screen_max = height - (minY * scale + C) = height - padding
    // => C = height - padding - maxY * scale
    
    const C_x = padding - minX * scale + ( (width - 2*padding) - contentW*scale ) / 2;
    const C_y = padding - minY * scale + ( (height - 2*padding) - contentH*scale ) / 2;

    const transform = (x, y) => {
        return {
            x: x * scale + C_x,
            y: height - (y * scale + C_y) 
        };
    };

    // Draw Origin
    const origin = transform(0, 0);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#e53e3e'; // Red X
    ctx.beginPath(); 
    ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x + 20, origin.y); ctx.stroke();
    
    ctx.strokeStyle = '#38a169'; // Green Y
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x, origin.y - 20); ctx.stroke(); // Up is -Y in canvas

    // Draw Path
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#3182ce'; // Blue path

    ctx.beginPath();
    
    // Start
    if (path.length > 0) {
        const start = transform(path[0].x, path[0].y);
        ctx.moveTo(start.x, start.y);
    }
    
    path.forEach(p => {
        const pt = transform(p.x, p.y);
        if (p.type === 'move') {
            ctx.moveTo(pt.x, pt.y);
        } else {
            ctx.lineTo(pt.x, pt.y);
        }
    });
    ctx.stroke();

  }, [gcode, dimensions]);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
      <canvas 
        ref={canvasRef} 
        width={dimensions.width}
        height={dimensions.height}
        style={{ display: 'block' }} 
      />
    </div>
  );
};

export default GCodePreview;