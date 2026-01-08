class Vector2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  add(v) { return new Vector2(this.x + v.x, this.y + v.y); }
  sub(v) { return new Vector2(this.x - v.x, this.y - v.y); }
  mul(s) { return new Vector2(this.x * s, this.y * s); }
  div(s) { return new Vector2(this.x / s, this.y / s); }
  dot(v) { return this.x * v.x + this.y * v.y; }
  length() { return Math.sqrt(this.x * this.x + this.y * this.y); }
  lengthSq() { return this.x * this.x + this.y * this.y; }
  normalize() {
    const l = this.length();
    return l === 0 ? new Vector2(0, 0) : this.div(l);
  }
  dist(v) { return this.sub(v).length(); }
}

class CubicBezier {
  constructor(p0, p1, p2, p3) {
    this.p0 = p0;
    this.p1 = p1;
    this.p2 = p2;
    this.p3 = p3;
  }

  sample(t) {
    const t1 = 1 - t;
    const a = t1 * t1 * t1;
    const b = 3 * t1 * t1 * t;
    const c = 3 * t1 * t * t;
    const d = t * t * t;
    return new Vector2(
      a * this.p0.x + b * this.p1.x + c * this.p2.x + d * this.p3.x,
      a * this.p0.y + b * this.p1.y + c * this.p2.y + d * this.p3.y
    );
  }
}

class SvgConverter {
  constructor(options = {}) {
    this.feedRate = options.feedRate || 300; // Default to 300 matching examples
    this.scale = options.scale || 1.0;
    this.offsetX = options.offsetX || 0;
    this.offsetY = options.offsetY || 0;
    // Tolerance for flattening curves. Lower = more segments = smoother but larger file.
    // Example files are very dense, implying low tolerance (high precision).
    this.tolerance = options.tolerance || 0.05; 
    this.decimals = 6; // High precision matching examples
  }

  convert(svgContent) {
    const gcode = [];
    gcode.push('G21'); // Metric
    gcode.push('G90'); // Absolute

    if (typeof DOMParser !== 'undefined') {
        const parser = new DOMParser();
        const doc = parser.parseFromString(svgContent, "image/svg+xml");
        
        const svgRoot = doc.querySelector('svg');
        let pageW = 0, pageH = 0;
        if (svgRoot) {
            // Try to get dimensions from viewBox or width/height
            // Note: unit parsing is complex (mm, in, px). We'll assume unitless or px/mm match for simple detection.
            // A robust solution would normalize units.
            const vb = svgRoot.getAttribute('viewBox');
            const w = svgRoot.getAttribute('width');
            const h = svgRoot.getAttribute('height');
            
            if (vb) {
                const parts = vb.split(/[\s,]+/).map(parseFloat);
                if (parts.length === 4) {
                    pageW = parts[2];
                    pageH = parts[3];
                }
            } else if (w && h) {
                pageW = parseFloat(w);
                pageH = parseFloat(h);
            }
        }

        // Query all convertable elements
        const elements = doc.querySelectorAll('path, rect, circle, ellipse, line, polyline, polygon');
        
        elements.forEach((el, index) => {
            // FILTER 1: Ignore elements inside non-rendering containers
            if (el.closest('defs, clipPath, mask, symbol, marker, pattern')) return;

            // FILTER 2: Ignore hidden elements
            const style = el.getAttribute('style') || '';
            const display = el.getAttribute('display');
            const visibility = el.getAttribute('visibility');
            if (
                display === 'none' || 
                visibility === 'hidden' || 
                visibility === 'collapse' ||
                style.includes('display:none') || 
                style.includes('display: none') || 
                style.includes('visibility:hidden')
            ) return;

            // FILTER 3: Smart Page Border Detection
            // If it's a rect at (0,0) with same dims as page, skip it.
            if (el.tagName.toLowerCase() === 'rect' && pageW > 0 && pageH > 0) {
                const x = parseFloat(el.getAttribute('x') || 0);
                const y = parseFloat(el.getAttribute('y') || 0);
                const w = parseFloat(el.getAttribute('width') || 0);
                const h = parseFloat(el.getAttribute('height') || 0);
                
                // Tolerance for floating point/unit diffs
                const matchesSize = (Math.abs(w - pageW) < 1.0) && (Math.abs(h - pageH) < 1.0);
                const isAtOrigin = (Math.abs(x) < 1.0) && (Math.abs(y) < 1.0);
                
                if (matchesSize && isAtOrigin) {
                    // It's likely a page border.
                    return; 
                }
            }

            let id = el.getAttribute('id') || `shape${index}`;
            gcode.push(`;${el.tagName}#${id}`);
            
            // Extract rudimentary transform (translate only for now)
            let offsetX = 0;
            let offsetY = 0;
            
            // Check self and parents for basic translation
            // Note: This is NOT a full matrix stack implementation, just a helper for simple offsets.
            // Full support requires a matrix library.
            let parent = el;
            while(parent && parent.tagName !== 'svg') {
                const transform = parent.getAttribute('transform');
                if (transform) {
                    const translateMatch = transform.match(/translate\(\s*([-+]?[\d.]+)\s*[,\s]\s*([-+]?[\d.]+)\s*\)/);
                    if (translateMatch) {
                        offsetX += parseFloat(translateMatch[1]);
                        offsetY += parseFloat(translateMatch[2]);
                    }
                }
                parent = parent.parentNode;
            }

            const commands = this.parseElement(el);
            // Apply offset to M commands (absolute positioning assumption)
            // If we are strictly absolute (G90), we just shift the coordinates.
            if (offsetX !== 0 || offsetY !== 0) {
                commands.forEach(cmd => {
                    if (cmd.args && cmd.args.length >= 2) {
                        // Apply to all coordinate pairs.
                        // M x y, L x y, C x1 y1 x2 y2 x y, etc.
                        for (let k = 0; k < cmd.args.length; k += 2) {
                            cmd.args[k] += offsetX;
                            cmd.args[k+1] += offsetY;
                        }
                    }
                });
            }

            const shapeGcode = this.generateGcode(commands);
            gcode.push(...shapeGcode);
        });
        
    } else {
       // Node.js fallback (simplified regex for path only)
        const pathRegex = /<path[^>]*\bd=["']([^"']+)["']/gi;
        let match;
        while ((match = pathRegex.exec(svgContent)) !== null) {
          const d = match[1];
          gcode.push(`;path`);
          const commands = this.parsePathData(d);
          const shapeGcode = this.generateGcode(commands);
          gcode.push(...shapeGcode);
        }
    }

    // No Footer required based on examples, or maybe M30/M02 is implicit?
    // Examples shown ended with the last move.
    return gcode.join('\n');
  }

  parseElement(el) {
      const tagName = el.tagName.toLowerCase();
      // Normalized to Path commands
      if (tagName === 'path') {
          return this.parsePathData(el.getAttribute('d') || '');
      } else if (tagName === 'rect') {
          const x = parseFloat(el.getAttribute('x') || 0);
          const y = parseFloat(el.getAttribute('y') || 0);
          const w = parseFloat(el.getAttribute('width') || 0);
          const h = parseFloat(el.getAttribute('height') || 0);
          return [
              { type: 'M', args: [x, y] },
              { type: 'L', args: [x + w, y] },
              { type: 'L', args: [x + w, y + h] },
              { type: 'L', args: [x, y + h] },
              { type: 'L', args: [x, y] } // Close
          ];
      } else if (tagName === 'circle') {
          const cx = parseFloat(el.getAttribute('cx') || 0);
          const cy = parseFloat(el.getAttribute('cy') || 0);
          const r = parseFloat(el.getAttribute('r') || 0);
          // Convert circle to 2 arcs or just approximate with beziers immediately
          // Using 4 cubic beziers to approximate a circle
          const k = 0.552284749831; // Magic number for circle approx
          return [
              { type: 'M', args: [cx + r, cy] },
              { type: 'C', args: [cx + r, cy + k*r, cx + k*r, cy + r, cx, cy + r] },
              { type: 'C', args: [cx - k*r, cy + r, cx - r, cy + k*r, cx - r, cy] },
              { type: 'C', args: [cx - r, cy - k*r, cx - k*r, cy - r, cx, cy - r] },
              { type: 'C', args: [cx + k*r, cy - r, cx + r, cy - k*r, cx + r, cy] }
          ];
      }
      // TODO: Implement ellipse, line, polyline, polygon
      return [];
  }

  parsePathData(d) {
     const tokens = d.match(/([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/g);
    if (!tokens) return [];
    return this.parseTokens(tokens);
  }

  parseTokens(tokens) {
      const commands = [];
      let i = 0;
      let lastCommand = null;

      const eat = (n) => {
          const args = [];
          for(let k=0; k<n; k++) {
              if (i >= tokens.length) break;
              args.push(parseFloat(tokens[i++]));
          }
          return args;
      };

      while(i < tokens.length) {
          let token = tokens[i];
          let cmdType = token;
          
          if (/^[a-zA-Z]$/.test(token)) {
              cmdType = token;
              i++;
          } else {
              if (lastCommand) {
                  if (lastCommand.toUpperCase() === 'M') {
                      cmdType = (lastCommand === 'M') ? 'L' : 'l';
                  } else {
                      cmdType = lastCommand;
                  }
              } else {
                  i++; continue;
              }
          }

          lastCommand = cmdType;
          let args = [];
          switch(cmdType.toUpperCase()) {
              case 'M': args = eat(2); break;
              case 'L': args = eat(2); break;
              case 'H': args = eat(1); break;
              case 'V': args = eat(1); break;
              case 'C': args = eat(6); break;
              case 'S': args = eat(4); break;
              case 'Q': args = eat(4); break;
              case 'T': args = eat(2); break;
              case 'A': args = eat(7); break;
              case 'Z': args = []; break;
              default: i++; break;
          }
          commands.push({ type: cmdType, args: args });
      }
      return commands;
  }

  generateGcode(commands) {
    const gcode = [];
    let cur = new Vector2(0, 0);
    let start = new Vector2(0, 0); 
    let lastControl = new Vector2(0, 0);
    let lastCmdType = '';

    // Track state to avoid redundant moves
    let isPenDown = false;

    commands.forEach(cmd => {
        const isRelative = (cmd.type === cmd.type.toLowerCase());
        const type = cmd.type.toUpperCase();
        const args = cmd.args;

        const getPt = (idx) => isRelative 
            ? new Vector2(cur.x + args[idx], cur.y + args[idx+1]) 
            : new Vector2(args[idx], args[idx+1]);

        switch (type) {
            case 'M': {
                const p = getPt(0);
                // M = Move (Pen Up)
                // G0 X Y
                const x = (p.x * this.scale) + this.offsetX;
                const y = (p.y * this.scale) + this.offsetY;
                gcode.push(`G0 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)}`);
                cur = p;
                start = p;
                lastControl = p;
                isPenDown = false;
                break;
            }
            case 'L': {
                const p = getPt(0);
                this.emitLinear(gcode, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'H': {
                const x = isRelative ? cur.x + args[0] : args[0];
                const p = new Vector2(x, cur.y);
                this.emitLinear(gcode, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'V': {
                const y = isRelative ? cur.y + args[0] : args[0];
                const p = new Vector2(cur.x, y);
                this.emitLinear(gcode, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'C': {
                const c1 = getPt(0);
                const c2 = getPt(2);
                const p = getPt(4);
                const bezier = new CubicBezier(cur, c1, c2, p);
                this.flattenBezier(gcode, bezier);
                cur = p;
                lastControl = c2;
                break;
            }
            case 'S': {
                let c1 = cur;
                if (lastCmdType === 'C' || lastCmdType === 'S') {
                    c1 = cur.add(cur.sub(lastControl));
                }
                const c2 = getPt(0);
                const p = getPt(2);
                const bezier = new CubicBezier(cur, c1, c2, p);
                this.flattenBezier(gcode, bezier);
                cur = p;
                lastControl = c2;
                break;
            }
            case 'Q': {
                const c1 = getPt(0);
                const p = getPt(2);
                const cp1 = cur.add(c1.sub(cur).mul(2/3));
                const cp2 = p.add(c1.sub(p).mul(2/3));
                const bezier = new CubicBezier(cur, cp1, cp2, p);
                this.flattenBezier(gcode, bezier);
                cur = p;
                lastControl = c1;
                break;
            }
            case 'T': {
                let c1 = cur;
                 if (lastCmdType === 'Q' || lastCmdType === 'T') {
                    c1 = cur.add(cur.sub(lastControl));
                }
                const p = getPt(0);
                 const cp1 = cur.add(c1.sub(cur).mul(2/3));
                const cp2 = p.add(c1.sub(p).mul(2/3));
                const bezier = new CubicBezier(cur, cp1, cp2, p);
                this.flattenBezier(gcode, bezier);
                cur = p;
                lastControl = c1;
                break;
            }
            case 'Z': {
                this.emitLinear(gcode, start);
                cur = start;
                lastControl = start;
                break;
            }
             case 'A': {
                 // Fallback: approximate arc as linear segment to end point
                 // Proper arc flattening is complex without `lyon`. 
                 // Given the examples use pure G1s, flattening is desired anyway.
                 // TODO: Implement actual arc subdivision for A command.
                 const p = getPt(5);
                 this.emitLinear(gcode, p);
                 cur = p;
                 lastControl = p;
                 break;
             }
        }
        lastCmdType = type;
    });

    return gcode;
  }

  emitLinear(gcode, p) {
      // G1 = Cut (Pen Down)
      const x = (p.x * this.scale) + this.offsetX;
      const y = (p.y * this.scale) + this.offsetY;
      gcode.push(`G1 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)} F${this.feedRate}`);
  }

  flattenBezier(gcode, bezier) {
      // Recursive subdivision or sampling
      // Examples show high density segments.
      // Let's use simple sampling for robustness and code size.
      const segments = 20; // Or dynamic based on length/curvature
      // Dynamic approach: check flatness
      this.subdivideBezier(gcode, bezier, 0, 1);
  }

  subdivideBezier(gcode, bezier, t0, t1) {
      const p0 = bezier.sample(t0);
      const p1 = bezier.sample(t1);
      
      // Check if segment is flat enough
      // Midpoint check
      const midT = (t0 + t1) / 2;
      const pMidActual = bezier.sample(midT);
      const pMidLinear = p0.add(p1.sub(p0).mul(0.5));
      
      const dist = pMidActual.dist(pMidLinear);
      
      if (dist < this.tolerance || (t1-t0) < 0.01) {
          // Flat enough, emit line to end
          this.emitLinear(gcode, p1);
      } else {
          this.subdivideBezier(gcode, bezier, t0, midT);
          this.subdivideBezier(gcode, bezier, midT, t1);
      }
  }
}

export default SvgConverter;