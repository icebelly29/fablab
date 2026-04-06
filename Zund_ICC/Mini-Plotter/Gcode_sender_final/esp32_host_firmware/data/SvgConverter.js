/**
 * ============================================================================
 *                         SVG TO G-CODE CONVERTER
 * ============================================================================
 * 
 * This module is a custom-built geometry engine designed to translate complex
 * vector mathematics into simple linear machine movements.
 * 
 * 1. THE GEOMETRY ENGINE (Vector2 & CubicBezier)
 *    SVG drawings aren't just lists of points; they are mathematical formulas.
 *    - Vector2: Handles the heavy lifting of 2D math (addition, subtraction, 
 *      normalization, and distance checking).
 *    - CubicBezier: Implements the Bernstein Polynomial formula. It allows us 
 *      to calculate any exact point (x,y) along a curve at time 't' (0 to 1).
 * 
 * 2. SHAPE NORMALIZATION (The "Standardizer")
 *    SVG is messy — it has circles, rects, and paths. We convert EVERYTHING into
 *    a "Unified Path" format.
 *    - Circles are converted into 4 Cubic Bezier curves using a magic constant
 *      (0.552284), which approximates a circular arc with 99.9% accuracy.
 *    - Rectangles are converted into a sequence of 4 Linear commands.
 * 
 * 3. THE PATH TOKENIZER (Parsing)
 *    SVG path strings (the 'd' attribute) look like "M10,20 L30,40". 
 *    Our parser:
 *    - Tokenizes: Splits numbers from letters.
 *    - State Tracking: Remembers the "Last Command" to support SVG's shorthand
 *      (where you can omit letters if the command type stays the same).
 *    - Relative vs Absolute: Converts lower-case commands (relative) into 
 *      global coordinates by adding them to the current pen position.
 * 
 * 4. CURVE FLATTENING (Recursive Subdivision)
 *    CNC machines (GRBL/Marlin) only understand straight lines (G1). To draw
 *    a curve, we use a "Divide and Conquer" strategy:
 *    
 *    A. Look at a curve segment from Point A to Point B.
 *    B. Calculate the actual midpoint of the curve (the mathematical "truth").
 *    C. Calculate where the midpoint WOULD be if the line were perfectly straight.
 *    D. If the distance (error) between the two is greater than 'tolerance' 
 *       (e.g., 0.05mm), the segment is "too curvy".
 *    E. Split the curve into two halves and repeat the process (Recursion).
 *    F. If the error is tiny, we emit a single straight G1 line.
 *    
 *    This ensures high detail on sharp turns and fewer lines on flat sections.
 * 
 * 5. COORDINATE TRANSFORMATION PIPELINE
 *    Before a number becomes G-code, it goes through a 4-stage filter:
 *    1. Scale: Convert SVG units to real-world millimeters.
 *    2. Flip Y: Vertical inversion (Screen Y-down vs Machine Y-up).
 *    3. Offset: Shifting the drawing to the center of the physical bed.
 *    4. Rounding: Truncating to 6 decimal places for machine compatibility.
 * 
 * 6. SMART FILTERING
 *    The converter automatically detects and deletes "Page Borders" (rectangles
 *    that perfectly match the document size) so your plotter doesn't try to
 *    cut the edge of the paper.
 * 
 * 7. TANGENTIAL KNIFE SUPPORT
 *    Line segments automatically calculate an 'A' parameter (rotation angle). 
 *    - Angle Calculation: A = Math.atan2(dy, dx) * 180 / PI. This is used to 
 *      find the physical target heading based on the segment's start and end.
 *    - Normalization: The raw angle is mathematically wrapped using the formula 
 *      `((A % 360) + 360) % 360` so it always stays strictly within [0, 360).
 *    - Shortest Path & Sharp Corners: A shortest rotational difference test is 
 *      performed. If the heading change exceeds the `angleThreshold`, the 
 *      system automatically executes a sequence to lift the tool (Z-Up), 
 *      orient to the new angle, and plunge (Z-Down) to prevent material tearing.
 * ============================================================================
 */

/**
 * @file SvgConverter.js
 * @description A utility class to convert SVG path data into G-code commands for a CNC plotter.
 * Handles vector mathematics, curve flattening (Beziers), and coordinate transformations.
 */

/**
 * @class Vector2
 * @description Represents a 2D vector with basic arithmetic operations.
 */
class Vector2 {
  /**
   * @constructor
   * @param {number} x - The X coordinate.
   * @param {number} y - The Y coordinate.
   */
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

/**
 * @class CubicBezier
 * @description Represents a Cubic Bezier curve defined by 4 control points.
 */
class CubicBezier {
  /**
   * @constructor
   * @param {Vector2} p0 - Start point.
   * @param {Vector2} p1 - First control point.
   * @param {Vector2} p2 - Second control point.
   * @param {Vector2} p3 - End point.
   */
  constructor(p0, p1, p2, p3) {
    this.p0 = p0;
    this.p1 = p1;
    this.p2 = p2;
    this.p3 = p3;
  }

  /**
   * @method sample
   * @description Calculates a point on the curve at parameter t using Bernstein polynomials.
   * @param {number} t - Interpolation factor (0.0 to 1.0).
   * @returns {Vector2} The point on the curve.
   */
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

  /**
   * @method getLUT
   * @description Generates a Look-Up Table (LUT) of arc lengths.
   * @param {number} steps - Number of samples (e.g., 100).
   * @returns {Array} Array of { t, dist } objects.
   */
  getLUT(steps = 100) {
      const lut = [{ t: 0, dist: 0 }];
      let cur = this.p0;
      let totalDist = 0;
      for (let i = 1; i <= steps; i++) {
          const t = i / steps;
          const next = this.sample(t);
          totalDist += cur.dist(next);
          lut.push({ t: t, dist: totalDist });
          cur = next;
      }
      return lut;
  }
}

/**
 * @class SvgConverter
 * @description Main class for parsing SVG strings and generating G-code.
 */
class SvgConverter {
  /**
   * @constructor
   * @param {Object} options - Configuration options.
   * @param {number} [options.feedRate=300] - Movement speed for cutting (G1).
   * @param {number} [options.scale=1.0] - Global scaling factor.
   * @param {number} [options.offsetX=0] - X offset for centering/positioning.
   * @param {number} [options.offsetY=0] - Y offset for centering/positioning.
   * @param {number} [options.segmentLength=1.0] - Desired length of linear segments (mm).
   */
  constructor(options = {}) {
    this.feedRate = options.feedRate || 300; 
    this.scale = options.scale || 1.0;
    this.offsetX = options.offsetX || 0;
    this.offsetY = options.offsetY || 0;
    this.segmentLength = options.segmentLength || 1.0; 
    this.flipY = options.flipY || false;
    this.decimals = 0; 
    
    // Tangential knife settings
    this.zUp = options.zUp !== undefined ? options.zUp : 5;
    this.zDown = options.zDown !== undefined ? options.zDown : 0;
    this.angleThreshold = options.angleThreshold !== undefined ? options.angleThreshold : 10;
  }

  /**
   * @method transform
   * @description Applies scaling, flipping, and offsets to a point.
   * @param {Vector2} p - The point to transform.
   * @returns {Object} The transformed coordinates {x, y}.
   */
  transform(p) {
      const x = (p.x * this.scale) + this.offsetX;
      let y = (p.y * this.scale);
      if (this.flipY) {
          y = -y;
      }
      y += this.offsetY;
      return { x, y };
  }

  /**
   * @method convert
   * @description Converts an SVG string into a G-code string.
   * @param {string} svgContent - The raw XML string of the SVG file.
   * @returns {string} The generated G-code.
   */
  convert(svgContent) {
    const gcode = [];
    gcode.push('G21'); // Metric
    gcode.push('G28'); // Home
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
                const parts = vb.split(/[\S,]+/).map(parseFloat);
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
                    const translateMatch = transform.match(/translate\(\s*([-+]?[\d.]+)\s*[\s, ]\s*([-+]?[\d.]+)\s*\)/);
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
        const pathRegex = /<path[^>]*\bd=[\"']([^\"']+)["']/gi;
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

  /**
   * @method parseElement
   * @description Parses a DOM element (path, rect, circle) into a standardized list of path commands.
   * @param {Element} el - The DOM element.
   * @returns {Array} List of path commands.
   */
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

  /**
   * @method parsePathData
   * @description Parses SVG 'd' attribute string into command objects.
   * @param {string} d - The path data string.
   * @returns {Array} Array of command objects {type: 'M', args: [...]}.
   */
  parsePathData(d) {
     const tokens = d.match(/([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/g);
    if (!tokens) return [];
    return this.parseTokens(tokens);
  }

  /**
   * @method parseTokens
   * @description Internal helper to consume tokens and build command list.
   * @param {Array} tokens - List of string tokens.
   * @returns {Array} Commands.
   */
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

  /**
   * @method generateGcode
   * @description Converts parsed SVG commands into G-code strings.
   * @param {Array} commands - List of parsed commands.
   * @returns {Array} Array of G-code lines.
   */
  generateGcode(commands) {
    const gcode = [];
    let cur = new Vector2(0, 0);
    let start = new Vector2(0, 0); 
    let lastControl = new Vector2(0, 0);
    let lastCmdType = '';

    // Machine state tracking for tangential knife
    const state = {
        isPenDown: false,
        machineX: 0,
        machineY: 0,
        machineA: 0
    };

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
                const { x, y } = this.transform(p);
                
                // Lift if pen is down
                if (state.isPenDown) {
                    gcode.push(`G0 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zUp} A${state.machineA.toFixed(2)}`);
                }
                
                // Move to new position while up
                gcode.push(`G0 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)} Z${this.zUp} A${state.machineA.toFixed(2)}`);
                
                state.machineX = x;
                state.machineY = y;
                state.isPenDown = false;
                
                cur = p;
                start = p;
                lastControl = p;
                break;
            }
            case 'L': {
                const p = getPt(0);
                this.emitLinear(gcode, state, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'H': {
                const x = isRelative ? cur.x + args[0] : args[0];
                const p = new Vector2(x, cur.y);
                this.emitLinear(gcode, state, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'V': {
                const y = isRelative ? cur.y + args[0] : args[0];
                const p = new Vector2(cur.x, y);
                this.emitLinear(gcode, state, p);
                cur = p;
                lastControl = p;
                break;
            }
            case 'C': {
                const c1 = getPt(0);
                const c2 = getPt(2);
                const p = getPt(4);
                const bezier = new CubicBezier(cur, c1, c2, p);
                this.flattenBezier(gcode, state, bezier);
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
                this.flattenBezier(gcode, state, bezier);
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
                this.flattenBezier(gcode, state, bezier);
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
                this.flattenBezier(gcode, state, bezier);
                cur = p;
                lastControl = c1;
                break;
            }
            case 'Z': {
                this.emitLinear(gcode, state, start);
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
                 this.emitLinear(gcode, state, p);
                 cur = p;
                 lastControl = p;
                 break;
             }
        }
        lastCmdType = type;
    });

    // Lift pen at the end of the drawing
    if (state.isPenDown) {
        gcode.push(`G0 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zUp} A${state.machineA.toFixed(2)}`);
        state.isPenDown = false;
    }

    return gcode;
  }

  /**
   * @method emitLinear
   * @description Helper to generate a G1 linear cut command with tangential knife support.
   */
  emitLinear(gcode, state, p) {
      const { x, y } = this.transform(p);
      
      const dx = x - state.machineX;
      const dy = y - state.machineY;
      const dSq = dx * dx + dy * dy;
      
      // Skip emitting if distance is effectively 0
      if (dSq < 0.000001) {
          return;
      }

      let targetA = Math.atan2(dy, dx) * 180 / Math.PI;
      
      // Normalize to 0 .. 360 degrees strictly
      targetA = ((targetA % 360) + 360) % 360;
      
      // Calculate shortest rotation difference for sharp corner detection
      let diff = targetA - state.machineA;
      // Normalize to -180 .. 180
      diff = ((diff + 180) % 360 + 360) % 360 - 180;

      // Ensure 'M' moves start fresh if we are just starting drawing
      if (!state.isPenDown) {
          // Orient tool then plunge
          gcode.push(`G0 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zUp} A${targetA.toFixed(2)}`);
          gcode.push(`G1 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zDown} A${targetA.toFixed(2)} F${this.feedRate}`);
          state.isPenDown = true;
      } else if (Math.abs(diff) > this.angleThreshold) {
          // Sharp corner: Lift, Rotate, Plunge
          gcode.push(`G0 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zUp} A${state.machineA.toFixed(2)}`);
          gcode.push(`G0 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zUp} A${targetA.toFixed(2)}`);
          gcode.push(`G1 X${state.machineX.toFixed(this.decimals)} Y${state.machineY.toFixed(this.decimals)} Z${this.zDown} A${targetA.toFixed(2)} F${this.feedRate}`);
      }
      
      // Cut to target
      gcode.push(`G1 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)} Z${this.zDown} A${targetA.toFixed(2)} F${this.feedRate}`);
      
      state.machineX = x;
      state.machineY = y;
      state.machineA = targetA;
  }

  /**
   * @method flattenBezier
   * @description Subdivides a Bezier curve into equidistant linear segments.
   * This uses an Arc-Length Parameterization approach (LUT) instead of recursive subdivision.
   * @param {Array} gcode - Output buffer.
   * @param {Object} state - Machine state for tangential knife.
   * @param {CubicBezier} bezier - The curve to flatten.
   */
  flattenBezier(gcode, state, bezier) {
      // 1. Build a Look-Up Table (LUT) to map 't' to 'Distance'
      const steps = 50; // High res sampling to estimate length
      const lut = bezier.getLUT(steps);
      const totalLength = lut[lut.length - 1].dist;

      // 2. Determine how many segments we need to achieve the target segmentLength
      // We use Math.ceil to ensure we don't have a huge remainder segment.
      // e.g. Length 10.5, Target 1.0 -> 11 segments of ~0.95mm
      const numSegments = Math.ceil(totalLength / this.segmentLength);
      
      // Prevent division by zero for degenerate curves
      if (numSegments <= 0) {
          this.emitLinear(gcode, state, bezier.p3);
          return;
      }

      const actualStep = totalLength / numSegments;

      // 3. Walk the curve at equidistant intervals
      for (let i = 1; i <= numSegments; i++) {
          const targetDist = i * actualStep;

          // Find the 't' that corresponds to targetDist
          // We search the LUT for the interval [prev, next] containing targetDist
          let tFound = 1.0;
          for (let k = 0; k < lut.length - 1; k++) {
              if (lut[k].dist <= targetDist && lut[k+1].dist >= targetDist) {
                  // Linear Interpolation for 't'
                  const dStart = lut[k].dist;
                  const dEnd = lut[k+1].dist;
                  const tStart = lut[k].t;
                  const tEnd = lut[k+1].t;
                  
                  const ratio = (targetDist - dStart) / (dEnd - dStart);
                  tFound = tStart + (tEnd - tStart) * ratio;
                  break;
              }
          }

          // Sample the exact point at tFound
          const p = bezier.sample(tFound);
          this.emitLinear(gcode, state, p);
      }
      // Ensure the exact end point is hit (the loop covers it, but floating point might drift)
      // Actually, since targetDist eventually equals totalLength, the loop emits the end point.
      // But to be safe against float errors not reaching 1.0:
      // We can rely on the loop, or force the last point. 
      // The logic `i <= numSegments` guarantees we emit `numSegments` points.
      // The last one is `totalLength`.
  }
}

export default SvgConverter;


// Alternate method for flattening using recursive subdivision (not used in final code, but kept for reference)
// /**
//  * ============================================================================
//  *                SVG TO G-CODE CONVERTER: THE DEEP DIVE
//  * ============================================================================
//  * 
//  * This module is a custom-built geometry engine designed to translate complex
//  * vector mathematics into simple linear machine movements.
//  * 
//  * 1. THE GEOMETRY ENGINE (Vector2 & CubicBezier)
//  *    SVG drawings aren't just lists of points; they are mathematical formulas.
//  *    - Vector2: Handles the heavy lifting of 2D math (addition, subtraction, 
//  *      normalization, and distance checking).
//  *    - CubicBezier: Implements the Bernstein Polynomial formula. It allows us 
//  *      to calculate any exact point (x,y) along a curve at time 't' (0 to 1).
//  * 
//  * 2. SHAPE NORMALIZATION (The "Standardizer")
//  *    SVG is messy—it has circles, rects, and paths. We convert EVERYTHING into
//  *    a "Unified Path" format.
//  *    - Circles are converted into 4 Cubic Bezier curves using a magic constant
//  *      (0.552284), which approximates a circular arc with 99.9% accuracy.
//  *    - Rectangles are converted into a sequence of 4 Linear commands.
//  * 
//  * 3. THE PATH TOKENIZER (Parsing)
//  *    SVG path strings (the 'd' attribute) look like "M10,20 L30,40". 
//  *    Our parser:
//  *    - Tokenizes: Splits numbers from letters.
//  *    - State Tracking: Remembers the "Last Command" to support SVG's shorthand
//  *      (where you can omit letters if the command type stays the same).
//  *    - Relative vs Absolute: Converts lower-case commands (relative) into 
//  *      global coordinates by adding them to the current pen position.
//  * 
//  * 4. CURVE FLATTENING (Recursive Subdivision)
//  *    CNC machines (GRBL/Marlin) only understand straight lines (G1). To draw
//  *    a curve, we use a "Divide and Conquer" strategy:
//  *    
//  *    A. Look at a curve segment from Point A to Point B.
//  *    B. Calculate the actual midpoint of the curve (the mathematical "truth").
//  *    C. Calculate where the midpoint WOULD be if the line were perfectly straight.
//  *    D. If the distance (error) between the two is greater than 'tolerance' 
//  *       (e.g., 0.05mm), the segment is "too curvy".
//  *    E. Split the curve into two halves and repeat the process (Recursion).
//  *    F. If the error is tiny, we emit a single straight G1 line.
//  *    
//  *    This ensures high detail on sharp turns and fewer lines on flat sections.
//  * 
//  * 5. COORDINATE TRANSFORMATION PIPELINE
//  *    Before a number becomes G-code, it goes through a 4-stage filter:
//  *    1. Scale: Convert SVG units to real-world millimeters.
//  *    2. Flip Y: Vertical inversion (Screen Y-down vs Machine Y-up).
//  *    3. Offset: Shifting the drawing to the center of the physical bed.
//  *    4. Rounding: Truncating to 6 decimal places for machine compatibility.
//  * 
//  * 6. SMART FILTERING
//  *    The converter automatically detects and deletes "Page Borders" (rectangles
//  *    that perfectly match the document size) so your plotter doesn't try to
//  *    cut the edge of the paper.
//  * ============================================================================
//  */

// /**
//  * @file SvgConverter.js
//  * @description A utility class to convert SVG path data into G-code commands for a CNC plotter.
//  * Handles vector mathematics, curve flattening (Beziers), and coordinate transformations.
//  */

// /**
//  * @class Vector2
//  * @description Represents a 2D vector with basic arithmetic operations.
//  */
// class Vector2 {
//   /**
//    * @constructor
//    * @param {number} x - The X coordinate.
//    * @param {number} y - The Y coordinate.
//    */
//   constructor(x, y) {
//     this.x = x;
//     this.y = y;
//   }
//   add(v) { return new Vector2(this.x + v.x, this.y + v.y); }
//   sub(v) { return new Vector2(this.x - v.x, this.y - v.y); }
//   mul(s) { return new Vector2(this.x * s, this.y * s); }
//   div(s) { return new Vector2(this.x / s, this.y / s); }
//   dot(v) { return this.x * v.x + this.y * v.y; }
//   length() { return Math.sqrt(this.x * this.x + this.y * this.y); }
//   lengthSq() { return this.x * this.x + this.y * this.y; }
//   normalize() {
//     const l = this.length();
//     return l === 0 ? new Vector2(0, 0) : this.div(l);
//   }
//   dist(v) { return this.sub(v).length(); }
// }

// /**
//  * @class CubicBezier
//  * @description Represents a Cubic Bezier curve defined by 4 control points.
//  */
// class CubicBezier {
//   /**
//    * @constructor
//    * @param {Vector2} p0 - Start point.
//    * @param {Vector2} p1 - First control point.
//    * @param {Vector2} p2 - Second control point.
//    * @param {Vector2} p3 - End point.
//    */
//   constructor(p0, p1, p2, p3) {
//     this.p0 = p0;
//     this.p1 = p1;
//     this.p2 = p2;
//     this.p3 = p3;
//   }

//   /**
//    * @method sample
//    * @description Calculates a point on the curve at parameter t.
//    * @param {number} t - Interpolation factor (0.0 to 1.0).
//    * @returns {Vector2} The point on the curve.
//    */
//   sample(t) {
//     const t1 = 1 - t;
//     const a = t1 * t1 * t1;
//     const b = 3 * t1 * t1 * t;
//     const c = 3 * t1 * t * t;
//     const d = t * t * t;
//     return new Vector2(
//       a * this.p0.x + b * this.p1.x + c * this.p2.x + d * this.p3.x,
//       a * this.p0.y + b * this.p1.y + c * this.p2.y + d * this.p3.y
//     );
//   }
// }

// /**
//  * @class SvgConverter
//  * @description Main class for parsing SVG strings and generating G-code.
//  */
// class SvgConverter {
//   /**
//    * @constructor
//    * @param {Object} options - Configuration options.
//    * @param {number} [options.feedRate=300] - Movement speed for cutting (G1).
//    * @param {number} [options.scale=1.0] - Global scaling factor.
//    * @param {number} [options.offsetX=0] - X offset for centering/positioning.
//    * @param {number} [options.offsetY=0] - Y offset for centering/positioning.
//    * @param {number} [options.tolerance=0.05] - Tolerance for curve flattening (smaller = smoother but more lines).
//    */
//   constructor(options = {}) {
//     this.feedRate = options.feedRate || 300; 
//     this.scale = options.scale || 1.0;
//     this.offsetX = options.offsetX || 0;
//     this.offsetY = options.offsetY || 0;
//     this.tolerance = options.tolerance || 0.05; 
//     this.flipY = options.flipY || false;
//     this.decimals = 6; 
//   }

//   /**
//    * @method transform
//    * @description Applies scaling, flipping, and offsets to a point.
//    * @param {Vector2} p - The point to transform.
//    * @returns {Object} The transformed coordinates {x, y}.
//    */
//   transform(p) {
//       const x = (p.x * this.scale) + this.offsetX;
//       let y = (p.y * this.scale);
//       if (this.flipY) {
//           y = -y;
//       }
//       y += this.offsetY;
//       return { x, y };
//   }

//   /**
//    * @method convert
//    * @description Converts an SVG string into a G-code string.
//    * @param {string} svgContent - The raw XML string of the SVG file.
//    * @returns {string} The generated G-code.
//    */
//   convert(svgContent) {
//     const gcode = [];
//     gcode.push('G21'); // Metric
//     gcode.push('G28'); // Home
//     gcode.push('G90'); // Absolute

//     if (typeof DOMParser !== 'undefined') {
//         const parser = new DOMParser();
//         const doc = parser.parseFromString(svgContent, "image/svg+xml");
        
//         const svgRoot = doc.querySelector('svg');
//         let pageW = 0, pageH = 0;
//         if (svgRoot) {
//             // Try to get dimensions from viewBox or width/height
//             // Note: unit parsing is complex (mm, in, px). We'll assume unitless or px/mm match for simple detection.
//             // A robust solution would normalize units.
//             const vb = svgRoot.getAttribute('viewBox');
//             const w = svgRoot.getAttribute('width');
//             const h = svgRoot.getAttribute('height');
            
//             if (vb) {
//                 const parts = vb.split(/[\S,]+/).map(parseFloat);
//                 if (parts.length === 4) {
//                     pageW = parts[2];
//                     pageH = parts[3];
//                 }
//             } else if (w && h) {
//                 pageW = parseFloat(w);
//                 pageH = parseFloat(h);
//             }
//         }

//         // Query all convertable elements
//         const elements = doc.querySelectorAll('path, rect, circle, ellipse, line, polyline, polygon');
        
//         elements.forEach((el, index) => {
//             // FILTER 1: Ignore elements inside non-rendering containers
//             if (el.closest('defs, clipPath, mask, symbol, marker, pattern')) return;

//             // FILTER 2: Ignore hidden elements
//             const style = el.getAttribute('style') || '';
//             const display = el.getAttribute('display');
//             const visibility = el.getAttribute('visibility');
//             if (
//                 display === 'none' || 
//                 visibility === 'hidden' || 
//                 visibility === 'collapse' ||
//                 style.includes('display:none') || 
//                 style.includes('display: none') || 
//                 style.includes('visibility:hidden')
//             ) return;

//             // FILTER 3: Smart Page Border Detection
//             // If it's a rect at (0,0) with same dims as page, skip it.
//             if (el.tagName.toLowerCase() === 'rect' && pageW > 0 && pageH > 0) {
//                 const x = parseFloat(el.getAttribute('x') || 0);
//                 const y = parseFloat(el.getAttribute('y') || 0);
//                 const w = parseFloat(el.getAttribute('width') || 0);
//                 const h = parseFloat(el.getAttribute('height') || 0);
                
//                 // Tolerance for floating point/unit diffs
//                 const matchesSize = (Math.abs(w - pageW) < 1.0) && (Math.abs(h - pageH) < 1.0);
//                 const isAtOrigin = (Math.abs(x) < 1.0) && (Math.abs(y) < 1.0);
                
//                 if (matchesSize && isAtOrigin) {
//                     // It's likely a page border.
//                     return; 
//                 }
//             }

//             let id = el.getAttribute('id') || `shape${index}`;
//             gcode.push(`;${el.tagName}#${id}`);
            
//             // Extract rudimentary transform (translate only for now)
//             let offsetX = 0;
//             let offsetY = 0;
            
//             // Check self and parents for basic translation
//             // Note: This is NOT a full matrix stack implementation, just a helper for simple offsets.
//             // Full support requires a matrix library.
//             let parent = el;
//             while(parent && parent.tagName !== 'svg') {
//                 const transform = parent.getAttribute('transform');
//                 if (transform) {
//                     const translateMatch = transform.match(/translate\(\s*([-+]?[\d.]+)\s*[\s, ]\s*([-+]?[\d.]+)\s*\)/);
//                     if (translateMatch) {
//                         offsetX += parseFloat(translateMatch[1]);
//                         offsetY += parseFloat(translateMatch[2]);
//                     }
//                 }
//                 parent = parent.parentNode;
//             }

//             const commands = this.parseElement(el);
//             // Apply offset to M commands (absolute positioning assumption)
//             // If we are strictly absolute (G90), we just shift the coordinates.
//             if (offsetX !== 0 || offsetY !== 0) {
//                 commands.forEach(cmd => {
//                     if (cmd.args && cmd.args.length >= 2) {
//                         // Apply to all coordinate pairs.
//                         // M x y, L x y, C x1 y1 x2 y2 x y, etc.
//                         for (let k = 0; k < cmd.args.length; k += 2) {
//                             cmd.args[k] += offsetX;
//                             cmd.args[k+1] += offsetY;
//                         }
//                     }
//                 });
//             }

//             const shapeGcode = this.generateGcode(commands);
//             gcode.push(...shapeGcode);
//         });
        
//     } else {
//        // Node.js fallback (simplified regex for path only)
//         const pathRegex = /<path[^>]*\bd=[\"']([^\"']+)["']/gi;
//         let match;
//         while ((match = pathRegex.exec(svgContent)) !== null) {
//           const d = match[1];
//           gcode.push(`;path`);
//           const commands = this.parsePathData(d);
//           const shapeGcode = this.generateGcode(commands);
//           gcode.push(...shapeGcode);
//         }
//     }

//     // No Footer required based on examples, or maybe M30/M02 is implicit?
//     // Examples shown ended with the last move.
//     return gcode.join('\n');
//   }

//   /**
//    * @method parseElement
//    * @description Parses a DOM element (path, rect, circle) into a standardized list of path commands.
//    * @param {Element} el - The DOM element.
//    * @returns {Array} List of path commands.
//    */
//   parseElement(el) {
//       const tagName = el.tagName.toLowerCase();
//       // Normalized to Path commands
//       if (tagName === 'path') {
//           return this.parsePathData(el.getAttribute('d') || '');
//       } else if (tagName === 'rect') {
//           const x = parseFloat(el.getAttribute('x') || 0);
//           const y = parseFloat(el.getAttribute('y') || 0);
//           const w = parseFloat(el.getAttribute('width') || 0);
//           const h = parseFloat(el.getAttribute('height') || 0);
//           return [
//               { type: 'M', args: [x, y] },
//               { type: 'L', args: [x + w, y] },
//               { type: 'L', args: [x + w, y + h] },
//               { type: 'L', args: [x, y + h] },
//               { type: 'L', args: [x, y] } // Close
//           ];
//       } else if (tagName === 'circle') {
//           const cx = parseFloat(el.getAttribute('cx') || 0);
//           const cy = parseFloat(el.getAttribute('cy') || 0);
//           const r = parseFloat(el.getAttribute('r') || 0);
//           // Convert circle to 2 arcs or just approximate with beziers immediately
//           // Using 4 cubic beziers to approximate a circle
//           const k = 0.552284749831; // Magic number for circle approx
//           return [
//               { type: 'M', args: [cx + r, cy] },
//               { type: 'C', args: [cx + r, cy + k*r, cx + k*r, cy + r, cx, cy + r] },
//               { type: 'C', args: [cx - k*r, cy + r, cx - r, cy + k*r, cx - r, cy] },
//               { type: 'C', args: [cx - r, cy - k*r, cx - k*r, cy - r, cx, cy - r] },
//               { type: 'C', args: [cx + k*r, cy - r, cx + r, cy - k*r, cx + r, cy] }
//           ];
//       }
//       // TODO: Implement ellipse, line, polyline, polygon
//       return [];
//   }

//   /**
//    * @method parsePathData
//    * @description Parses SVG 'd' attribute string into command objects.
//    * @param {string} d - The path data string.
//    * @returns {Array} Array of command objects {type: 'M', args: [...]}.
//    */
//   parsePathData(d) {
//      const tokens = d.match(/([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/g);
//     if (!tokens) return [];
//     return this.parseTokens(tokens);
//   }

//   /**
//    * @method parseTokens
//    * @description Internal helper to consume tokens and build command list.
//    * @param {Array} tokens - List of string tokens.
//    * @returns {Array} Commands.
//    */
//   parseTokens(tokens) {
//       const commands = [];
//       let i = 0;
//       let lastCommand = null;

//       const eat = (n) => {
//           const args = [];
//           for(let k=0; k<n; k++) {
//               if (i >= tokens.length) break;
//               args.push(parseFloat(tokens[i++]));
//           }
//           return args;
//       };

//       while(i < tokens.length) {
//           let token = tokens[i];
//           let cmdType = token;
          
//           if (/^[a-zA-Z]$/.test(token)) {
//               cmdType = token;
//               i++;
//           } else {
//               if (lastCommand) {
//                   if (lastCommand.toUpperCase() === 'M') {
//                       cmdType = (lastCommand === 'M') ? 'L' : 'l';
//                   } else {
//                       cmdType = lastCommand;
//                   }
//               } else {
//                   i++; continue;
//               }
//           }

//           lastCommand = cmdType;
//           let args = [];
//           switch(cmdType.toUpperCase()) {
//               case 'M': args = eat(2); break;
//               case 'L': args = eat(2); break;
//               case 'H': args = eat(1); break;
//               case 'V': args = eat(1); break;
//               case 'C': args = eat(6); break;
//               case 'S': args = eat(4); break;
//               case 'Q': args = eat(4); break;
//               case 'T': args = eat(2); break;
//               case 'A': args = eat(7); break;
//               case 'Z': args = []; break;
//               default: i++; break;
//           }
//           commands.push({ type: cmdType, args: args });
//       }
//       return commands;
//   }

//   /**
//    * @method generateGcode
//    * @description Converts parsed SVG commands into G-code strings.
//    * @param {Array} commands - List of parsed commands.
//    * @returns {Array} Array of G-code lines.
//    */
//   generateGcode(commands) {
//     const gcode = [];
//     let cur = new Vector2(0, 0);
//     let start = new Vector2(0, 0); 
//     let lastControl = new Vector2(0, 0);
//     let lastCmdType = '';

//     // Track state to avoid redundant moves
//     let isPenDown = false;

//     commands.forEach(cmd => {
//         const isRelative = (cmd.type === cmd.type.toLowerCase());
//         const type = cmd.type.toUpperCase();
//         const args = cmd.args;

//         const getPt = (idx) => isRelative 
//             ? new Vector2(cur.x + args[idx], cur.y + args[idx+1]) 
//             : new Vector2(args[idx], args[idx+1]);

//         switch (type) {
//             case 'M': {
//                 const p = getPt(0);
//                 // M = Move (Pen Up)
//                 // G0 X Y
//                 const { x, y } = this.transform(p);
//                 gcode.push(`G0 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)}`);
//                 cur = p;
//                 start = p;
//                 lastControl = p;
//                 isPenDown = false;
//                 break;
//             }
//             case 'L': {
//                 const p = getPt(0);
//                 this.emitLinear(gcode, p);
//                 cur = p;
//                 lastControl = p;
//                 break;
//             }
//             case 'H': {
//                 const x = isRelative ? cur.x + args[0] : args[0];
//                 const p = new Vector2(x, cur.y);
//                 this.emitLinear(gcode, p);
//                 cur = p;
//                 lastControl = p;
//                 break;
//             }
//             case 'V': {
//                 const y = isRelative ? cur.y + args[0] : args[0];
//                 const p = new Vector2(cur.x, y);
//                 this.emitLinear(gcode, p);
//                 cur = p;
//                 lastControl = p;
//                 break;
//             }
//             case 'C': {
//                 const c1 = getPt(0);
//                 const c2 = getPt(2);
//                 const p = getPt(4);
//                 const bezier = new CubicBezier(cur, c1, c2, p);
//                 this.flattenBezier(gcode, bezier);
//                 cur = p;
//                 lastControl = c2;
//                 break;
//             }
//             case 'S': {
//                 let c1 = cur;
//                 if (lastCmdType === 'C' || lastCmdType === 'S') {
//                     c1 = cur.add(cur.sub(lastControl));
//                 }
//                 const c2 = getPt(0);
//                 const p = getPt(2);
//                 const bezier = new CubicBezier(cur, c1, c2, p);
//                 this.flattenBezier(gcode, bezier);
//                 cur = p;
//                 lastControl = c2;
//                 break;
//             }
//             case 'Q': {
//                 const c1 = getPt(0);
//                 const p = getPt(2);
//                 const cp1 = cur.add(c1.sub(cur).mul(2/3));
//                 const cp2 = p.add(c1.sub(p).mul(2/3));
//                 const bezier = new CubicBezier(cur, cp1, cp2, p);
//                 this.flattenBezier(gcode, bezier);
//                 cur = p;
//                 lastControl = c1;
//                 break;
//             }
//             case 'T': {
//                 let c1 = cur;
//                  if (lastCmdType === 'Q' || lastCmdType === 'T') {
//                     c1 = cur.add(cur.sub(lastControl));
//                 }
//                 const p = getPt(0);
//                  const cp1 = cur.add(c1.sub(cur).mul(2/3));
//                 const cp2 = p.add(c1.sub(p).mul(2/3));
//                 const bezier = new CubicBezier(cur, cp1, cp2, p);
//                 this.flattenBezier(gcode, bezier);
//                 cur = p;
//                 lastControl = c1;
//                 break;
//             }
//             case 'Z': {
//                 this.emitLinear(gcode, start);
//                 cur = start;
//                 lastControl = start;
//                 break;
//             }
//              case 'A': {
//                  // Fallback: approximate arc as linear segment to end point
//                  // Proper arc flattening is complex without `lyon`.
//                  // Given the examples use pure G1s, flattening is desired anyway.
//                  // TODO: Implement actual arc subdivision for A command.
//                  const p = getPt(5);
//                  this.emitLinear(gcode, p);
//                  cur = p;
//                  lastControl = p;
//                  break;
//              }
//         }
//         lastCmdType = type;
//     });

//     return gcode;
//   }

//   /**
//    * @method emitLinear
//    * @description Helper to generate a G1 linear cut command.
//    */
//   emitLinear(gcode, p) {
//       // G1 = Cut (Pen Down)
//       const { x, y } = this.transform(p);
//       gcode.push(`G1 X${x.toFixed(this.decimals)} Y${y.toFixed(this.decimals)} F${this.feedRate}`);
//   }

//   /**
//    * @method flattenBezier
//    * @description Recursively subdivides a bezier curve into linear segments.
//    */
//   flattenBezier(gcode, bezier) {
//       // Recursive subdivision or sampling
//       // Examples show high density segments.
//       // Let's use simple sampling for robustness and code size.
//       const segments = 20; // Or dynamic based on length/curvature
//       // Dynamic approach: check flatness
//       this.subdivideBezier(gcode, bezier, 0, 1);
//   }

//   /**
//    * @method subdivideBezier
//    * @description Recursive logic for bezier flattening.
//    */
//   subdivideBezier(gcode, bezier, t0, t1) {
//       const p0 = bezier.sample(t0);
//       const p1 = bezier.sample(t1);
      
//       // Check if segment is flat enough
//       // Midpoint check
//       const midT = (t0 + t1) / 2;
//       const pMidActual = bezier.sample(midT);
//       const pMidLinear = p0.add(p1.sub(p0).mul(0.5));
      
//       const dist = pMidActual.dist(pMidLinear);
      
//       if (dist < this.tolerance || (t1-t0) < 0.01) {
//           // Flat enough, emit line to end
//           this.emitLinear(gcode, p1);
//       } else {
//           this.subdivideBezier(gcode, bezier, t0, midT);
//           this.subdivideBezier(gcode, bezier, midT, t1);
//       }
//   }
// }

// export default SvgConverter;