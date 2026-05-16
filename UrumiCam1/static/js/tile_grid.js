/**
 * ============================================================================
 *  URUMICAM — TILE GRID RENDERER
 * ============================================================================
 * Canvas-based tile grid with colour-coded states, dashed ROI overlay,
 * and MANDATORY settling pulse animation.
 *
 * Tile Colour States:
 *   pending      — transparent, neutral border
 *   targeting    — blue fill, blue border
 *   settling     — amber fill, amber border, PULSING ANIMATION
 *   capturing    — green fill, green border
 *   complete     — green fill, faint border
 *   failed_focus — amber fill, amber border, "!F" label
 *   failed_motor — red fill, red border, "!M" label
 * ============================================================================
 */

const TileGrid = (() => {
    // Colour definitions per tile state
    const COLORS = {
        pending:      { fill: 'rgba(255,255,255,0.03)', stroke: 'rgba(255,255,255,0.08)', label: null },
        targeting:    { fill: 'rgba(88,166,255,0.25)',   stroke: '#58a6ff',                label: null },
        settling:     { fill: 'rgba(210,153,34,0.3)',    stroke: '#d29922',                label: null },
        capturing:    { fill: 'rgba(63,185,80,0.3)',     stroke: '#3fb950',                label: null },
        complete:     { fill: 'rgba(63,185,80,0.15)',    stroke: 'rgba(63,185,80,0.3)',    label: null },
        failed_focus: { fill: 'rgba(210,153,34,0.3)',    stroke: '#d29922',                label: '!F' },
        failed_motor: { fill: 'rgba(248,81,73,0.3)',     stroke: '#f85149',                label: '!M' },
    };

    let canvas = null;
    let ctx = null;
    let container = null;
    let emptyState = null;
    let tiles = {};         // Map: "row_col" -> tile data
    let gridRows = 0;
    let gridCols = 0;
    let roi = null;         // ROI bounding box data
    let animFrame = null;
    let pulsePhase = 0;

    function init() {
        canvas = document.getElementById('tileCanvas');
        ctx = canvas.getContext('2d');
        container = document.getElementById('gridContainer');
        emptyState = document.getElementById('gridEmptyState');

        // Wire WebSocket events
        WS.on('tile_update', (data) => {
            const key = `${data.row}_${data.col}`;
            tiles[key] = data;

            // Track grid dimensions
            gridRows = Math.max(gridRows, data.row + 1);
            gridCols = Math.max(gridCols, data.col + 1);

            if (emptyState) emptyState.classList.add('hidden');
            requestRender();
        });

        WS.on('roi_overlay', (data) => {
            roi = data;
            requestRender();
        });

        WS.on('scan_progress', (data) => {
            document.getElementById('tileCount').textContent = `${data.completed} / ${data.total}`;
        });

        // Handle resize
        const resizeObserver = new ResizeObserver(() => {
            sizeCanvas();
            requestRender();
        });
        resizeObserver.observe(container);

        sizeCanvas();
        startAnimationLoop();
    }

    function sizeCanvas() {
        const rect = container.getBoundingClientRect();
        canvas.width = rect.width * devicePixelRatio;
        canvas.height = rect.height * devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }

    function startAnimationLoop() {
        function loop() {
            pulsePhase = (Date.now() % 1500) / 1500;
            render();
            animFrame = requestAnimationFrame(loop);
        }
        loop();
    }

    function requestRender() {
        // Render happens in animation loop
    }

    function render() {
        if (!ctx) return;

        const w = canvas.width / devicePixelRatio;
        const h = canvas.height / devicePixelRatio;

        ctx.clearRect(0, 0, w, h);

        if (gridRows === 0 || gridCols === 0) return;

        // Calculate tile size to fit in canvas with padding
        const pad = 20;
        const gap = 2;
        const tileW = Math.max(4, (w - pad * 2 - gap * (gridCols - 1)) / gridCols);
        const tileH = Math.max(4, (h - pad * 2 - gap * (gridRows - 1)) / gridRows);
        const tileSize = Math.min(tileW, tileH, 60);

        // Center the grid
        const gridW = gridCols * tileSize + (gridCols - 1) * gap;
        const gridH = gridRows * tileSize + (gridRows - 1) * gap;
        const offsetX = (w - gridW) / 2;
        const offsetY = (h - gridH) / 2;

        // Draw ROI bounding box (dashed)
        if (roi && roi.rois_px && roi.rois_px.length > 0) {
            ctx.save();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = 'rgba(188, 140, 255, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(offsetX - 4, offsetY - 4, gridW + 8, gridH + 8);
            ctx.restore();
        }

        // Draw tiles
        for (let row = 0; row < gridRows; row++) {
            for (let col = 0; col < gridCols; col++) {
                const key = `${row}_${col}`;
                const tile = tiles[key];
                const status = tile ? tile.status : 'pending';
                const colors = COLORS[status] || COLORS.pending;

                const x = offsetX + col * (tileSize + gap);
                const y = offsetY + row * (tileSize + gap);

                // ── SETTLING ANIMATION (MANDATORY) ──
                if (status === 'settling') {
                    const pulse = Math.sin(pulsePhase * Math.PI * 2) * 0.5 + 0.5;
                    const alpha = 0.15 + pulse * 0.35;
                    const borderAlpha = 0.4 + pulse * 0.6;

                    ctx.fillStyle = `rgba(210, 153, 34, ${alpha})`;
                    ctx.fillRect(x, y, tileSize, tileSize);

                    ctx.strokeStyle = `rgba(210, 153, 34, ${borderAlpha})`;
                    ctx.lineWidth = 2 + pulse;
                    ctx.strokeRect(x - 1, y - 1, tileSize + 2, tileSize + 2);

                    // Pulse ring
                    const ringSize = pulse * 6;
                    ctx.strokeStyle = `rgba(210, 153, 34, ${0.3 * (1 - pulse)})`;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x - ringSize, y - ringSize, tileSize + ringSize * 2, tileSize + ringSize * 2);
                } else {
                    // Standard tile rendering
                    ctx.fillStyle = colors.fill;
                    ctx.fillRect(x, y, tileSize, tileSize);

                    ctx.strokeStyle = colors.stroke;
                    ctx.lineWidth = status === 'targeting' || status === 'capturing' ? 2 : 1;
                    ctx.strokeRect(x, y, tileSize, tileSize);
                }

                // Failure labels
                if (colors.label && tileSize >= 16) {
                    ctx.fillStyle = colors.stroke;
                    ctx.font = `bold ${Math.min(10, tileSize * 0.4)}px ${getComputedStyle(document.body).getPropertyValue('--font-mono')}`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(colors.label, x + tileSize / 2, y + tileSize / 2);
                }
            }
        }
    }

    function reset() {
        tiles = {};
        gridRows = 0;
        gridCols = 0;
        roi = null;
        if (emptyState) emptyState.classList.remove('hidden');
        document.getElementById('tileCount').textContent = '0 / 0';
    }

    return { init, reset };
})();
