import SvgConverter from './SvgConverter.js';

/**
 * @file script.js
 * @description Frontend logic for the Mini-Plotter G-code Sender.
 * Handles WebSocket communication, file parsing (SVG/GCode), G-code visualization,
 * and UI interactions.
 */

// --- UI State ---
/**
 * @typedef {Object} AppState
 * @property {boolean} connected - WebSocket connection status.
 * @property {WebSocket|null} socket - The WebSocket instance.
 * @property {string[]} gcodeQueue - Array of G-code commands waiting to be sent.
 * @property {boolean} isSending - Whether a job is currently in progress.
 * @property {string} gcode - The current loaded G-code string.
 * @property {string} svgContent - The raw SVG content (if applicable).
 */
const state = {
    connected: false,
    socket: null,
    gcodeQueue: [],
    isSending: false,
    gcode: '',
    svgContent: ''
};

// --- DOM Elements ---
const consoleOutput = document.getElementById('consoleOutput');
const cmdInput = document.getElementById('cmdInput');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const statusDot = statusBadge.querySelector('.status-dot');
const editor = document.getElementById('gcodeEditor');
const btnStart = document.getElementById('btnStart');
const dropZone = document.getElementById('dropZone');

// --- Console Logic ---
/**
 * @function log
 * @description Appends a message to the on-screen console log.
 * @param {string} msg - The message text.
 * @param {string} [type='info'] - The log type ('info', 'success', 'error', 'tx').
 */
function log(msg, type = 'info') {
    const line = document.createElement('div');
    line.className = `console-line log-${type}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// --- WebSocket Logic ---
/**
 * @function connect
 * @description Initiates a WebSocket connection to the ESP32 host.
 * Automatically determines the URL based on the current hostname.
 */
function connect() {
    if (state.connected) return;

    log('Connecting to Machine...', 'info');
    
    const hostname = window.location.hostname;
    // Fallback for local development testing
    const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
    const wsUrl = isLocal ? "ws://localhost:8080" : `ws://${hostname}:81`;

    try {
        state.socket = new WebSocket(wsUrl);

        state.socket.onopen = () => {
            state.connected = true;
            updateStatus(true);
            log(`Connected to ${wsUrl}`, 'success');
        };

        state.socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleServerMessage(msg);
            } catch (e) {
                log(`Invalid JSON: ${event.data}`, 'error');
            }
        };

        state.socket.onclose = (event) => {
            state.connected = false;
            updateStatus(false);
            let reason = event.reason ? ` Reason: ${event.reason}` : '';
            log(`Disconnected (Code: ${event.code}).${reason}`, 'error');
            state.socket = null;
            if(state.isSending) stopJob();
        };

        state.socket.onerror = (err) => {
            console.error("WebSocket Error:", err);
            log(`Connection Error. Check console for details.`, 'error');
        };

    } catch (e) {
        log(`Connection failed: ${e.message}`, 'error');
    }
}

/**
 * @function updateStatus
 * @description Updates the UI connection badge.
 * @param {boolean} isConnected - True if connected, false otherwise.
 */
function updateStatus(isConnected) {
    if (isConnected) {
        statusText.textContent = "Connected";
        statusBadge.style.backgroundColor = "#a7f3d0";
        statusBadge.style.color = "#064e3b";
        statusDot.style.backgroundColor = "#10b981";
        btnStart.disabled = false;
    } else {
        statusText.textContent = "Disconnected";
        statusBadge.style.backgroundColor = "#fca5a5";
        statusBadge.style.color = "#7f1d1d";
        statusDot.style.backgroundColor = "#ef4444";
        btnStart.disabled = true;
    }
}

/**
 * @function handleServerMessage
 * @description Processes incoming JSON messages from the ESP32.
 * @param {Object} msg - The parsed JSON message object.
 */
function handleServerMessage(msg) {
    switch (msg.type) {
        case 'ack':
            // 'ack' means the machine is ready for the next command
            if (state.isSending) sendNextLine();
            break;
        case 'serial':
            log(`ESP32: ${msg.data}`, 'info');
            // Some firmwares send 'READY' string instead of explicit ACKs
            if (state.isSending && msg.data.toUpperCase().includes('READY')) {
                sendNextLine();
            }
            break;
        case 'error':
            log(`Error: ${msg.data}`, 'error');
            break;
        case 'userCount':
            const count = msg.count;
            const userText = count === 1 ? 'user' : 'users';
            document.getElementById('userCount').textContent = `(${count} ${userText})`;
            break;
        default:
            console.log('Unknown msg:', msg);
    }
}

/**
 * @function sendCommand
 * @description Sends a raw G-code string to the machine via WebSocket.
 * @param {string} cmd - The G-code command (e.g., "G1 X10 Y10").
 * @param {boolean} [isManual=false] - True if triggered by manual input (logs to UI).
 */
function sendCommand(cmd, isManual = false) {
    if (!state.connected || !state.socket) {
        log('Error: Not connected', 'error');
        return;
    }
    
    const payload = JSON.stringify({ type: 'gcode', data: cmd });
    state.socket.send(payload);
    
    if (isManual) log(`> ${cmd}`, 'tx');
}

// --- Sending Queue (Start/Stop) ---
/**
 * @function startJob
 * @description Initiates the streaming of the current G-code in the editor.
 */
function startJob() {
    const code = editor.value;
    if (!code) {
        log('No G-Code to send.', 'error');
        return;
    }

    state.gcodeQueue = code.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    
    if (state.gcodeQueue.length === 0) return;

    log(`Starting Job: ${state.gcodeQueue.length} lines.`, 'success');
    state.isSending = true;
    
    btnStart.textContent = "Stop Cutting";
    btnStart.classList.remove('btn-start');
    btnStart.classList.add('btn-stop');

    sendNextLine();
}

/**
 * @function stopJob
 * @description Aborts the current job and clears the queue.
 */
function stopJob() {
    state.isSending = false;
    state.gcodeQueue = []; // Clear queue
    log('Job Stopped by user or error.', 'error');
    
    btnStart.textContent = "Start Cutting";
    btnStart.classList.remove('btn-stop');
    btnStart.classList.add('btn-start');
}

/**
 * @function sendNextLine
 * @description Pops the next command from the queue and sends it.
 * Called automatically when an 'ack' is received.
 */
function sendNextLine() {
    if (!state.isSending) return;

    if (state.gcodeQueue.length > 0) {
        const line = state.gcodeQueue.shift();
        sendCommand(line);
        log(`> ${line}`, 'tx'); 
    } else {
        finishJob();
    }
}

/**
 * @function finishJob
 * @description cleans up state after the last command is sent.
 */
function finishJob() {
    state.isSending = false;
    log('Job Complete.', 'success');
    
    btnStart.textContent = "Start Cutting";
    btnStart.classList.remove('btn-stop');
    btnStart.classList.add('btn-start');
}

// --- Event Listeners ---
btnStart.addEventListener('click', () => {
    if (state.isSending) {
        stopJob();
    } else {
        startJob();
    }
});

// Manual Command
function handleManualSend() {
    const cmd = cmdInput.value.trim();
    if (!cmd) return;

    if (cmd.toLowerCase() === 'clear' || cmd.toLowerCase() === '/clear') {
        consoleOutput.innerHTML = '';
        log('Console cleared.', 'info');
        cmdInput.value = '';
        return;
    }

    sendCommand(cmd, true);
    cmdInput.value = '';
}

document.getElementById('btnClear').addEventListener('click', () => {
    consoleOutput.innerHTML = '';
    log('Console cleared.', 'info');
});

document.getElementById('btnRun').addEventListener('click', handleManualSend);
cmdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleManualSend();
});

// Reconnect on Badge Click
statusBadge.addEventListener('click', connect);
statusBadge.style.cursor = 'pointer';

// --- Tabs ---
window.switchTab = function(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const activeTab = Array.from(document.querySelectorAll('.tab')).find(t => t.innerText.toLowerCase().includes(tabName.split('-')[0]));
    if(activeTab) activeTab.classList.add('active');

    document.querySelectorAll('.view-panel').forEach(v => v.classList.add('hidden'));
    
    if (tabName === 'gcode-preview') {
        document.getElementById('gcodePreview').classList.remove('hidden');
        
        if (state.gcode && state.gcode.length > 0) {
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('canvasContainer').classList.remove('hidden');
            setTimeout(() => renderGCode(state.gcode), 10);
        } else {
            document.getElementById('emptyState').classList.remove('hidden');
            document.getElementById('canvasContainer').classList.add('hidden');
        }
    }
    
    if (tabName === 'svg-preview') {
        document.getElementById('svgPreview').classList.remove('hidden');
    }
    
    if (tabName === 'editor') {
        document.getElementById('gcodeEditor').classList.remove('hidden');
    }
}

// --- G-Code Renderer ---
/**
 * @function renderGCode
 * @description Parses G-code and draws a preview on the HTML5 Canvas.
 * @param {string} gcode - The G-code string to visualize.
 */
function renderGCode(gcode) {
    const canvas = document.getElementById('gcodeCanvas');
    const container = document.getElementById('canvasContainer');
    const ctx = canvas.getContext('2d');

    const bedW = 230;
    const bedH = 310;

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const lines = gcode.split('\n');
    const paths = [];
    let cur = { x: 0, y: 0 };

    lines.forEach(line => {
        line = line.split(';')[0].trim().toUpperCase();
        if (!line) return;

        const isMove = line.startsWith('G0') || line.startsWith('G1');
        if (isMove) {
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
            cur = next;
        }
    });

    const padding = 40;
    const availableW = canvas.width - padding * 2;
    const availableH = canvas.height - padding * 2;
    
    const scaleX = availableW / bedW;
    const scaleY = availableH / bedH;
    const scale = Math.min(scaleX, scaleY);

    const offsetX = (canvas.width - bedW * scale) / 2;
    const offsetY = (canvas.height - bedH * scale) / 2;

    const mapX = (x) => x * scale + offsetX;
    const mapY = (y) => canvas.height - (y * scale + offsetY); 

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.setLineDash([10, 5]);
    ctx.strokeStyle = '#cbd5e1'; 
    ctx.lineWidth = 1;
    
    const bedX_canvas = mapX(0);
    const bedY_canvas = mapY(bedH);
    
    ctx.strokeRect(bedX_canvas, bedY_canvas, bedW * scale, bedH * scale);
    
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px ui-monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`0,0 (BL)`, mapX(0), mapY(0) + 15);
    ctx.textAlign = 'right';
    ctx.fillText(`${bedW}x${bedH}mm`, mapX(bedW), mapY(bedH) - 5);

    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    paths.forEach(p => {
        ctx.beginPath();
        ctx.moveTo(mapX(p.from.x), mapY(p.from.y));
        ctx.lineTo(mapX(p.to.x), mapY(p.to.y));
        
        if (p.type === 'move') {
            ctx.strokeStyle = '#d1d5db'; // Light Gray for G0
            ctx.setLineDash([5, 5]);
        } else {
            ctx.strokeStyle = '#3b82f6'; // Blue for G1
            ctx.setLineDash([]);
        }
        ctx.stroke();
    });

    if (paths.length === 0) {
        ctx.fillStyle = '#9ca3af';
        ctx.textAlign = 'center';
        ctx.setLineDash([]);
        ctx.fillText("No paths found", canvas.width/2, canvas.height/2);
    }
}

// Handle Resize
window.addEventListener('resize', () => {
    if (state.gcode && !document.getElementById('canvasContainer').classList.contains('hidden')) {
         requestAnimationFrame(() => renderGCode(state.gcode));
    }
});

// --- File Handling & Drag/Drop ---
/**
 * @function handleFile
 * @description Process an uploaded file (SVG or GCode).
 * Automatically detects file type, parses/converts SVG, and loads it into the editor.
 * @param {File} file - The file object from Input or Drag/Drop.
 */
async function handleFile(file) {
    if (!file) return;
    log(`Loading ${file.name}...`, 'info');
    
    try {
        const text = await file.text();

        if (file.name.toLowerCase().endsWith('.svg')) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(text, 'image/svg+xml');
            const svg = doc.querySelector('svg');

            if (svg) {
                svg.style.width = '100%';
                svg.style.height = '100%';
                
                const svgPreview = document.getElementById('svgPreview');
                svgPreview.innerHTML = ''; // Clear empty state
                svgPreview.appendChild(svg);
            }

            const bedW = 230;
            const bedH = 310;
            let w_mm = 0, h_mm = 0;
            let viewbox = [0, 0, 0, 0];

            if(svg) {
                svg.style.width = '100%';
                svg.style.height = '100%';
                
                const wAttr = svg.getAttribute('width');
                const hAttr = svg.getAttribute('height');
                const vbAttr = svg.getAttribute('viewBox');

                if (vbAttr) {
                    viewbox = vbAttr.split(/[ ,]+/).map(parseFloat);
                }
                
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
                    return val; // Assume mm if no unit
                };

                w_mm = parseToMM(wAttr);
                h_mm = parseToMM(hAttr);

                if (w_mm === 0 && viewbox.length === 4) w_mm = viewbox[2];
                if (h_mm === 0 && viewbox.length === 4) h_mm = viewbox[3];
            }

            let vbW = viewbox.length === 4 ? viewbox[2] : w_mm;
            let vbH = viewbox.length === 4 ? viewbox[3] : h_mm;
            
            if (vbW === 0) vbW = w_mm;
            if (vbH === 0) vbH = h_mm;

            let scale = (vbW > 0) ? (w_mm / vbW) : 1.0;

            const margin = 10;
            
            let currentW = vbW * scale;
            let currentH = vbH * scale;

            if (currentW > (bedW - margin) || currentH > (bedH - margin)) {
                const scaleW = (bedW - margin) / currentW;
                const scaleH = (bedH - margin) / currentH;
                const fitScale = Math.min(scaleW, scaleH);
                scale *= fitScale;
                log(`Scaled down to fit bed (${(fitScale * 100).toFixed(0)}%)`, 'info');
            }

            const finalW = vbW * scale;
            const finalH = vbH * scale;

            const offsetX = (bedW - finalW) / 2;
            const offsetY = (bedH - finalH) / 2;
            const vbMinX = viewbox.length === 4 ? viewbox[0] : 0;
            const vbMinY = viewbox.length === 4 ? viewbox[1] : 0;

            const finalOffsetX = offsetX - (vbMinX * scale);
            const finalOffsetY = offsetY - (vbMinY * scale);

            try {
                const converter = new SvgConverter({
                    feedRate: 1000, 
                    scale: scale,
                    offsetX: finalOffsetX,
                    offsetY: finalOffsetY
                });
                state.gcode = converter.convert(text);
                editor.value = state.gcode;
                log(`Converted (Size: ${finalW.toFixed(1)}x${finalH.toFixed(1)}mm)`, 'success');
                switchTab('gcode-preview');
            } catch (err) {
                log(`Conversion Error: ${err.message}`, 'error');
            }

        } else {
            state.gcode = text;
            editor.value = text;
            log('G-Code loaded.', 'success');
            switchTab('gcode-preview');
        }
    } catch (err) {
        log(`File Read Error: ${err.message}`, 'error');
    }
}

// Input Change
document.getElementById('fileInput').addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag & Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

dropZone.addEventListener('dragover', () => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.add('drag-over'));
});

dropZone.addEventListener('dragleave', () => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.remove('drag-over'));
});

dropZone.addEventListener('drop', (e) => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.remove('drag-over'));
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFile(files[0]);
});

// Initial Connect
connect();
