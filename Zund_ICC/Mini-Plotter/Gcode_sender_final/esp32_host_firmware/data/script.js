import { log, clearConsole } from './Console.js';
import { updateStatus, setStartButtonState } from './UI.js';
import { MachineConnection } from './Connection.js';
import { setupTabs } from './Tabs.js';
import { renderGCode } from './Viewer.js';
import { handleFile } from './FileHandler.js';

/**
 * @file script.js
 * @description MAIN CONTROLLER
 * 
 * This is the "brain" of the application. It brings together all the separate
 * modules (UI, Connection, Files) to make the application work.
 * 
 * CORE LOGIC:
 * 1. It maintains the "State" of the application (is it sending? is it connected?).
 * 2. It handles the "Job Loop": 
 *    - User clicks Start -> Split G-code into lines -> Add to Queue.
 *    - Send Line 1 -> Wait for "Ack" from Machine -> Send Line 2...
 */

// --- Global State ---
// We keep all important variables in one place so it's easy to track what's happening.
const state = {
    gcodeQueue: [],      // Array holding the lines of G-code waiting to be sent
    isSending: false,    // Flag: Are we currently running a job?
    gcode: ''            // The full text of the loaded G-code file
};

// --- DOM Elements ---
// References to HTML elements we need to interact with
const editor = document.getElementById('gcodeEditor');
const cmdInput = document.getElementById('cmdInput');
const btnStart = document.getElementById('btnStart');
const dropZone = document.getElementById('dropZone');

// --- Connection Setup ---
// Initialize the WebSocket connection. We provide "callbacks" here.
// Callbacks are functions that run automatically when specific events happen.
const connection = new MachineConnection({
    // When the machine says "I received the command" (Ack), we send the next one.
    onAck: sendNextLine,
    
    // If the connection drops mid-job, we must stop everything for safety.
    onDisconnect: stopJob
});

// --- Job Control Logic ---

/**
 * START JOB
 * Called when the user clicks "Start Cutting".
 * It prepares the G-code and starts the sending loop.
 */
function startJob() {
    const code = editor.value; // Get code directly from the text area
    if (!code) {
        log('No G-Code to send.', 'error');
        return;
    }

    // 1. Prepare the Queue
    // We split the text by "newline" (\n) to get individual commands.
    // We also "trim" whitespace and remove empty lines.
    state.gcodeQueue = code.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    
    if (state.gcodeQueue.length === 0) return;

    log(`Starting Job: ${state.gcodeQueue.length} lines.`, 'success');
    
    // 2. Update State
    state.isSending = true;
    setStartButtonState(true); // Visual change (Turn button Red/Stop)

    // 3. Kickoff
    sendNextLine();
}

/**
 * STOP JOB
 * Called by user or on error. Clears the queue immediately.
 */
function stopJob() {
    if (!state.isSending) return; // Prevent duplicate logs if already stopped
    
    state.isSending = false;
    state.gcodeQueue = []; // Delete all remaining commands
    
    log('Job Stopped.', 'error');
    setStartButtonState(false); // Turn button back to Green/Start
}

/**
 * SEND NEXT LINE
 * The "Heartbeat" of the job.
 * 
 * Logic:
 * 1. Check if we are still supposed to be sending.
 * 2. If there are lines left in the queue:
 *    - Take the first one out (shift).
 *    - Send it to the machine.
 *    - Wait. (The 'onAck' callback will trigger this function again).
 * 3. If no lines left:
 *    - We are done!
 */
function sendNextLine() {
    if (!state.isSending) return;

    if (state.gcodeQueue.length > 0) {
        const line = state.gcodeQueue.shift(); // Remove first item from array
        connection.send(line);
        log(`> ${line}`, 'tx'); 
    } else {
        finishJob();
    }
}

/**
 * FINISH JOB
 * Clean up after the last command is sent.
 */
function finishJob() {
    state.isSending = false;
    log('Job Complete.', 'success');
    setStartButtonState(false);
}

// --- Event Listeners ---

// 1. Start/Stop Button Logic
btnStart.addEventListener('click', () => {
    if (state.isSending) {
        stopJob();
    } else {
        startJob();
    }
});

// 2. Manual Command Input (The text box at the bottom)
function handleManualSend() {
    const cmd = cmdInput.value.trim();
    if (!cmd) return;

    // Special command to clear the screen
    if (cmd.toLowerCase() === 'clear' || cmd.toLowerCase() === '/clear') {
        clearConsole();
        cmdInput.value = '';
        return;
    }

    connection.send(cmd, true); // true = Log this as a manual command
    cmdInput.value = '';
}

// Wire up the manual input buttons/keys
document.getElementById('btnClear').addEventListener('click', clearConsole);
document.getElementById('btnRun').addEventListener('click', handleManualSend);
cmdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleManualSend();
});

// 3. Reconnect on Badge Click
// If user clicks the "Disconnected" red badge, try to reconnect.
document.getElementById('statusBadge').addEventListener('click', () => connection.connect());

// 4. Sync Editor changes
// When user types in the editor, update our global variable so the preview knows.
editor.addEventListener('input', () => {
    state.gcode = editor.value;
});

// --- Initialization ---

// Setup the Tab clicking logic (Preview vs Editor)
setupTabs(() => state.gcode);

// Handle Window Resize
// If the window size changes, we need to redraw the canvas so it doesn't look stretched.
window.addEventListener('resize', () => {
    if (state.gcode && !document.getElementById('canvasContainer').classList.contains('hidden')) {
         requestAnimationFrame(() => renderGCode(state.gcode));
    }
});

// --- File Handling Setup ---

// Callback: What to do when a file is processed and ready?
const onGCodeReady = (newGCode) => {
    state.gcode = newGCode;
    editor.value = newGCode;
};

// Handle "Open File" button
document.getElementById('fileInput').addEventListener('change', (e) => {
    handleFile(e.target.files[0], onGCodeReady, window.switchTab);
});

// Handle Drag & Drop
// We need to prevent the default browser behavior (which is opening the file in the tab)
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

// Visual cue when dragging over
dropZone.addEventListener('dragover', () => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.add('drag-over'));
});

dropZone.addEventListener('dragleave', () => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.remove('drag-over'));
});

// Handle the Drop
dropZone.addEventListener('drop', (e) => {
    document.querySelectorAll('.empty-state').forEach(el => el.classList.remove('drag-over'));
    handleFile(e.dataTransfer.files[0], onGCodeReady, window.switchTab);
});

// Start the connection immediately when page loads
connection.connect();