/**
 * ============================================================================
 *  URUMICAM — LOG PANEL
 * ============================================================================
 * Scrolling monospace log window. Last 6 lines visible.
 * Color-coded: UART (blue), success (green), error (red), warning (amber).
 * ============================================================================
 */

const LogPanel = (() => {
    const MAX_LINES = 200;
    let logEl = null;
    let inputEl = null;
    let history = [];
    let historyIndex = -1;

    function init() {
        logEl = document.getElementById('logOutput');
        inputEl = document.getElementById('terminalInput');
        document.getElementById('btnClearLog').addEventListener('click', clear);

        // Wire WebSocket
        WS.on('log_message', (data) => {
            addLine(data.message, data.level, data.timestamp);
        });

        // Terminal input handling
        if (inputEl) {
            inputEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const cmd = inputEl.value.trim();
                    if (cmd) {
                        sendCommand(cmd);
                        inputEl.value = '';
                    }
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    navigateHistory(1);
                } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    navigateHistory(-1);
                }
            });
        }
    }

    function addLine(message, level = 'info', timestamp = null) {
        if (!logEl) return;

        const ts = timestamp || new Date().toLocaleTimeString('en-GB', { hour12: false });
        const line = document.createElement('div');
        line.className = `log-line log-${level}`;
        line.textContent = `[${ts}] ${message}`;
        logEl.appendChild(line);

        // Trim old lines
        while (logEl.children.length > MAX_LINES) {
            logEl.removeChild(logEl.firstChild);
        }

        // Auto-scroll to bottom
        logEl.scrollTop = logEl.scrollHeight;
    }

    function sendCommand(cmd) {
        // Add to log locally
        addLine(`> ${cmd}`, 'info');
        
        // Add to history
        history.unshift(cmd);
        if (history.length > 50) history.pop();
        historyIndex = -1;

        // Send to server
        WS.emit('terminal_command', { command: cmd });
    }

    function navigateHistory(direction) {
        if (history.length === 0) return;

        historyIndex += direction;
        if (historyIndex < -1) historyIndex = -1;
        if (historyIndex >= history.length) historyIndex = history.length - 1;

        if (historyIndex === -1) {
            inputEl.value = '';
        } else {
            inputEl.value = history[historyIndex];
        }
    }

    function clear() {
        if (logEl) logEl.innerHTML = '';
    }

    return { init, addLine, clear };
})();
