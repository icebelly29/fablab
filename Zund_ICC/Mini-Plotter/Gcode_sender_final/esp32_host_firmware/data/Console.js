/**
 * @file Console.js
 * @description ON-SCREEN LOGGER
 * 
 * Takes messages and displays them in the scrolling black box on the UI.
 * This is crucial for debugging and letting the user know what the machine is doing.
 */

const consoleOutput = document.getElementById('consoleOutput');

/**
 * Appends a message to the on-screen console log.
 * 
 * @param {string} msg - The message text.
 * @param {string} [type='info'] - The log type, which determines the color (CSS class).
 *                                 Options: 'info' (white), 'success' (green), 'error' (red), 'tx' (dim grey).
 */
export function log(msg, type = 'info') {
    const line = document.createElement('div');
    line.className = `console-line log-${type}`;
    
    // Add timestamp [10:30:05]
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    
    consoleOutput.appendChild(line);
    
    // Auto-scroll to the bottom so the newest message is visible
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

/**
 * Clears the console output.
 */
export function clearConsole() {
    consoleOutput.innerHTML = '';
    log('Console cleared.', 'info');
}