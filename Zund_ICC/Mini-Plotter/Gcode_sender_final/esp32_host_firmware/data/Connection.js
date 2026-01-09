/**
 * @file Connection.js
 * @description NETWORK COMMUNICATIONS
 * 
 * This module handles talking to the ESP32.
 * We use "WebSockets" instead of standard HTTP requests because WebSockets allow
 * for two-way, real-time communication. The machine can talk back to us instantly!
 */

import { log } from './Console.js';
import { updateStatus, updateUserCount } from './UI.js';

export class MachineConnection {
    /**
     * @constructor
     * @param {Object} callbacks - Functions to run when events happen.
     *                             e.g. { onAck: function(){...} }
     */
    constructor(callbacks) {
        this.socket = null;      // Holds the active connection object
        this.connected = false;  // Simple flag to check status
        this.callbacks = callbacks || {}; 
    }

    /**
     * CONNECT
     * Initiates the connection to the machine.
     */
    connect() {
        if (this.connected) return;

        log('Connecting to Machine...', 'info');
        
        // --- Hostname Logic ---
        // 1. If we are running this file directly from the ESP32 (in AP mode), 
        //    window.location.hostname will be the IP of the ESP32 (e.g., 192.168.4.1).
        // 2. If we are testing on our laptop ('localhost'), we need to tell it 
        //    where the fake server is (localhost:8080).
        const hostname = window.location.hostname;
        const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
        
        // Port 81 is defined in the Arduino sketch for WebSockets.
        const wsUrl = isLocal ? "ws://localhost:8080" : `ws://${hostname}:81`;

        try {
            // Create the WebSocket object
            this.socket = new WebSocket(wsUrl);

            // --- Event: Connection Opened ---
            this.socket.onopen = () => {
                this.connected = true;
                updateStatus(true); // Turn badge Green
                log(`Connected to ${wsUrl}`, 'success');
            };

            // --- Event: Message Received ---
            this.socket.onmessage = (event) => {
                try {
                    // The machine sends data as text. We expect it to be JSON format.
                    // JSON Example: { "type": "serial", "data": "ok" }
                    const msg = JSON.parse(event.data);
                    this.handleMessage(msg);
                } catch (e) {
                    log(`Invalid JSON: ${event.data}`, 'error');
                }
            };

            // --- Event: Connection Lost ---
            this.socket.onclose = (event) => {
                this.connected = false;
                updateStatus(false); // Turn badge Red
                let reason = event.reason ? ` Reason: ${event.reason}` : '';
                log(`Disconnected (Code: ${event.code}).${reason}`, 'error');
                this.socket = null;
                
                // Tell the main script we disconnected (so it can stop the job)
                if (this.callbacks.onDisconnect) this.callbacks.onDisconnect();
            };

            // --- Event: Error ---
            this.socket.onerror = (err) => {
                console.error("WebSocket Error:", err);
                log(`Connection Error. Check console for details.`, 'error');
            };

        } catch (e) {
            log(`Connection failed: ${e.message}`, 'error');
        }
    }

    /**
     * HANDLE MESSAGE
     * Decides what to do based on the "type" of message received.
     * 
     * @param {Object} msg - The parsed message object.
     */
    handleMessage(msg) {
        switch (msg.type) {
            case 'ack':
                // "ack" = Acknowledge. The machine is saying "I finished the last command".
                // We should tell the main script to send the next one.
                if (this.callbacks.onAck) this.callbacks.onAck();
                break;
            
            case 'serial':
                // "serial" = Raw text from the CNC controller (GRBL/Marlin/etc).
                // We just log this to the screen so the user can see it.
                log(`ESP32: ${msg.data}`, 'info');
                
                // Fallback: Some machines send "READY" or "OK" text instead of a specific "ack" event.
                if (msg.data.toUpperCase().includes('READY')) {
                    if (this.callbacks.onAck) this.callbacks.onAck();
                }
                break;
            
            case 'error':
                log(`Error: ${msg.data}`, 'error');
                break;
            
            case 'userCount':
                // Updates the "Users Online" count in the header.
                updateUserCount(msg.count);
                break;
            
            default:
                console.log('Unknown msg:', msg);
        }
    }

    /**
     * SEND COMMAND
     * Sends a G-code string to the ESP32.
     * 
     * @param {string} cmd - The G-code command (e.g. "G1 X10 Y10").
     * @param {boolean} [isManual=false] - If true, we log it as a user-typed command.
     */
    send(cmd, isManual = false) {
        if (!this.connected || !this.socket) {
            log('Error: Not connected', 'error');
            return;
        }
        
        // Wrap the command in JSON so the ESP32 knows it's G-code.
        const payload = JSON.stringify({ type: 'gcode', data: cmd });
        this.socket.send(payload);
        
        if (isManual) log(`> ${cmd}`, 'tx');
    }
}