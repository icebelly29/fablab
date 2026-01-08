const { WebSocketServer } = require('ws');

const PORT = 8080;
const wss = new WebSocketServer({ port: PORT });

console.log(`Mock Bridge running on ws://localhost:${PORT}`);
console.log('Waiting for connection...');

const broadcast = (msg) => {
    wss.clients.forEach(client => {
        if (client.readyState === 1) {
            client.send(JSON.stringify(msg));
        }
    });
};

wss.on('connection', (ws) => {
    console.log('Client connected!');
    // Broadcast new count
    broadcast({ type: 'userCount', count: wss.clients.size });

    ws.on('message', (message) => {
        try {
            const msg = JSON.parse(message);
            if (msg.type === 'gcode') {
                const cmd = msg.data;
                console.log(`[RX] ${cmd}`);
                
                // Simulate processing delay
                setTimeout(() => {
                    // Send "READY" to simulate the specific firmware behavior
                    ws.send(JSON.stringify({ type: 'serial', data: 'READY' }));
                }, 100); // 100ms simulated delay
            }
        } catch (e) {
            console.error('Error parsing JSON:', e);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        broadcast({ type: 'userCount', count: wss.clients.size });
    });
});
