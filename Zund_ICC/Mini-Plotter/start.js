const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// 1. Start the HTTP Server for UI
const HTTP_PORT = 8000;
const BASE_DIR = './Gcode_sender_final';

const server = http.createServer((req, res) => {
    let urlPath = req.url === '/' ? '/index.html' : req.url;
    let filePath = path.join(BASE_DIR, urlPath);

    const extname = path.extname(filePath);
    let contentType = 'text/html';
    switch (extname) {
        case '.js': contentType = 'text/javascript'; break;
        case '.css': contentType = 'text/css'; break;
        case '.png': contentType = 'image/png'; break;
        case '.svg': contentType = 'image/svg+xml'; break;
    }

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if(error.code == 'ENOENT'){
                res.writeHead(404);
                res.end('404 File Not Found');
            } else {
                res.writeHead(500);
                res.end('500 Error: '+error.code);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(HTTP_PORT, () => {
    console.log(`UI Server running at http://localhost:${HTTP_PORT}/`);
});

// 2. Start the Mock Bridge
console.log('Starting Mock Bridge...');
// Try to resolve 'ws' from the subdirectory
const bridgePath = path.join(__dirname, 'mock_bridge.js');
const wsPath = path.join(__dirname, 'g-code-sender', 'node_modules', 'ws');

if (!fs.existsSync(wsPath)) {
    console.error('ERROR: "ws" module not found in g-code-sender/node_modules.');
    console.error('Please run "npm install" inside the "g-code-sender" folder first.');
    process.exit(1);
}

// We run the bridge as a child process, setting NODE_PATH so it finds 'ws'
const bridge = spawn('node', [bridgePath], {
    env: { ...process.env, NODE_PATH: path.join(__dirname, 'g-code-sender', 'node_modules') },
    stdio: 'inherit'
});

bridge.on('close', (code) => {
    console.log(`Bridge exited with code ${code}`);
    process.exit(code);
});

// Cleanup on exit
const cleanup = () => {
    console.log('Stopping Bridge...');
    bridge.kill();
    process.exit();
};

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
