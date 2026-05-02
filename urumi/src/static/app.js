const canvas = document.getElementById('bedCanvas');
const ctx = canvas.getContext('2d');
const valX = document.getElementById('valX');
const valY = document.getElementById('valY');

// Canvas dimensions matches bed dimensions exactly (950x700)
// For higher resolution screens, we could multiply by a scale factor.
const SCALE = 1.0; 
const FOV_W = 40 * SCALE; // 40mm
const FOV_H = 30 * SCALE; // 30mm

// WebSocket connection
const ws = new WebSocket(`ws://${location.host}/ws`);

let currentX = 475;
let currentY = 350;
let path = [];

// Base image object for tiles
const latestTile = new Image();
let tileReady = false;

latestTile.onload = () => {
    tileReady = true;
    draw();
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'telemetry') {
        currentX = data.x;
        currentY = data.y;
        
        valX.innerText = currentX.toFixed(2);
        valY.innerText = currentY.toFixed(2);
        
        path.push({x: currentX, y: currentY});
        if (path.length > 500) path.shift(); // Limit path history
        
        latestTile.src = "data:image/jpeg;base64," + data.image;
    }
};

function drawGrid() {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let x = 0; x <= canvas.width; x += 50 * SCALE) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += 50 * SCALE) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }
}

function drawPath() {
    if (path.length < 2) return;
    ctx.strokeStyle = 'cyan';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(path[0].x * SCALE, path[0].y * SCALE);
    for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x * SCALE, path[i].y * SCALE);
    }
    ctx.stroke();
}

function drawCrosshair() {
    const cx = currentX * SCALE;
    const cy = currentY * SCALE;
    
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 20, cy); ctx.lineTo(cx + 20, cy);
    ctx.moveTo(cx, cy - 20); ctx.lineTo(cx, cy + 20);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, Math.PI * 2);
    ctx.stroke();
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();
    
    // Draw tile at correct kinematic position
    if (tileReady) {
        const cx = currentX * SCALE;
        const cy = currentY * SCALE;
        const tx = cx - (FOV_W / 2);
        const ty = cy - (FOV_H / 2);
        ctx.drawImage(latestTile, tx, ty, FOV_W, FOV_H);
    }
    
    drawPath();
    drawCrosshair();
}

// Initial draw
draw();
