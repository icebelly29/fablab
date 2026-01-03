const { WebSocketServer } = require('ws');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');
const SvgConverter = require('./SvgConverter.cjs');

const WS_PORT = 8080;
const SERIAL_PORT_PATH = 'COM8'; // The user specified COM8

// --- Create WebSocket Server ---
const wss = new WebSocketServer({ port: WS_PORT });
console.log(`WebSocket server started on ws://localhost:${WS_PORT}`);

// --- Setup Serial Port ---
const port = new SerialPort({
  path: SERIAL_PORT_PATH,
  baudRate: 115200,
}, (err) => {
  if (err) {
    console.error(`Error opening serial port ${SERIAL_PORT_PATH}:`, err.message);
    console.log('Please ensure the port is correct and not in use by another application (like the Arduino IDE Serial Monitor).');
    process.exit(1);
  }
});

const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));
console.log(`Attempting to open serial port ${SERIAL_PORT_PATH}...`);


// --- WebSocket Server Logic ---
wss.on('connection', (ws) => {
  console.log('Frontend client connected via WebSocket.');

  // 1. When a message is received from the serial port, send it to the WebSocket client
  const serialListener = (data) => {
    const message = data.toString().trim();
    console.log(`From ESP32 -> Frontend: ${message}`);
    // Send raw 'ok' messages for acknowledgement
    if (message === 'ok') {
      ws.send(JSON.stringify({ type: 'ack' }));
    } else {
      ws.send(JSON.stringify({ type: 'serial', data: message }));
    }
  };
  parser.on('data', serialListener);

  // 2. When a message is received from the WebSocket client, process it
  ws.on('message', async (message) => {
    try {
      const msg = JSON.parse(message);

      if (msg.type === 'gcode') {
        const gcodeLine = msg.data;
        console.log(`From Frontend -> ESP32: ${gcodeLine}`);
        port.write(gcodeLine + '\n', (err) => {
          if (err) {
            console.error('Error writing to serial port:', err.message);
          }
        });
      } else if (msg.type === 'svg') {
        console.log('Received SVG from frontend for conversion.');
        
        try {
          // Use our custom SvgConverter
          const converter = new SvgConverter({
            feedRate: 1000,
            safeZ: 5,
            cutZ: 0,
            toolOn: 'M3 S1000',
            toolOff: 'M5'
          });
          
          const convertedGcode = converter.convert(msg.data);
          
          console.log('SVG converted successfully using custom SvgConverter.');
          ws.send(JSON.stringify({ type: 'gcode-from-svg', data: convertedGcode }));
        } catch (error) {
          console.error('Error converting SVG:', error);
          ws.send(JSON.stringify({ type: 'error', data: 'Error converting SVG: ' + error.message }));
        }
      }
    } catch (e) {
      console.error('Invalid message from client:', e);
    }
  });

  ws.on('close', () => {
    console.log('Frontend client disconnected.');
    // Clean up the serial port listener when the client disconnects
    parser.removeListener('data', serialListener);
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
});
