import { useState, useRef, useEffect } from "react";
import "./App.css";
import SvgConverter from "./utils/SvgConverter";
import GCodePreview from "./components/GCodePreview";

function AppESP32() {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [gcode, setGcode] = useState("G1 X10 Y10\nG1 Z5\nG1 X0 Y0"); // Default G-code for testing
  const [svgContent, setSvgContent] = useState(null);
  const [status, setStatus] = useState("Connecting to Machine...");

  const socketRef = useRef(null);
  const gcodeQueueRef = useRef([]);
  const fileInputRef = useRef(null);
  const svgFileInputRef = useRef(null);

  // Auto-connect on mount
  useEffect(() => {
    handleConnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleConnect = () => {
    if (isConnected || isConnecting) return;

    setIsConnecting(true);
    setStatus("Connecting to ESP32...");

    // Determine host: if localhost, use localhost:81 (dev simulation), else use window.location.hostname
    const host = window.location.hostname === 'localhost' ? 'localhost' : window.location.hostname;
    const socket = new WebSocket(`ws://${host}:81`);

    socket.onopen = () => {
      setIsConnecting(false);
      setIsConnected(true);
      setStatus("Connected to Machine (ESP32).");
      socketRef.current = socket;
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case 'ack':
            sendNextGcodeLine();
            break;
          case 'gcode-from-svg':
             // NOTE: The ESP32 version does conversion in the BROWSER, so we don't expect this from server
             // But if we kept logic symmetric, we might. 
             // In this version, we do conversion locally.
            break;
          case 'serial':
            console.log("Message from ESP32:", msg.data);
            setStatus(msg.data);
            break;
          case 'error':
            console.error("Error from bridge:", msg.data);
            setStatus(`Error: ${msg.data}`);
            break;
          default:
            console.log("Unknown message from server:", msg);
        }
      } catch (e) {
        // If it's not JSON, treat as raw string (simple ESP32 debug)
        console.log("Raw message:", event.data);
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      setIsConnecting(false); // Reset connecting state
      setStatus("Disconnected. Reconnecting in 3s...");
      socketRef.current = null;
      // Auto-reconnect
      setTimeout(handleConnect, 3000);
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      // Let onclose handle the state reset
    };
  };

  const sendNextGcodeLine = () => {
    if (gcodeQueueRef.current.length > 0) {
      const line = gcodeQueueRef.current.shift();
      setStatus(`Sending (${gcodeQueueRef.current.length} left): ${line}`);
      socketRef.current.send(JSON.stringify({ type: 'gcode', data: line }));
    } else {
      setStatus("G-code sending complete.");
    }
  };

  const handleSend = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      setStatus("Error: Not connected.");
      return;
    }

    const gcodeLines = gcode
      .split("\n")
      .filter((line) => line.trim().length > 0);
    
    if (gcodeLines.length === 0) {
      setStatus("Nothing to send.");
      return;
    }

    gcodeQueueRef.current = gcodeLines;
    sendNextGcodeLine(); // Start the sending process
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setGcode(e.target.result);
        setSvgContent(null);
        setStatus(`Loaded file: ${file.name}`);
      };
      reader.readAsText(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleSvgFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setSvgContent(content);
        try {
          setStatus(`Converting SVG: ${file.name}...`);
          // Client-side conversion!
          const converter = new SvgConverter({
            feedRate: 1000,
            safeZ: 5,
            scale: 1.0 
          });
          const generatedGcode = converter.convert(content);
          setGcode(generatedGcode);
          setStatus(`SVG converted successfully: ${file.name}`);
        } catch (error) {
          console.error("SVG Conversion failed:", error);
          setStatus(`Error converting SVG: ${error.message}`);
        }
      };
      reader.readAsText(file);
    }
  };

  const triggerSvgFileInput = () => {
    svgFileInputRef.current.click();
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">📡</div>
          <h1>Cutter (WiFi)</h1>
        </div>
        <div className="header-controls">
           <div className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              {isConnected ? "Connected" : "Reconnecting..."}
           </div>
        </div>
      </header>

      <div className="status-bar">
        <span className="status-label">STATUS:</span>
        <span className="status-message">{status}</span>
      </div>

      <main className="workspace">
        <div className="panel editor-panel">
          <div className="panel-header">
            <h2>G-Code Editor</h2>
            <div className="panel-toolbar">
               <button className="btn-secondary" onClick={triggerFileInput} title="Load .gcode file">
                 📂 Load
               </button>
               <button className="btn-secondary" onClick={triggerSvgFileInput} title="Import .svg file">
                 🎨 Import SVG
               </button>
            </div>
          </div>
          <div className="editor-container">
            <textarea
              className="code-editor"
              value={gcode}
              onChange={(e) => setGcode(e.target.value)}
              spellCheck="false"
            />
          </div>
          
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
            accept=".gcode,.gc,.nc"
          />
          <input
            type="file"
            ref={svgFileInputRef}
            onChange={handleSvgFileChange}
            style={{ display: "none" }}
            accept=".svg"
          />
        </div>

        <div className="panel preview-panel">
           <div className="panel-header">
            <h2>Visualizer</h2>
           </div>
           <div className="canvas-container" style={{ display: 'block', overflowY: 'auto', padding: '20px' }}>
              {gcode.trim().length === 0 && !svgContent && (
                  <div className="canvas-instruction">
                      Load a file to preview
                  </div>
              )}
              
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px', width: '100%' }}>
                <div className="gcode-preview-wrapper" style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                   <GCodePreview gcode={gcode} />
                </div>

                {svgContent && (
                  <div className="svg-preview-container" style={{ width: '100%', maxWidth: '600px' }}>
                     <h3 style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '10px' }}>Original SVG</h3>
                     <div 
                        className="svg-wrapper"
                        style={{ 
                          border: '1px solid var(--border-color)', 
                          padding: '10px', 
                          background: '#fff',
                          borderRadius: 'var(--radius-sm)',
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center'
                        }}
                        dangerouslySetInnerHTML={{ __html: svgContent }}
                     />
                  </div>
                )}
              </div>
           </div>
        </div>
      </main>

      <footer className="app-footer">
        <button 
          className="btn-primary btn-large" 
          onClick={handleSend} 
          disabled={!isConnected}
        >
          🚀 START CUTTING
        </button>
      </footer>
    </div>
  );
}

export default AppESP32;
