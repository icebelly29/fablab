import { useState, useRef } from "react";
import "./App.css";
import SvgConverter from "./utils/SvgConverter";
import GCodePreview from "./components/GCodePreview";

function App() {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [gcode, setGcode] = useState("G1 X10 Y10\nG1 Z5\nG1 X0 Y0"); // Default G-code for testing
  const [svgContent, setSvgContent] = useState(null);
  const [status, setStatus] = useState("Disconnected");

  const socketRef = useRef(null);
  const gcodeQueueRef = useRef([]);
  const fileInputRef = useRef(null);
  const svgFileInputRef = useRef(null);

  const handleConnect = () => {
    setIsConnecting(true);
    setStatus("Connecting to WebSocket bridge...");

    const socket = new WebSocket("ws://localhost:8080");

    socket.onopen = () => {
      setIsConnecting(false);
      setIsConnected(true);
      setStatus("Connected to WebSocket bridge.");
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
            setGcode(msg.data);
            setStatus("SVG converted successfully and loaded.");
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
        console.error("Invalid message from server:", event.data, e);
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      setStatus("Disconnected from WebSocket bridge.");
      socketRef.current = null;
    };

    socket.onerror = (error) => {
      setIsConnecting(false);
      setIsConnected(false);
      setStatus("Error: Could not connect to WebSocket bridge. Is it running?");
      console.error("WebSocket error:", error);
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
          <div className="logo-icon">✂️</div>
          <h1>Cutter</h1>
        </div>
        <div className="header-controls">
           <div className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              {isConnected ? "Machine Ready" : "Machine Offline"}
           </div>
           <button
            className="btn-connect"
            onClick={handleConnect}
            disabled={isConnected || isConnecting}
          >
            {isConnecting ? "Connecting..." : isConnected ? "Disconnect" : "🔌 Connect Machine"}
          </button>
        </div>
      </header>

      <div className="status-bar">
        <span className="status-label">SYSTEM STATUS:</span>
        <span className="status-message">{status}</span>
      </div>

      <main className="workspace">
        <div className="panel editor-panel">
          <div className="panel-header">
            <h2>G-Code Editor</h2>
            <div className="panel-toolbar">
               <button className="btn-secondary" onClick={triggerFileInput} title="Load .gcode file">
                 📂 Load G-Code
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
             <div className="panel-toolbar">
                {/* Future: Zoom/Pan controls */}
             </div>
           </div>
           <div className="canvas-container" style={{ display: 'block', overflowY: 'auto', padding: '20px' }}>
              {gcode.trim().length === 0 && !svgContent && (
                  <div className="canvas-instruction">
                      Load a file to see the preview here
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
          title={!isConnected ? "Please connect the machine first" : "Start cutting"}
        >
          {!isConnected ? "🔌 Connect to Start" : "🚀 START CUTTING"}
        </button>
      </footer>
    </div>
  );
}

export default App;