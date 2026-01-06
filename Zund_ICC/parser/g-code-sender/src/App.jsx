import { useState, useRef, useEffect } from "react";
import "./App.css";
import SvgConverter from "./utils/SvgConverter";
import GCodePreview from "./components/GCodePreview";

function App() {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [gcode, setGcode] = useState("G1 X10 Y10\nG1 Z5\nG1 X0 Y0"); // Default G-code for testing
  const [svgContent, setSvgContent] = useState(null);
  const [logs, setLogs] = useState([]);
  
  // New UI states
  const [activeTab, setActiveTab] = useState("gcode"); // 'gcode' | 'svg'
  const [commandInput, setCommandInput] = useState("");

  const socketRef = useRef(null);
  const gcodeQueueRef = useRef([]);
  const fileInputRef = useRef(null);
  const svgFileInputRef = useRef(null);
  const consoleEndRef = useRef(null);

  const addLog = (message, type = 'info') => {
      setLogs(prev => [...prev, { time: new Date().toLocaleTimeString(), message, type }]);
  };

  useEffect(() => {
    // Auto-scroll console
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    // Auto-connect on mount
    handleConnect();
    
    // Cleanup on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  const handleConnect = () => {
    if (isConnected || isConnecting) return;
    
    setIsConnecting(true);
    addLog("Connecting to WebSocket bridge...", 'system');

    // Determine WebSocket URL based on environment
    const hostname = window.location.hostname;
    const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
    const wsUrl = isLocal ? "ws://localhost:8080" : `ws://${hostname}:81`;
    
    addLog(`Attempting connection to: ${wsUrl}`, 'info');

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      setIsConnecting(false);
      setIsConnected(true);
      addLog(`Connected to ${isLocal ? 'Bridge' : 'Machine'}.`, 'success');
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
            addLog("SVG converted successfully and loaded.", 'success');
            break;
          case 'serial':
            console.log("Message from ESP32:", msg.data);
            addLog(`ESP32: ${msg.data}`, 'rx');
            break;
          case 'error':
            console.error("Error from bridge:", msg.data);
            addLog(`Error: ${msg.data}`, 'error');
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
      setIsConnecting(false);
      addLog("Disconnected from WebSocket bridge.", 'warning');
      socketRef.current = null;
    };

    socket.onerror = (error) => {
      setIsConnecting(false);
      setIsConnected(false);
      addLog("Error: Could not connect to WebSocket bridge.", 'error');
      console.error("WebSocket error:", error);
    };
  };

  const sendNextGcodeLine = () => {
    if (gcodeQueueRef.current.length > 0) {
      const line = gcodeQueueRef.current.shift();
      addLog(`Sending (${gcodeQueueRef.current.length} left): ${line}`, 'tx');
      socketRef.current.send(JSON.stringify({ type: 'gcode', data: line }));
    } else {
      addLog("G-code sending complete.", 'success');
    }
  };

  const handleSend = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      addLog("Error: Not connected.", 'error');
      return;
    }

    const gcodeLines = gcode
      .split("\n")
      .filter((line) => line.trim().length > 0);
    
    if (gcodeLines.length === 0) {
      addLog("Nothing to send.", 'warning');
      return;
    }

    gcodeQueueRef.current = gcodeLines;
    sendNextGcodeLine(); // Start the sending process
  };
  
  const handleManualCommand = () => {
    const cmd = commandInput.trim();
    if (!cmd) return;

    if (cmd.toLowerCase() === 'clear') {
        setLogs([]);
        setCommandInput("");
        return;
    }
    
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        addLog(`> ${cmd}`, 'tx'); // Echo command
        socketRef.current.send(JSON.stringify({ type: 'gcode', data: cmd }));
    } else {
        addLog("Error: Not connected.", 'error');
    }
    setCommandInput("");
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setGcode(e.target.result);
        setSvgContent(null);
        setActiveTab("gcode");
        addLog(`Loaded file: ${file.name}`, 'system');
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
        setActiveTab("svg");
        try {
          addLog(`Converting SVG: ${file.name}...`, 'system');
          const converter = new SvgConverter({
            feedRate: 1000,
            safeZ: 5,
            scale: 1.0 
          });
          const generatedGcode = converter.convert(content);
          setGcode(generatedGcode);
          addLog(`SVG converted successfully: ${file.name}`, 'success');
        } catch (error) {
          console.error("SVG Conversion failed:", error);
          addLog(`Error converting SVG: ${error.message}`, 'error');
        }
      };
      reader.readAsText(file);
    }
  };

  const triggerSvgFileInput = () => {
    svgFileInputRef.current.click();
  };

  // Status Pill Helpers
  const getStatusText = () => {
      if (isConnecting) return "Connecting...";
      if (isConnected) return "Connected";
      return "Disconnected";
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="header-title">Cutter (WiFi)</h1>
        <div 
            className={`status-pill ${isConnected ? 'connected' : 'disconnected'}`}
            onClick={!isConnected ? handleConnect : undefined}
            title={!isConnected ? "Click to connect" : "Machine Connected"}
            style={{ cursor: !isConnected ? 'pointer' : 'default' }}
        >
          <span className="status-dot"></span>
          {getStatusText()}
        </div>
      </header>

      <main className="main-layout">
        <div className="left-column">
          {/* G-Code Editor Panel */}
          <div className="panel editor-panel">
            <div className="panel-header">
              <h2>G-Code Editor</h2>
              <div className="toolbar">
                 <button className="btn-light" onClick={triggerFileInput}>Load</button>
                 <button className="btn-light" onClick={triggerSvgFileInput}>Import SVG</button>
              </div>
            </div>
            <div className="editor-content">
              <textarea
                className="code-editor"
                value={gcode}
                onChange={(e) => setGcode(e.target.value)}
                spellCheck="false"
              />
            </div>
            <div className="panel-footer-action">
                <button 
                  className="btn-dark" 
                  onClick={handleSend}
                  disabled={!isConnected}
                >
                  Start Cutting
                </button>
            </div>
          </div>

          {/* Console Panel */}
          <div className="panel console-panel">
             <div className="panel-header dark-header">
                <h2>Terminal</h2>
             </div>
             <div className="console-body">
                {logs.map((log, i) => (
                    <div key={i} className={`log-entry log-${log.type}`}>
                        <span className="log-time">[{log.time}]</span> {log.message}
                    </div>
                ))}
                <div ref={consoleEndRef} />
             </div>
             <div className="console-input-wrapper">
                <span className="prompt">&gt;</span>
                <input 
                  type="text" 
                  className="console-input"
                  value={commandInput} 
                  onChange={(e) => setCommandInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleManualCommand()}
                  placeholder="Enter G-Code command..."
                />
             </div>
             <div className="panel-footer-action dark-footer">
                <button className="btn-light" onClick={handleManualCommand}>Run</button>
             </div>
          </div>
        </div>

        <div className="right-column">
           {/* Preview Panel */}
           <div className="panel preview-panel">
              <div className="panel-header tabs-header">
                 <button 
                    className={`tab-btn ${activeTab === 'gcode' ? 'active' : ''}`}
                    onClick={() => setActiveTab('gcode')}
                 >
                    G-Code Preview
                 </button>
                 <button 
                    className={`tab-btn ${activeTab === 'svg' ? 'active' : ''}`}
                    onClick={() => setActiveTab('svg')}
                    disabled={!svgContent}
                    title={!svgContent ? "No SVG loaded" : ""}
                 >
                    SVG Preview
                 </button>
              </div>
              <div className="preview-content">
                 {activeTab === 'gcode' && (
                    <div className="gcode-preview-wrapper">
                       <GCodePreview gcode={gcode} />
                    </div>
                 )}
                 {activeTab === 'svg' && svgContent && (
                    <div className="svg-preview-wrapper">
                       <div dangerouslySetInnerHTML={{ __html: svgContent }} />
                    </div>
                 )}
                 {activeTab === 'svg' && !svgContent && (
                     <div className="empty-state">No SVG loaded</div>
                 )}
              </div>
           </div>
        </div>
      </main>

      {/* Hidden File Inputs */}
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
  );
}

export default App;
