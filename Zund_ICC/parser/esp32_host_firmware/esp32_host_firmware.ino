#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>

// --- Configuration ---
const char* ssid = "Cutter-Link";
const char* password = "Qatar2026"; 
const int dns_port = 53;
const int ws_port = 81;

DNSServer dnsServer;
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(ws_port);

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
      
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
        // Send initial status
        webSocket.sendTXT(num, "{\"type\":\"serial\",\"data\":\"Connected to Cutter-Link Host\"}");
      }
      break;
      
    case WStype_TEXT:
      {
        String msg = "";
        for(size_t i = 0; i < length; i++) msg += (char)payload[i];
        
        // Simple JSON Parsing for "data" field
        // Expected: {"type":"gcode", "data":"G1 X..."}
        if(msg.indexOf("\"type\":\"gcode\"") != -1) {
            int dataStart = msg.indexOf("\"data\":\"");
            if(dataStart != -1) {
                dataStart += 8; // Skip "data":"
                int dataEnd = msg.lastIndexOf("\"");
                if(dataEnd > dataStart) {
                    String gcode = msg.substring(dataStart, dataEnd);
                    // Unescape if necessary (simple version)
                    gcode.replace("\n", "\n");
                    
                    Serial.println(gcode); // Send to Machine
                    
                    // Acknowledge to UI
                    webSocket.sendTXT(num, "{\"type\":\"ack\"}");
                }
            }
        }
      }
      break;
  }
}

void handleCaptivePortal() {
  server.sendHeader("Location", String("http://") + WiFi.softAPIP().toString(), true);
  server.send(302, "text/plain", "");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  pinMode(2, OUTPUT); // Onboard LED

  // Start WiFi
  WiFi.softAP(ssid, password);
  Serial.println("WiFi Started: " + String(ssid));
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());
  
  dnsServer.start(dns_port, "*", WiFi.softAPIP());
  
  if(!LittleFS.begin(true)) {
      Serial.println("LittleFS Mount Failed");
  }

  // Debug: List files
  Serial.println("--- Files ---");
  File root = LittleFS.open("/");
  File file = root.openNextFile();
  while(file){
      Serial.printf("  %s (%d bytes)\n", file.name(), file.size());
      file = root.openNextFile();
  }
  
  server.on("/", HTTP_GET, []() {
      File f = LittleFS.open("/index.html", "r");
      if(!f) return server.send(404, "text/plain", "Missing index.html");
      server.streamFile(f, "text/html");
      f.close();
  });
  
  server.serveStatic("/", LittleFS, "/");
  server.onNotFound(handleCaptivePortal);
  server.begin();
  
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  
  Serial.println("Cutter-Link Ready!");
}

void loop() {
  dnsServer.processNextRequest();
  server.handleClient();
  webSocket.loop();
  
  // Read from Machine (Serial) -> Send to UI (WebSocket)
  if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if(line.length() > 0) {
          // Wrap in JSON
          String json = "{\"type\":\"serial\",\"data\":\"" + line + "\"}";
          webSocket.broadcastTXT(json);
      }
  }
  
  // Heartbeat blink
  static unsigned long lastBlink = 0;
  if(millis() - lastBlink > 1000) {
      digitalWrite(2, !digitalRead(2));
      lastBlink = millis();
  }
}