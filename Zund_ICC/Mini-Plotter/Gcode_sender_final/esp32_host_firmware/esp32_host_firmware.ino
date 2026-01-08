/**
 * @file esp32_host_firmware.ino
 * @brief ESP32 Firmware for Mini-Plotter Host 
 * 
 * This firmware transforms the ESP32 into a WiFi Access Point and Web Server.
 * It serves the "Cutter-Link" dashboard (HTML/JS) from LittleFS and relays
 * G-code commands from the connected WebSocket client to the Machine via Serial.
 * 
 * Key Features:
 * - WiFi Access Point (AP) Mode (Open or Protected)
 * - LittleFS for serving static web files
 * - WebSockets for real-time bi-directional communication
 * - Captive Portal handling for easy connectivity (Android/iOS)
 * - LED Status Indicators for boot and connection states
 */

#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>

// --- Configuration ---
/** @brief SSID for the Access Point */
const char* ssid = "Cutter-Link";
/** @brief Password for the Access Point (NULL for Open Network) */
const char* password = "Qatar2026"; 
/** @brief DNS Port for Captive Portal */
const int dns_port = 53;
/** @brief WebSocket Port */
const int ws_port = 81;

/** @brief DNS Server instance for Captive Portal */
DNSServer dnsServer;
/** @brief HTTP Web Server instance on port 80 */
WebServer server(80);
/** @brief WebSocket Server instance */
WebSocketsServer webSocket = WebSocketsServer(ws_port);

// LED Status: 2 is usually the onboard LED on ESP32 Dev Kits
#define LED_PIN 2 

/**
 * @brief WebSocket Event Handler
 * @details Processes connection events and incoming text messages (G-code).
 * 
 * @param num Client ID
 * @param type Event type (CONNECTED, DISCONNECTED, TEXT, etc.)
 * @param payload Data received
 * @param length Length of data
 */
void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
        webSocket.sendTXT(num, "{\"type\":\"serial\",\"data\":\"Connected to Cutter-Link Host\"}");
      }
      break;
    case WStype_TEXT:
      {
        String msg = "";
        for(size_t i = 0; i < length; i++) msg += (char)payload[i];
        if(msg.indexOf("\"type\":\"gcode\"") != -1) {
            int dataStart = msg.indexOf("\"data\":\"");
            if(dataStart != -1) {
                dataStart += 8; 
                int dataEnd = msg.lastIndexOf("\"");
                if(dataEnd > dataStart) {
                    String gcode = msg.substring(dataStart, dataEnd);
                    gcode.replace("\n", "\n");
                    Serial.println(gcode); // Send G-code to Machine via UART
                    webSocket.sendTXT(num, "{\"type\":\"ack\"}"); // Confirm receipt
                }
            }
        }
      }
      break;
  }
}

/**
 * @brief Captive Portal Redirect Handler
 * @details Redirects any unknown request to the ESP32's IP address.
 * Essential for "Sign in to network" popups on mobile devices.
 */
void handleCaptivePortal() {
  server.sendHeader("Location", String("http://") + WiFi.softAPIP().toString(), true);
  server.send(302, "text/plain", "");
}

/**
 * @brief Standard Arduino Setup
 * @details Initializes Serial, WiFi, LittleFS, and Servers.
 */
void setup() {
  // 1. Hardware Init
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Off initially

  Serial.begin(115200);
  
  // 2. Safety Delay (Give user time to open monitor)
  for(int i=0; i<3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(100); digitalWrite(LED_PIN, LOW); delay(900);
    Serial.print("."); 
  }
  Serial.println("\n--- BOOTING DIAGNOSTIC MODE ---");

  // 3. Start WiFi (Open Network, Channel 1)
  WiFi.disconnect(true); 
  WiFi.mode(WIFI_AP);
  
  Serial.println("Starting OPEN Access Point...");
  // passing NULL for password makes it an OPEN network
  if (WiFi.softAP(ssid, NULL, 1, 0, 4)) { 
    Serial.println(" [SUCCESS] OPEN WiFi Started!");
    Serial.println(" SSID: " + String(ssid));
    Serial.println(" (No Password Required)");
    Serial.print(" IP:   ");
    Serial.println(WiFi.softAPIP());
    digitalWrite(LED_PIN, HIGH); // LED On = WiFi Ready
  } else { 
    Serial.println(" [FAILED] WiFi did not start.");
    while(1) { // Trap error
        digitalWrite(LED_PIN, HIGH); delay(100); digitalWrite(LED_PIN, LOW); delay(100);
        Serial.println("WiFi Fail Loop");
    }
  }
  
  // Start DNS Server for Captive Portal (Redirect all domains to local IP)
  dnsServer.start(dns_port, "*", WiFi.softAPIP());
  
  if(!LittleFS.begin(true)) {
      Serial.println(" [WARN] LittleFS Mount Failed (Formatting...)");
  } else {
      Serial.println(" [OK] LittleFS Mounted");
  }

  // 4. Web Server Routes
  // Handle specific "noise" requests to stop VFS errors
  server.on("/favicon.ico", []() { server.send(404, "text/plain", ""); });
  server.on("/connecttest.txt", handleCaptivePortal); // Windows
  server.on("/generate_204", handleCaptivePortal);    // Android
  server.on("/gen_204", handleCaptivePortal);         // Android
  server.on("/canonical.html", handleCaptivePortal);  // Firefox
  server.on("/ncsi.txt", handleCaptivePortal);        // Windows

  // Serve the main index page
  server.on("/", HTTP_GET, []() {
      File f = LittleFS.open("/index.html", "r");
      if(!f) return server.send(404, "text/plain", "Missing index.html");
      server.streamFile(f, "text/html");
      f.close();
  });
  
  // Serve other static files (js, css) automatically from LittleFS
  server.serveStatic("/", LittleFS, "/");
  
  // Catch-all for Captive Portal
  server.onNotFound(handleCaptivePortal);
  server.begin();
  
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  
  Serial.println("--- SYSTEM READY ---");
}

/**
 * @brief Main Loop
 * @details Handles server client processing and Serial bridging.
 */
void loop() {
  dnsServer.processNextRequest();
  server.handleClient();
  webSocket.loop();
  
  // Read from Machine (Serial) -> Send to UI (WebSocket)
  // This allows the machine to send responses (e.g., "ok", position data) back to the browser.
  if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if(line.length() > 0) {
          String json = "{\"type\":\"serial\",\"data\":\"" + line + "\"}";
          webSocket.broadcastTXT(json);
      }
  }
}
