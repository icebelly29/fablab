#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>

// --- Configuration ---
const char* ssid = "Cutter-Link";
const char* password = "Qatar2026"; // Minimum 8 characters for WPA2!
const int dns_port = 53;
const int ws_port = 81;

DNSServer dnsServer;
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(ws_port);

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if(type == WStype_TEXT) {
    String msg = "";
    for(size_t i = 0; i < length; i++) msg += (char)payload[i];
    Serial.println("Received: " + msg);
    webSocket.sendTXT(num, "{\"type\":\"ack\"}");
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
  
  // Heartbeat blink
  static unsigned long lastBlink = 0;
  if(millis() - lastBlink > 1000) {
      digitalWrite(2, !digitalRead(2));
      lastBlink = millis();
  }
}
