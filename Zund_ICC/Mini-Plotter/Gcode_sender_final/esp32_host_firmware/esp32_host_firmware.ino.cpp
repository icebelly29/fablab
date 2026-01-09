# 1 "C:\\Users\\nikhi\\AppData\\Local\\Temp\\tmpdc5writx"
#include <Arduino.h>
# 1 "C:/Users/nikhi/fablab/Zund_ICC/Mini-Plotter/Gcode_sender_final/esp32_host_firmware/esp32_host_firmware.ino"





#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>

const char* ssid = "Cutter-Link";
const char* password = "Qatar2026";
const int dns_port = 53;
const int ws_port = 81;

DNSServer dnsServer;
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(ws_port);

#define LED_PIN 2
void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);
void handleCaptivePortal();
void setup();
void loop();
#line 23 "C:/Users/nikhi/fablab/Zund_ICC/Mini-Plotter/Gcode_sender_final/esp32_host_firmware/esp32_host_firmware.ino"
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
                    Serial.println(gcode);
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
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- BOOTING ---");

  WiFi.disconnect(true);
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, NULL, 1, 0, 4);

  dnsServer.start(dns_port, "*", WiFi.softAPIP());
  LittleFS.begin(true);

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
  digitalWrite(LED_PIN, HIGH);
}

void loop() {
  dnsServer.processNextRequest();
  server.handleClient();
  webSocket.loop();

  if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if(line.length() > 0) {
          String json = "{\"type\":\"serial\",\"data\":\"" + line + "\"}";
          webSocket.broadcastTXT(json);
      }
  }
}