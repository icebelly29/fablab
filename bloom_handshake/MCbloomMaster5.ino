// OpenCloseSpeedMaster.ino
#include <Wire.h>

const int ledPin = LED_BUILTIN;

// Define slave addresses
#define NUM_SLAVES 5
int slaveAddresses[NUM_SLAVES] = {10, 11, 12, 13, 14};

// --- Define default speed values ---
const int OPEN_SPEED  = 30;   // Speed sent with "open"
const int CLOSE_SPEED = 200;  // Speed sent with "close"

int speedValue = OPEN_SPEED; // Current speed value

// --- Timer Variables ---
unsigned long openStartTime = 0;
bool isOpenCommandActive = false;
const unsigned long OPEN_DURATION = 20000;  // 20 seconds in milliseconds

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  Wire.begin();  // Master mode

  while (!Serial) {
    ;  // Wait for serial to connect
  }

  Serial.println("Master Ready. Type 'open', 'close', or 'speed <value>'");
}

void loop() {
  // --- Handle Serial Commands ---
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toLowerCase();  // Normalize input

    if (command == "open") {
      speedValue = OPEN_SPEED;
      sendCommandToAll("open ", speedValue);
      digitalWrite(ledPin, HIGH);
      Serial.print("Sent 'open ");
      Serial.print(speedValue);
      Serial.println("' to all slaves.");
      openStartTime = millis();
      isOpenCommandActive = true;
    } 
    else if (command == "close") {
      speedValue = CLOSE_SPEED;
      sendCommandToAll("close", speedValue);
      digitalWrite(ledPin, LOW);
      Serial.print("Sent 'close ");
      Serial.print(speedValue);
      Serial.println("' to all slaves.");
      isOpenCommandActive = false;
    } 
    else if (command.startsWith("speed ")) {
      int newSpeed = command.substring(6).toInt();
      if (newSpeed >= 0 && newSpeed <= 255) {
        speedValue = newSpeed;
        Serial.print("Speed updated to: ");
        Serial.println(speedValue);
      } else {
        Serial.println("Speed must be between 0 and 255.");
      }
    } 
    else {
      Serial.println("Unknown command. Use 'open', 'close', or 'speed <value>'");
    }
  }

  // --- Auto Close Logic after 15 seconds ---
  if (isOpenCommandActive && (millis() - openStartTime >= OPEN_DURATION)) {
    speedValue = CLOSE_SPEED;
    sendCommandToAll("close", speedValue);
    digitalWrite(ledPin, LOW);
    Serial.print("Auto Sent 'close ");
    Serial.print(speedValue);
    Serial.println("' to all slaves after 15 seconds.");
    isOpenCommandActive = false;
  }
}

void sendCommandToAll(const char* command, int value) {
  for (int i = 0; i < NUM_SLAVES; i++) {
    Wire.beginTransmission(slaveAddresses[i]);
    Wire.write(command);     // e.g., "open " or "close"
    Wire.write(value);       // speed value
    Wire.endTransmission();
    delay(5);
  }
}
