#include <AccelStepper.h>
#include <Adafruit_NeoPixel.h>
#include <Wire.h>

#define STEP_PIN PA6
#define DIR_PIN PA7
#define ENABLE_PIN PA5
#define LIMIT_PIN PB1

#define LED_PIN PB0
#define NUM_LEDS 4

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

int closingLength = 18000;
long closeStartPos = 0;

int Received_value;
int globalSpeed = 8000;  // Default speed in steps/sec

enum Stages {
  FAST_HOMING,
  CLOSE,
  OPEN,
  STANDBY,
};

Stages stage = FAST_HOMING;

bool CheckLimit() {
  return !digitalRead(LIMIT_PIN);  // Active LOW
}

void setLEDBrightness(uint8_t baseR, uint8_t baseG, uint8_t baseB, int brightness) {
  uint8_t r = (baseR * brightness) / 255;
  uint8_t g = (baseG * brightness) / 255;
  uint8_t b = (baseB * brightness) / 255;

  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
}

void setupStepper(int maxSpeed, int acceleration, long target) {
  stepper.setMaxSpeed(maxSpeed);
  stepper.setAcceleration(acceleration);
  stepper.moveTo(target);
}

// Receive I2C data
char command[7];  // 6 chars max + null terminator
int cmdIndex = 0;

void receiveEvent(int howMany) {
  while (Wire.available() > 1 && cmdIndex < 6) {
    command[cmdIndex++] = Wire.read();
  }
  command[cmdIndex] = '\0';  // null-terminate
  int value = Wire.read();   // speed value (0–255)
  cmdIndex = 0;

  Serial.print("Command: ");
  Serial.print(command);
  Serial.print(" Value: ");
  Serial.println(value);

  // Set global speed
  globalSpeed = value * 100;  // scale: 0–255 → 0–25500

  if (strcmp(command, "open ") == 0) {
    Received_value = 100;
  } else if (strcmp(command, "close") == 0) {
    Received_value = 200;
  }
}

void setup() {
  strip.begin();
  strip.clear();
  strip.show();

  Wire.begin(12);  // Set this device's I2C address
  Wire.onReceive(receiveEvent);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);  // Enable motor driver

  pinMode(LIMIT_PIN, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop() {
  Function();
}

void Function() {
  static bool initialized = false;
  static int prevBrightness = -1;

  switch (stage) {
    case FAST_HOMING:
      setupStepper(4000, 50000, 200000);
      if (CheckLimit()) {
        stepper.stop();
        stepper.setCurrentPosition(0);
        Serial.println("Limit switch hit. Switching to CLOSE.");
        stage = CLOSE;
        initialized = false;
      }
      break;

    case CLOSE:
      if (!initialized) {
        digitalWrite(ENABLE_PIN, LOW);
        setupStepper(globalSpeed, 5000, -closingLength);
        closeStartPos = stepper.currentPosition();
        initialized = true;
      }

      if (stepper.distanceToGo() == 0) {
        Serial.println("Closing done.");
        delay(2000);
        stage = STANDBY;
        digitalWrite(ENABLE_PIN, HIGH);
        initialized = false;
        break;
      }

      {
        long progress = abs(stepper.currentPosition() - closeStartPos);
        long total = abs(stepper.targetPosition() - closeStartPos);
        int brightness = map(progress, 0, total, 255, 0);

        if (brightness != prevBrightness) {
          prevBrightness = brightness;
          setLEDBrightness(252, 113, 25, brightness);
        }
      }
      break;

    case OPEN:
      if (!initialized) {
        digitalWrite(ENABLE_PIN, LOW);
        setupStepper(globalSpeed, 10000, -1000);  // Start with full speed
        closeStartPos = stepper.currentPosition();
        initialized = true;
      }

      if (stepper.distanceToGo() == 0) {
        Serial.println("Opening done.");
        delay(2000);
        stage = STANDBY;
        digitalWrite(ENABLE_PIN, HIGH);
        initialized = false;
        break;
      }

      {
        long progress = abs(stepper.currentPosition() - closeStartPos);
        long total = abs(stepper.targetPosition() - closeStartPos);

        // Decrease speed gradually
        // Easing out: start fast, end slow
        float distanceRatio = (float)progress / total;
        float easing = pow(1.0 - distanceRatio, 2);               // Quadratic easing out
        float dynamicSpeed = globalSpeed * (0.3 + 0.7 * easing);  // Taper to 30% at end
        stepper.setMaxSpeed(dynamicSpeed);
        

        // LED brightness map (for effect)
        int brightness = map(progress, 0, total, 0, 255);
        if (brightness != prevBrightness) {
          prevBrightness = brightness;
          setLEDBrightness(252, 113, 25, brightness);
        }
      }
      break;

    case STANDBY:
      digitalWrite(ENABLE_PIN, HIGH);
      if (Received_value == 100) {
        stage = OPEN;
        Received_value = 0;
      } else if (Received_value == 200) {
        stage = CLOSE;
        Received_value = 0;
      } else {
        Received_value = 0;
      }
      break;
  }

  stepper.run();
}
