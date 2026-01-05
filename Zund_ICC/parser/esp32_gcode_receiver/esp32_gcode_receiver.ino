
void setup() {
  // Start serial communication at a standard baud rate.
  Serial.begin(115200);
  Serial.println("ESP32 G-code Receiver Ready");
}

void loop() {
  // Check if there's any data available to read from the serial port.
  if (Serial.available() > 0) {
    // Read the incoming line of G-code. The line should be terminated with a newline character (\n).
    String gcodeLine = Serial.readStringUntil('\n');

    // Trim any leading/trailing whitespace from the received line.
    gcodeLine.trim();

    if (gcodeLine.length() > 0) {
      // Print the received G-code line to the ESP32's serial monitor for debugging.
      Serial.print("Received: ");
      Serial.println(gcodeLine);

      // Here you could add your own logic to process the G-code command.
      // For example, control motors, lasers, etc.

      // Send an acknowledgement back to the PC to signal that the next line can be sent.
      Serial.print("ok\n");
    }
  }
}

