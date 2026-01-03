#include <Wire.h>
#include "Adafruit_PWMServoDriver.h"

#include <EEPROM.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <DNSServer.h> // Include for DNS server functionality
#include <vector> // Required for std::vector

// --- WiFi Access Point (AP) Configuration ---
const char *ap_ssid = "ESP32_SERVO_AP";       // Name of the Wi-Fi network created by ESP32
const char *ap_password = "your_ap_password"; // Password for the Wi-Fi network (min 8 characters)

// --- Web Server, WebSocket Server, and DNS Server Instances ---
WebServer server(80);                                // Web server on port 80 (HTTP)
WebSocketsServer webSocket = WebSocketsServer(8765); // WebSocket server on port 8765
DNSServer dnsServer;                                 // DNS server for captive portal

// --- Servo Control Definitions ---
#define NO_OF_BOARD 7             // Total number of PCA9685 drivers
#define NO_OF_SERVOS_IN_A_BOARD 16 // Total number of servos per driver board
#define NO_OF_MODULE 108 // Total number of servos controlled by the system (now 108, last 4 pins of 7th board unused)

#define PWM_MIN 110 // Minimum PWM pulse length for servos (out of 4096)
#define PWM_MAX 600 // Maximum PWM pulse length for servos (out of 4096)

#define SERVO_MIN_ANGLE 0   // Standard minimum angle in degrees for frontend (0-180)
#define SERVO_MAX_ANGLE 180 // Standard maximum angle in degrees for frontend (0-180)

// --- Structs for Servo Configuration ---
// This struct stores the configuration that will be saved to EEPROM.
// It uses uint8_t for angles to save space, as 0-180 fits perfectly.
struct limits_eeprom {
    uint8_t lower_limit_angle[NO_OF_MODULE]; // Stored as angle (0-180)
    uint8_t upper_limit_angle[NO_OF_MODULE]; // Stored as angle (0-180)
    uint8_t board_idx[NO_OF_MODULE];         // PCA9685 board index
    uint8_t servo_pin_idx[NO_OF_MODULE];     // Pin on the PCA9685 board
};
limits_eeprom motor_eeprom_data; // Global instance for EEPROM data

// This struct stores the active servo configuration in RAM for faster access.
struct servo_config_ram {
    int lower_limit_pwm;  // Lower angle limit for this servo (PWM_MIN-PWM_MAX range)
    int upper_limit_pwm;  // Upper angle limit for this servo (PWM_MIN-PWM_MAX range)
    uint8_t board_idx;      // Index of the PCA9685 board (0 to NO_OF_BOARD-1)
    uint8_t servo_pin_idx;  // Servo pin on that board (0-15)
    int current_pwm_value; // Current PWM pulse value for the servo
};
servo_config_ram all_servos_ram[NO_OF_MODULE]; // Global array for all servos in RAM

// --- PCA9685 Driver Instances ---
Adafruit_PWMServoDriver pwm_drivers[NO_OF_BOARD];
// I2C addresses for PCA9685 boards (ensure these match your hardware setup)
byte pca9685_addresses[NO_OF_BOARD] = {0x40, 0x41, 0x42, 0x43,0x46, 0x47,0x45};

// --- EEPROM Versioning ---
const int EEPROM_VERSION_ADDRESS = 0;
const byte EEPROM_CURRENT_VERSION = 2; // Increment version if struct or data format changes
const int EEPROM_DATA_START_ADDRESS = 1; // Start address for actual data after version byte

// --- Interactive Setup Variables ---
bool interactive_setup_running = false;
int current_physical_servo_index = 0; // Tracks the physical servo being blinked (0 to NO_OF_BOARD*NO_OF_SERVOS_IN_A_BOARD - 1)
int interactive_setup_mean_position = 90; // Default mean position for setup blinking
int interactive_setup_test_range = 90;    // Default test range for setup blinking (e.g., 90 +/- 40)

// Global vector to store unresponsive board indices for frontend reporting
std::vector<uint8_t> unresponsive_boards;

// --- Function Prototypes ---
void bloom_init();
void load_all_servo_configs_from_eeprom();
void save_single_servo_config_to_eeprom(int numeric_servo_id, int board, int pin, int lower_limit, int upper_limit);
int angle_to_pwm_pulse(int angle);
int pwm_pulse_to_angle(int pwm_pulse);
int map_angle_to_constrained_pwm(int servo_idx, int angle);
int map_constrained_pwm_to_angle(int servo_idx, int pwm);
void set_servo_pwm_pulse(int servo_idx, int pulse); // Renamed from move_servo_to_angle
void update_all_servos_from_ram();
void start_interactive_setup();
void next_interactive_servo(); // Logic modified to be non-recursive
void stop_interactive_setup();
void i2c_scanner(); // New prototype for I2C scanner

// --- Embedded HTML/CSS/JavaScript Frontend ---
// IMPORTANT: Use PROGMEM for large HTML content to avoid RAM issues
const char *html_content PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Servo Motor Controller</title>
    <style>
        /* CSS Styling */
        :root {
            --body-padding: 20px;
            --grid-padding: 10px;
            --base-grid-unit: 55px;
        }

        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }

        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            background-color: #f0f0f0;
            padding: var(--body-padding);
            box-sizing: border-box;
            height: 100vh;
        }

        h1 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
            flex-shrink: 0;
            font-size: 1.8em; /* Reduced and consistent */
        }

        .motion-controls h2, .settings-controls h2 { /* Apply to both sidebars */
            margin-top: 0;
            margin-bottom: 15px;
            color: #555;
            text-align: center;
            font-size: 1.2em; /* Reduced and consistent */
        }

        .main-layout {
            display: flex;
            flex-grow: 1;
            width: 100%;
            max-width: 100%;
            justify-content: center; /* Center content horizontally */
            align-items: flex-start;
        }

        .motion-controls, .settings-controls { /* Styles common to both sidebars */
            flex-shrink: 0;
            width: 150px; /* Fixed width for the control panel */
            padding: 15px;
            background-color: #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            gap: 8px; /* Slightly reduced gap between sections (if multiple) */
            align-self: stretch;
            height: fit-content;
            max-height: 100%;
            overflow-y: auto;
        }

        .motion-controls {
            margin-right: 20px; /* Margin to the right of the left panel */
        }

        .settings-controls { /* Specific styles for the right panel */
            margin-left: 20px; /* Margin to the left of the right panel */
        }

        /* Styles for buttons within motion controls and settings controls */
        .motion-button {
            background-color: #6c5ce7;
            color: white;
            padding: 8px 15px; /* Reduced vertical padding */
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em; /* Slightly reduced font size */
            transition: background-color 0.2s ease, transform 0.1s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 100%; /* Make buttons full width of their container */
            box-sizing: border-box; /* Include padding and border in the element's total width and height */
        }

        .motion-button:hover {
            background-color: #5242d0;
            transform: translateY(-1px);
        }

        .motion-button:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        .motion-button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .content-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-grow: 1; /* Allows the content area to take up available space */
            overflow: hidden;
        }

        .grid-container {
            display: grid;
            gap: 0;
            padding: var(--grid-padding);
            background-color: #eee;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 100%;
            flex-grow: 1;
            overflow: auto;
            box-sizing: border-box;
            justify-content: center;
            align-content: center;
        }

        .servo-tile {
            width: calc(1.27 * var(--base-grid-unit));
            height: calc(1.27 * var(--base-grid-unit));
            border: 2px solid #555;
            display: flex;
            justify-content: center;
            align-items: center;
            font-weight: bold;
            color: #fff;
            cursor: pointer;
            transform: rotate(45deg);
            transition: background-color 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            position: relative;
            z-index: 1;
            justify-self: center;
            align-self: center;
        }

        .servo-tile:hover {
            filter: brightness(1.2);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            z-index: 2;
        }

        .servo-tile span {
            transform: rotate(-45deg);
            padding: 5px;
            border: none;
            display: block;
            text-align: center;
            white-space: nowrap;
            opacity: 0; /* Default hidden */
            transition: opacity 0.3s ease;
            font-size: calc(0.4 * var(--base-grid-unit));
        }

        .servo-tile:hover span {
            opacity: 1; /* Show on hover */
        }

        /* Highlight for testing */
        .servo-tile.highlighted {
            border: 4px solid #ff0000; /* Red border */
            box-shadow: 0 0 15px rgba(255, 0, 0, 0.7); /* Red glow */
        }

        /* Popup styles */
        .popup-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .popup-content {
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            text-align: center;
            width: 300px; /* Fixed width for the popup content */
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        .popup-content h2 {
            margin-top: 0;
            color: #333;
            margin-bottom: 20px;
        }

        .popup-content label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
            align-self: flex-start;
            width: 100%;
            text-align: left;
        }

        .popup-content input[type="number"] {
            width: calc(100% - 20px); /* This applies to single inputs like Lower/Upper/Move To Position */
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1em;
            box-sizing: border-box;
        }

        .input-group {
            display: flex;
            align-items: center;
            width: 100%;
            margin-bottom: 15px;
        }

        .input-group input[type="number"] {
            flex-grow: 1;
            margin-bottom: 0;
            margin-right: 10px;
            width: auto;
        }

        .send-button {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .send-button:hover {
            background-color: #0056b3;
        }
        .send-button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
        }

        /* Style for SVG icons within buttons */
        .send-button svg {
            fill: currentColor;
            width: 1em;
            height: 1em;
            vertical-align: middle;
        }

        .close-popup-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            font-size: 1.5em;
            color: #333;
            cursor: pointer;
            padding: 5px;
            line-height: 1;
            z-index: 1;
            transition: color 0.2s ease;
        }

        .close-popup-btn:hover {
            color: #f44336;
        }

        .popup-content .button-group {
            display: flex;
            justify-content: center;
            margin-top: 15px;
            width: 100%;
        }

        .popup-content .button-group button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin: 5px;
            transition: background-color 0.2s ease;
            width: auto;
        }

        .popup-content .button-group button:hover {
            background-color: #45a049;
        }
        .popup-content .button-group button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
        }

        .test-button {
            background-color: #ffc107;
            color: #333;
        }

        .test-button:hover {
            background-color: #e0a800;
        }

        #connectionStatus {
            margin-top: 20px;
            font-weight: bold;
            color: grey;
        }
        #connectionStatus.connected {
            color: green;
        }
        #connectionStatus.disconnected {
            color: red;
        }

        /* New styles for side-by-side module and pin inputs */
        .module-pin-group {
            display: flex;
            justify-content: space-between;
            width: 100%; /* This width refers to the inner content area of popup-content */
            margin-bottom: 15px;
            gap: 15px; /* Space between Board and Pin columns */
        }

        .module-pin-group > div {
            width: calc(50% - 7.5px); /* Half the width of the group minus half the gap */
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }

        .module-pin-group label {
            margin-bottom: 8px;
            width: auto;
            text-align: left;
        }

        /* This rule applies to inputs within module-pin-group, specifically the Board input */
        .module-pin-group input[type="number"] {
            width: 100%; /* Make input fill its parent div */
            box-sizing: border-box; /* Include padding and border in the element's total width */
            padding: 10px; /* Consistent padding */
            border: 1px solid #ccc; /* Consistent border */
            border-radius: 5px; /* Consistent border-radius */
            font-size: 1em; /* Consistent font size */
            margin-bottom: 0; /* No extra margin */
        }

        /* Specific styling for the Pin input and its button, ensuring it stays within bounds */
        .module-pin-input-group {
            display: flex;
            align-items: center;
            width: 100%;
            gap: 10px; /* Space between Pin input and its button */
        }

        .module-pin-input-group input[type="number"] {
            flex-grow: 1; /* Pin input takes remaining space */
            /* No margin-right here, gap handles it */
        }

        /* Styles for the new FPS control */
        .fps-control {
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #ccc;
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            align-items: center;
        }

        .fps-control label {
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
        }

        .fps-input-group {
            display: flex;
            align-items: center;
            width: 100%;
            max-width: 120px;
        }

        .fps-input-group input {
            flex-grow: 1;
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 0.85em;
            margin-right: 8px;
            text-align: center;
        }

        .fps-input-group button {
            background-color: #28a745;
            color: white;
            padding: 6px 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background-color 0.2s ease;
        }

        .fps-input-group button:hover {
            background-color: #218838;
        }
        .fps-input-group button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
        }

        /* New styles for Configure All popup */
        .configure-all-popup-content {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            text-align: center;
            width: 90%;
            max-width: 800px;
            height: 90%;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .config-table-container {
            flex-grow: 1;
            overflow-y: auto;
            margin-top: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            background-color: #f9f9f9;
        }

        .config-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }

        .config-table th, .config-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }

        .config-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .config-table input[type="number"] {
            width: 60px;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
            text-align: center;
        }

        /* Changed from update-config-button to test-config-button */
        .test-config-button {
            background-color: #ffc107;
            color: #333;
            padding: 6px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background-color 0.2s ease;
        }

        .test-config-button:hover {
            background-color: #e0a800;
        }

        /* Sections within sidebars - no bottom margin/border */
        .predefined-motions-section, .settings-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ccc;
            margin-bottom: 15px;
        }

        .motion-controls > *:last-child {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }

        .green-button {
            background-color: #4CAF50;
            color: white;
        }

        .green-button:hover {
            background-color: #45a049;
        }

        /* Custom Confirmation Dialog Styles */
        #confirmationDialog {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            justify-content: center;
            align-items: center;
            z-index: 2000;
        }

        #confirmationDialogContent {
            background-color: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            text-align: center;
            max-width: 350px;
            width: 90%;
            position: relative;
        }

        #confirmationDialogContent p {
            margin-bottom: 20px;
            font-size: 1.1em;
            color: #333;
        }

        #confirmationDialogContent .dialog-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
        }

        #confirmationDialogContent .dialog-buttons button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.2s ease;
        }

        #confirmYes {
            background-color: #28a745;
            color: white;
        }

        #confirmYes:hover {
            background-color: #218838;
        }

        #confirmNo {
            background-color: #dc3545;
            color: white;
        }

        #confirmNo:hover {
            background-color: #c82333;
        }

        /* New styles for Export/Import buttons */
        .config-actions {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }

        .config-actions button {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.2s ease;
        }

        .config-actions button:hover {
            background-color: #0056b3;
        }

        /* Message box for interactive setup */
        #interactiveSetupMessage, #testAllServosMessage {
            margin-top: 15px;
            padding: 10px;
            background-color: #e0f7fa;
            border: 1px solid #b2ebf2;
            border-radius: 5px;
            color: #006064;
            font-weight: bold;
            text-align: center;
            display: none;
            width: 90%;
            max-width: 400px;
        }
        /* New style for blink button within interactive setup message */
        #interactiveSetupMessage .interactive-setup-buttons button {
            margin-top: 10px;
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.2s ease;
            margin-left: 5px;
            margin-right: 5px;
        }

        #interactiveSetupMessage .interactive-setup-buttons button:hover {
            background-color: #0056b3;
        }
        #interactiveSetupMessage .interactive-setup-buttons button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
        }


        /* Styles for I2C Status Message */
        .status-message {
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            text-align: center;
            width: 90%;
            max-width: 400px;
            box-sizing: border-box;
        }
        .status-message.error-message {
            background-color: #ffe0e0;
            border: 1px solid #ffb2b2;
            color: #d32f2f;
        }
        .status-message.success-message {
            background-color: #e0f7fa;
            border: 1px solid #b2ebf2;
            color: #006064;
        }

        /* Icon button specific styles */
        .icon-button {
            background-color: #4CAF50;
            color: white;
            padding: 8px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1.2em;
            transition: background-color 0.2s ease, transform 0.1s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-grow: 1;
            box-sizing: border-box;
        }

        .icon-button:hover {
            background-color: #45a049;
            transform: translateY(-1px);
        }

        .icon-button:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        .icon-button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }
        .icon-button svg {
            fill: currentColor;
            width: 1em;
            height: 1em;
        }

        /* New style for mapped servos in interactive setup */
        .servo-tile.mapped {
            background-color: #8bc34a !important; /* A distinct green */
            border-color: #558b2f !important;
        }

        /* Media Queries for Responsiveness */
        @media (max-width: 768px) {
            h1 {
                font-size: 1.5em;
            }
            .motion-controls h2, .settings-controls h2 {
                font-size: 1.1em;
            }
            .main-layout {
                flex-direction: column;
                align-items: center;
            }
            .motion-controls {
                width: 90%;
                margin-right: 0;
                margin-bottom: 20px;
                padding: 10px;
            }
            .settings-controls {
                width: 90%;
                margin-left: 0;
                margin-bottom: 20px;
                padding: 10px;
            }
            .content-area {
                margin-bottom: 20px;
            }
            .motion-button {
                padding: 7px 12px;
                font-size: 0.85em;
            }
            .icon-button {
                padding: 6px;
                font-size: 1em;
            }
            .fps-control {
                margin-top: 10px;
                gap: 5px;
            }
            .fps-input-group {
                max-width: 100%;
            }
            .fps-input-group input, .fps-input-group button {
                padding: 5px 8px;
                font-size: 0.8em;
            }
            .popup-content {
                width: 95%;
                padding: 20px;
            }
            .popup-content input[type="number"], .input-group input[type="number"] {
                padding: 8px;
                font-size: 0.9em;
            }
            .send-button {
                padding: 8px 12px;
                font-size: 0.9em;
            }
            .config-table th, .config-table td {
                padding: 6px;
                font-size: 0.8em;
            }
            .config-table input[type="number"] {
                width: 50px;
                padding: 3px;
            }
            .config-actions button {
                padding: 8px 12px;
                font-size: 0.8em;
            }
        }

        @media (max-width: 480px) {
            h1 {
                font-size: 1.3em;
            }
            .motion-controls h2, .settings-controls h2 {
                font-size: 1em;
            }
            .motion-button {
                font-size: 0.8em;
                padding: 6px 10px;
            }
            .icon-button {
                font-size: 0.9em;
                padding: 5px;
            }
            .fps-input-group input, .fps-input-group button {
                font-size: 0.75em;
                padding: 4px 6px;
            }
            .popup-content {
                padding: 15px;
            }
            .popup-content input[type="number"], .input-group input[type="number"] {
                font-size: 0.8em;
                padding: 6px;
            }
            .send-button {
                font-size: 0.8em;
                padding: 6px 10px;
            }
        }
    </style>
</head>
<body>
    <h1>Interactive Servo Controller</h1>
    <div class="main-layout">
        <div class="motion-controls">
            <div class="predefined-motions-section">
                <h2>Movements</h2>
                <button id="waveMotionButton" class="motion-button">Wave</button>
                <button id="pulseMotionButton" class="motion-button">Pulse</button>
                <button id="sweepMotionButton" class="motion-button">Sweep</button>
                <button id="rippleMotionButton" class="motion-button">Ripple</button>
                <button id="paintModeButton" class="motion-button">Paint Mode</button> <!-- New Paint Mode Button -->
            </div>
        </div>
        <div class="content-area">
            <div id="interactiveSetupMessage">
                 <!-- Message content will be added here dynamically by JS -->
                 <div class="interactive-setup-buttons" style="margin-top: 10px; display: flex; justify-content: center;">
                    <!-- Back and Repeat buttons will be added here dynamically by JS -->
                 </div>
            </div>
            <div id="testAllServosMessage" class="status-message" style="display: none;"></div>
            <div id="i2cStatusMessage" class="status-message" style="display: none;"></div>
            <div class="grid-container" id="servoGrid">
            </div>
        </div>
        <div class="settings-controls">
            <div class="settings-section">
                <h2>Settings</h2>
                <button id="configureAllButton" class="motion-button green-button">Configure All</button>
                <button id="interactiveSetupButton" class="motion-button green-button">Interactive Setup</button>
                
                <!-- New Toggle Buttons -->
                <button id="toggleIdVisibilityButton" class="motion-button green-button">Toggle Module IDs</button>
                <button id="toggleHiddenModulesButton" class="motion-button green-button">Toggle Inactive Modules</button>

                <!-- New Test Controls Container -->
                <div id="testControlsContainer">
                    <button id="startTestButton" class="motion-button green-button">Start Test</button> <!-- Initial Start Test button -->

                    <div id="testControlIcons" style="display: none; width: 100%; gap: 5px;">
                        <button id="backwardButton" class="icon-button green-button">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512"><path d="M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l192 192c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L77.3 256 246.6 86.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-192 192z"/></svg>
                        </button>
                        <button id="playPauseButton" class="icon-button green-button">
                            <svg id="playIcon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"><path d="M73 39c-14.8-9.1-33.4-9.9-49.4-2.3S0 51.7 0 64V448c0 12.3 7.7 23.6 19.3 28.3s26.2 4.1 38.6-3.6l266-176c14.5-9.6 23.2-26.2 23.2-44.1s-8.7-34.5-23.2-44.1L73 39z"/></svg>
                            <svg id="pauseIcon" style="display:none;" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512"><path d="M48 64C21.5 64 0 85.5 0 112V400c0 26.5 21.5 48 48 48H80c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48H48zm192 0c-26.5 0-48 21.5-48 48V400c0 26.5 21.5 48 48 48h32c26.5 0 48-21.5 48-48V112c0-26.5-21.5-48-48-48H240z"/></svg>
                        </button>
                        <button id="forwardButton" class="icon-button green-button">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512"><path d="M310.6 233.4c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L242.7 256 73.4 86.6c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0l192 192z"/></svg>
                        </button>
                    </div>
                    <button id="exitTestButton" class="motion-button green-button" style="display: none; margin-top: 10px;">Exit Test</button>
                </div>
                
                <div class="fps-control">
                    <label for="fpsInput">Frames/Second:</label>
                    <div class="fps-input-group">
                        <input type="number" id="fpsInput" value="10" min="1" max="60">
                        <button id="setFpsButton">Set</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Single Servo Configuration Popup -->
    <div id="popup" class="popup-overlay">
        <div class="popup-content">
            <button class="close-popup-btn" id="closePopupButton">&times;</button>
            <h2>Configure Servo <span id="currentTileId"></span></h2>
            
            <!-- Board and Pin fields side-by-side -->
            <div class="module-pin-group">
                <div>
                    <label for="moduleNumber">Board:</label>
                    <input type="number" id="moduleNumber" value="0" min="0">
                </div>
                <div>
                    <label for="pinNumber">Pin:</label>
                    <div class="module-pin-input-group">
                        <input type="number" id="pinNumber" value="0" min="0">
                        <button id="sendBoardPinButton" class="send-button">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M438.6 105.4c12.5 12.5 12.5 32.8 0 45.3l-256 256c-12.5 12.5-32.8 12.5-45.3 0l-128-128c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0L160 338.7 393.4 105.4c12.5-12.5 32.8-12.5 45.3 0z"/></svg>
                        </button>
                    </div>
                </div>
            </div>

            <label for="lowerLimit">Lower Limit:</label>
            <div class="input-group">
                <input type="number" id="lowerLimit" value="0" min="0" max="180">
                <button id="sendLowerLimitButton" class="send-button">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M438.6 105.4c12.5 12.5 12.5 32.8 0 45.3l-256 256c-12.5 12.5-32.8 12.5-45.3 0l-128-128c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0L160 338.7 393.4 105.4c12.5-12.5 32.8-12.5 45.3 0z"/></svg>
                </button>
            </div>

            <label for="upperLimit">Upper Limit:</label>
            <div class="input-group">
                <input type="number" id="upperLimit" value="180" min="0" max="180">
                <button id="sendUpperLimitButton" class="send-button">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M438.6 105.4c12.5 12.5 12.5 32.8 0 45.3l-256 256c-12.5 12.5-32.8 12.5-45.3 0l-128-128c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0L160 338.7 393.4 105.4c12.5-12.5 32.8-12.5 45.3 0z"/></svg>
                </button>
            </div>

            <label for="moveToPosition">Move To Position:</label>
            <div class="input-group">
                <input type="number" id="moveToPosition" value="90" min="0" max="180">
                <button id="sendToPositionButton" class="send-button">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M498.1 5.6c10.1 7 15.4 19.1 13.5 31.2l-64 416c-1.5 9.7-7.4 18.2-16 23s-18.9 5.4-28 1.6L279.8 288.1 100 379.9c-11.7 5.3-25.7 1.7-33.8-8.7s-8.5-24.1-4.9-35.7L193.8 62.7c8.4-26.4 32.5-45.5 60.6-49.6L498.1 5.6zm-42.9 224.2L162.2 404.1 279.8 288.1 455.2 229.8z"/></svg>
                </button>
            </div>

            <div class="button-group">
                <button id="testButton" class="test-button">Test</button>
            </div>
            <div id="connectionStatus">Disconnected</div>
        </div>
    </div>

    <!-- Configure All Servos Popup -->
    <div id="configureAllPopup" class="popup-overlay">
        <div class="configure-all-popup-content">
            <button class="close-popup-btn" id="closeConfigureAllPopupButton">&times;</button>
            <h2>Configure All Servos</h2>
            <div class="config-table-container">
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Board</th>
                            <th>Pin</th>
                            <th>Lower Limit</th>
                            <th>Upper Limit</th>
                            <th>Test</th>
                        </tr>
                    </thead>
                    <tbody id="configTableBody">
                        <!-- Rows will be dynamically inserted here -->
                    </tbody>
                </table>
            </div>
            <div class="config-actions">
                <button id="exportCsvButton">Export CSV</button>
                <input type="file" id="importCsvInput" accept=".csv" style="display: none;">
                <button id="importCsvButton">Import CSV</button>
            </div>
        </div>
    </div>

    <!-- Custom Confirmation Dialog -->
    <div id="confirmationDialog" class="popup-overlay">
        <div id="confirmationDialogContent">
            <p id="confirmationMessage"></p>
            <div class="dialog-buttons">
                <button id="confirmYes">Yes</button>
                <button id="confirmNo">No</button>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const servoGrid = document.getElementById('servoGrid');
            const popup = document.getElementById('popup');
            const currentTileIdSpan = document.getElementById('currentTileId');
            const moduleNumberInput = document.getElementById('moduleNumber');
            const pinNumberInput = document.getElementById('pinNumber');
            const sendBoardPinButton = document.getElementById('sendBoardPinButton');

            const lowerLimitInput = document.getElementById('lowerLimit');
            const upperLimitInput = document.getElementById('upperLimit');
            const moveToPositionInput = document.getElementById('moveToPosition');
            const sendLowerLimitButton = document.getElementById('sendLowerLimitButton');
            const sendUpperLimitButton = document.getElementById('sendUpperLimitButton');
            const sendToPositionButton = document.getElementById('sendToPositionButton');
            const closePopupButton = document.getElementById('closePopupButton');
            const testButton = document.getElementById('testButton');
            
            const waveMotionButton = document.getElementById('waveMotionButton'); 
            const pulseMotionButton = document.getElementById('pulseMotionButton'); 
            const sweepMotionButton = document.getElementById('sweepMotionButton'); 
            const rippleMotionButton = document.getElementById('rippleMotionButton'); 
            const paintModeButton = document.getElementById('paintModeButton'); // New: Paint Mode Button

            const fpsInput = document.getElementById('fpsInput');
            const setFpsButton = document.getElementById('setFpsButton');

            const connectionStatusDiv = document.getElementById('connectionStatus');

            // Elements for Configure All popup
            const configureAllButton = document.getElementById('configureAllButton');
            const configureAllPopup = document.getElementById('configureAllPopup');
            const closeConfigureAllPopupButton = document.getElementById('closeConfigureAllPopupButton');
            const configTableBody = document.getElementById('configTableBody');
            const exportCsvButton = document.getElementById('exportCsvButton'); 
            const importCsvButton = document.getElementById('importCsvButton'); 
            const importCsvInput = document.getElementById('importCsvInput'); 

            // Elements for Custom Confirmation Dialog
            const confirmationDialog = document.getElementById('confirmationDialog');
            const confirmationMessage = document.getElementById('confirmationMessage');
            const confirmYesButton = document.getElementById('confirmYes');
            const confirmNoButton = document.getElementById('confirmNo');

            // New elements for Interactive Setup
            const interactiveSetupButton = document.getElementById('interactiveSetupButton');
            const interactiveSetupMessage = document.getElementById('interactiveSetupMessage');
            const interactiveSetupButtonsContainer = interactiveSetupMessage.querySelector('.interactive-setup-buttons');

            // New element for I2C Status Message
            const i2cStatusMessageDiv = document.getElementById('i2cStatusMessage');
            // New element for Test All Servos Message
            const testAllServosMessageDiv = document.getElementById('testAllServosMessage');

            // Test Control Buttons
            const startTestButton = document.getElementById('startTestButton'); 
            const testControlIcons = document.getElementById('testControlIcons'); 
            const backwardButton = document.getElementById('backwardButton');
            const playPauseButton = document.getElementById('playPauseButton');
            const forwardButton = document.getElementById('forwardButton');
            const exitTestButton = document.getElementById('exitTestButton');
            const playIcon = document.getElementById('playIcon');
            const pauseIcon = document.getElementById('pauseIcon');

            // New Toggle Buttons
            const toggleIdVisibilityButton = document.getElementById('toggleIdVisibilityButton');
            const toggleHiddenModulesButton = document.getElementById('toggleHiddenModulesButton');

            // New: Repeat button for interactive setup (renamed from blinkCurrentServoButton)
            const repeatBlinkServoButton = document.createElement('button');
            repeatBlinkServoButton.id = 'repeatBlinkServoButton';
            repeatBlinkServoButton.textContent = 'Repeat';
            
            // New: Back button for interactive setup
            const previousServoButton = document.createElement('button');
            previousServoButton.id = 'previousServoButton';
            previousServoButton.textContent = 'Back';

            // Append buttons to the new container
            interactiveSetupButtonsContainer.appendChild(previousServoButton);
            interactiveSetupButtonsContainer.appendChild(repeatBlinkServoButton);


            // Define the total number of servos to display in the table
            const TOTAL_BOARDS = 7; // Must match C++ NO_OF_BOARD
            const SERVOS_PER_BOARD = 16; // Must match C++ NO_OF_SERVOS_IN_A_BOARD
            const TOTAL_SERVOS_TO_DISPLAY = 108; // Now 108, as per user's request

            let activeTile = null;
            const servoSettings = {}; // Stores client-side settings, updated from server

            let servoIdCounter = 0;
            const servoIdToNumericIdMap = new Map();
            const numericIdToServoIdMap = new Map();
            const visualCoordToNumericIdMap = new Map(); // For grid-to-numeric mapping

            const totalOriginalRows = 6;
            const totalOriginalCols = 25; 

            // Deactivated modules for visual grid layout (these are *additional* to the 108 limit)
            const deactivatedModules = new Set([
                '[1,2]', '[1,3]', '[1,7]', '[1,11]', '[1,15]', '[1,19]', '[1,23]', '[1,24]',
                '[3,2]', '[3,24]',
                '[5,2]', '[5,24]',
                '[6,2]', '[6,3]', '[6,4]', '[6,6]', '[6,7]', '[6,8]', '[6,10]', '[6,11]', '[6,12]', '[6,14]', '[6,15]', '[6,16]', '[6,18]', '[6,19]', '[6,20]', '[6,22]', '[6,23]', '[6,24]'
            ]);

            // Fix: Use a fallback for window.location.hostname
            const WS_SERVER_ADDRESS = `ws://${window.location.hostname || 'localhost'}:8765`;
            let websocket;

            let animationFrameId = null;
            let animationIntervalId = null;
            const animationDuration = 3000; 
            let currentFPS = parseInt(fpsInput.value);
            let streamFrequency = 1000 / currentFPS;

            let currentAnimationFunction = null; 

            // Global state flags for different modes
            let isInteractiveSetupMode = false;
            let isTestAllServosMode = false;
            let isPaintMode = false; // New: Paint Mode State

            // New Toggle States
            let showModuleIds = true; // Default to showing IDs
            let showHiddenModules = false; // Default to hiding inactive modules

            let currentBlinkingPhysicalServo = { board: -1, pin: -1 }; 
            let nextServoToMapIndex = 0; 

            // Test All Servos Variables
            let isTestPaused = false; 
            let resolvePausePromise = null; 
            let currentTestingServoTileId = null; 
            const TEST_SERVO_INTERVAL_MS = 500; 
            let activeNumericIds = []; 
            let currentTestServoIndex = 0; 
            let testStepPromiseResolve = null; 
            let testStepPromiseReject = null; 
            
            function connectWebSocket() {
                if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
                    console.log("WebSocket is already connected or connecting.");
                    return;
                }

                websocket = new WebSocket(WS_SERVER_ADDRESS);
                websocket.binaryType = 'arraybuffer';

                websocket.onopen = () => {
                    console.log('WebSocket Connected!');
                    connectionStatusDiv.textContent = 'Connected';
                    connectionStatusDiv.classList.remove('disconnected');
                    connectionStatusDiv.classList.add('connected');
                };

                websocket.onmessage = (event) => {
                    console.log('Message from server:', event.data);
                    try {
                        const response = JSON.parse(event.data);
                        if (response.type === "data_fetched" && response.numericServoId !== undefined) {
                            const fetchedNumericId = response.numericServoId;
                            const correspondingTileId = numericIdToServoIdMap.get(fetchedNumericId);

                            if (correspondingTileId) {
                                moduleNumberInput.value = response.boardNumber !== undefined ? response.boardNumber : '';
                                pinNumberInput.value = response.pinNumber !== undefined ? response.pinNumber : '';
                                servoSettings[correspondingTileId].boardNumber = response.boardNumber;
                                servoSettings[correspondingTileId].pinNumber = response.pinNumber;

                                lowerLimitInput.value = response.lowerLimit;
                                upperLimitInput.value = response.upperLimit;
                                servoSettings[correspondingTileId].lowerLimit = response.lowerLimit;
                                servoSettings[correspondingTileId].upperLimit = response.upperLimit;
                                if (!animationFrameId && !isPaintMode) { // Only update if not in animation or paint mode
                                    updateTileColor(correspondingTileId);
                                }
                                console.log(`Fetched limits for servo ID ${correspondingTileId} (numeric: ${fetchedNumericId}): Lower=${response.lowerLimit}, Upper=${response.upperLimit}, Board=${response.boardNumber}, Pin=${response.pinNumber}`);
                            }
                        } else if (response.type === "all_data_fetched") {
                            populateConfigTable(response.servos);
                            console.log("All servo data fetched for configuration table.");
                        } else if (response.type === "received") {
                            console.log("Server acknowledged sent data.");
                        } else if (response.type === "blinkingServoInfo") {
                            currentBlinkingPhysicalServo.board = response.board;
                            currentBlinkingPhysicalServo.pin = response.pin;
                            interactiveSetupMessage.innerHTML = `Physical Servo (Board: ${response.board}, Pin: ${response.pin}) is now blinking. Click its corresponding tile on the grid.`;
                            interactiveSetupButtonsContainer.style.display = 'flex'; // Show the button container
                            interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer); // Re-add the container
                            
                            // Enable buttons once a servo is blinking
                            repeatBlinkServoButton.disabled = false;
                            previousServoButton.disabled = false;

                            console.log(`Received blinking servo info: Board ${response.board}, Pin ${response.pin}`);
                        } else if (response.type === "interactiveSetupComplete") {
                            stopInteractiveSetup(); 
                            interactiveSetupMessage.textContent = "Interactive setup complete! All servos mapped.";
                            interactiveSetupMessage.style.display = 'block';
                            console.log("Interactive setup completed on ESP32.");
                        } else if (response.type === "i2cStatus") {
                            const unresponsiveBoards = response.unresponsiveBoards;
                            if (unresponsiveBoards && unresponsiveBoards.length > 0) {
                                let message = "Warning: Could not find PCA9685 boards at indices: ";
                                unresponsiveBoards.forEach((boardIdx, index) => {
                                    message += `Board ${boardIdx}`;
                                    if (index < unresponsiveBoards.length - 1) {
                                        message += ", ";
                                    }
                                });
                                message += ". Please check wiring and power.";
                                i2cStatusMessageDiv.textContent = message;
                                i2cStatusMessageDiv.classList.add('error-message');
                                i2cStatusMessageDiv.style.display = 'block';
                                console.warn("I2C Warning:", message);
                            } else {
                                i2cStatusMessageDiv.textContent = "All PCA9685 boards initialized successfully.";
                                i2cStatusMessageDiv.classList.remove('error-message');
                                i2cStatusMessageDiv.classList.add('success-message');
                                i2cStatusMessageDiv.style.display = 'block';
                                setTimeout(() => {
                                    i2cStatusMessageDiv.style.display = 'none';
                                    i2cStatusMessageDiv.classList.remove('success-message');
                                }, 5000);
                                console.log("All I2C boards found.");
                            }
                        }
                    } catch (e) {
                        console.error("Failed to parse WebSocket message as JSON:", e, event.data);
                    }
                };

                websocket.onclose = () => {
                    console.log('WebSocket Disconnected. Attempting to reconnect in 3 seconds...');
                    connectionStatusDiv.textContent = 'Disconnected';
                    connectionStatusDiv.classList.remove('connected');
                    connectionStatusDiv.classList.add('disconnected');
                    setTimeout(connectWebSocket, 3000);
                };

                websocket.onerror = (error) => {
                    console.error('WebSocket connection error (likely server not reachable or network issue):', error);
                    connectionStatusDiv.textContent = 'Error! Disconnected';
                    connectionStatusDiv.classList.remove('connected');
                    connectionStatusDiv.classList.add('disconnected');
                };
            }

            function sendDataViaWebSocket(type, value, numericId = null) { 
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    if (type === 'streamPositions') {
                        websocket.send(value); 
                    } else {
                        const data = { type: type, value: value };
                        if (numericId !== null) { 
                            data.numericServoId = numericId;
                        }
                        websocket.send(JSON.stringify(data));
                        console.log(`Sent JSON via WebSocket: ${JSON.stringify(data)}`);
                    }
                } else {
                    console.warn('WebSocket not connected. Data not sent:', { type, value, numericId });
                    connectionStatusDiv.textContent = 'Not Connected! (Data not sent)';
                    connectionStatusDiv.classList.remove('connected');
                    connectionStatusDiv.classList.add('disconnected');
                }
            }

            function getServoColor(position) {
                position = Math.max(0, Math.min(180, position));
                const factor = position / 180;
                const r = Math.round(0 + (255 - 0) * factor);
                const g = Math.round(0 + (255 - 0) * factor);
                const b = 255; 
                return `rgb(${r},${g},${b})`;
            }

            function updateTileColor(tileId) {
                const tileElement = document.querySelector(`.servo-tile[data-tile-id="${tileId}"]`);
                if (tileElement) {
                    const isActive = servoSettings[tileId].isActive;
                    const spanElement = tileElement.querySelector('span');

                    // Handle visibility of the tile itself based on isActive and showHiddenModules
                    if (!isActive && !showHiddenModules) {
                        tileElement.style.display = 'none';
                    } else {
                        tileElement.style.display = 'flex'; // Always show if active or if showHiddenModules is true
                        
                        // Set background and border color
                        if (isInteractiveSetupMode && tileElement.classList.contains('mapped')) {
                             tileElement.style.backgroundColor = ''; // Reset to default for 'mapped' class to take over
                             tileElement.style.borderColor = ''; // Reset to default for 'mapped' class to take over
                        } else {
                            tileElement.style.backgroundColor = isActive ? getServoColor(servoSettings[tileId].moveToPosition) : '#e0e0e0'; // Grey for inactive
                            tileElement.style.borderColor = isActive ? '#555' : '#b0b0b0'; // Lighter border for inactive
                        }

                        if (spanElement) {
                            // Set text content for ID, but only if tile is active or if inactive modules are shown
                            if (isActive || showHiddenModules) {
                                spanElement.textContent = servoSettings[tileId].numericId !== null ? servoSettings[tileId].numericId : '';
                            } else {
                                spanElement.textContent = ''; // Clear text content if tile is hidden
                            }

                            // Set text color based on brightness for active tiles, or default for inactive
                            if (isActive && !tileElement.classList.contains('mapped')) { // Don't override mapped color
                                const color = getServoColor(servoSettings[tileId].moveToPosition);
                                const rgb = color.match(/\d+/g);
                                const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                                spanElement.style.color = (brightness > 125) ? '#222' : '#fff';
                            } else if (tileElement.classList.contains('mapped')) {
                                spanElement.style.color = '#fff'; // White text for mapped green background
                            }
                            else {
                                spanElement.style.color = '#555'; // Default text color for inactive tiles
                            }

                            // Control opacity of the ID text based on showModuleIds and tile visibility
                            if (showModuleIds && (isActive || showHiddenModules)) {
                                spanElement.style.opacity = 1; // Force visibility
                            } else {
                                spanElement.style.opacity = ''; // Revert to CSS default (0, with 1 on hover)
                            }
                        }
                    }
                } else {
                    console.warn(`updateTileColor: Tile element not found for ${tileId}`); 
                }
            }

            function generateGrid() {
                servoGrid.innerHTML = '';
                servoIdCounter = 0; 
                servoIdToNumericIdMap.clear();
                numericIdToServoIdMap.clear();
                visualCoordToNumericIdMap.clear(); // Clear for re-population

                let currentVisualRow = 1;
                let maxColumnIndexUsed = 0; 
                let maxRowIndexUsed = 0;    

                for (let originalRow = 1; originalRow <= totalOriginalRows; originalRow++) {
                    for (let originalCol = 1; originalCol <= totalOriginalCols; originalCol++) {
                        const tileId = `[${originalRow},${originalCol}]`; 
                        initializeServoSettings(tileId, originalCol); 

                        if (originalCol % 2 !== 0) {
                            const tileDiv = document.createElement('div'); 
                            tileDiv.classList.add('servo-tile');
                            tileDiv.dataset.tileId = tileId;
                            const numericId = servoSettings[tileId].numericId; 
                            tileDiv.dataset.numericId = numericId !== null ? numericId : ''; 

                            const span = document.createElement('span');
                            // Text content and opacity will be set by updateTileColor
                            tileDiv.appendChild(span);

                            tileDiv.addEventListener('click', () => {
                                if (isInteractiveSetupMode) {
                                    handleInteractiveTileClick(tileId);
                                } else if (isPaintMode) { // New condition for Paint Mode
                                    handlePaintModeClick(tileId);
                                }
                                else {
                                    openPopup(tileId);
                                }
                            });
                            
                            const gridRowStart = (currentVisualRow - 1) * 2 + 1;
                            const gridColumnStart = (originalCol - 1) + 1;

                            tileDiv.style.gridRowStart = gridRowStart;
                            tileDiv.style.gridRowEnd = `span 2`;
                            tileDiv.style.gridColumnStart = gridColumnStart;
                            tileDiv.style.gridColumnEnd = `span 2`;

                            servoGrid.appendChild(tileDiv);
                            updateTileColor(tileId); 
                            
                            maxColumnIndexUsed = Math.max(maxColumnIndexUsed, gridColumnStart + 1);
                            maxRowIndexUsed = Math.max(maxRowIndexUsed, gridRowStart + 1);

                            // Populate visualCoordToNumericIdMap
                            if (numericId !== null) {
                                visualCoordToNumericIdMap.set(`${gridRowStart},${gridColumnStart}`, numericId);
                            }
                        }
                    }

                    for (let originalCol = 1; originalCol <= totalOriginalCols; originalCol++) {
                        const tileId = `[${originalRow},${originalCol}]`; 

                        if (originalCol % 2 === 0) {
                            const tileDiv = document.createElement('div'); 
                            tileDiv.classList.add('servo-tile');
                            tileDiv.dataset.tileId = tileId;
                            const numericId = servoSettings[tileId].numericId; 
                            tileDiv.dataset.numericId = numericId !== null ? numericId : ''; 

                            const span = document.createElement('span');
                            // Text content and opacity will be set by updateTileColor
                            tileDiv.appendChild(span);

                            tileDiv.addEventListener('click', () => {
                                if (isInteractiveSetupMode) {
                                    handleInteractiveTileClick(tileId);
                                } else if (isPaintMode) { // New condition for Paint Mode
                                    handlePaintModeClick(tileId);
                                }
                                else {
                                    openPopup(tileId);
                                }
                            });

                            const gridRowStart = (currentVisualRow - 1) * 2 + 2;
                            const gridColumnStart = (originalCol - 1) + 1;

                            tileDiv.style.gridRowStart = gridRowStart;
                            tileDiv.style.gridRowEnd = `span 2`;
                            tileDiv.style.gridColumnStart = gridColumnStart;
                            tileDiv.style.gridColumnEnd = `span 2`;

                            servoGrid.appendChild(tileDiv);
                            updateTileColor(tileId); 
                            
                            maxColumnIndexUsed = Math.max(maxColumnIndexUsed, gridColumnStart + 1);
                            maxRowIndexUsed = Math.max(maxRowIndexUsed, gridRowStart + 1);

                            // Populate visualCoordToNumericIdMap
                            if (numericId !== null) {
                                visualCoordToNumericIdMap.set(`${gridRowStart},${gridColumnStart}`, numericId);
                            }
                        }
                    }
                    currentVisualRow++;
                }

                const rootStyle = getComputedStyle(document.documentElement);
                const bodyPaddingX = parseFloat(rootStyle.getPropertyValue('--body-padding')) * 2;
                const bodyPaddingY = parseFloat(rootStyle.getPropertyValue('--body-padding')) * 2;
                const gridPaddingX = parseFloat(rootStyle.getPropertyValue('--grid-padding')) * 2;
                const gridPaddingY = parseFloat(rootStyle.getPropertyValue('--grid-padding')) * 2;

                const h1Element = document.querySelector('h1');
                const h1Style = getComputedStyle(h1Element);
                const h1Height = h1Element.offsetHeight;
                const h1MarginBottom = parseFloat(h1Style.marginBottom);

                const motionControls = document.querySelector('.motion-controls');
                const motionControlsWidth = motionControls.offsetWidth;
                const motionControlsMarginRight = parseFloat(getComputedStyle(motionControls).marginRight);
                
                const settingsControls = document.querySelector('.settings-controls'); 
                const settingsControlsWidth = settingsControls.offsetWidth; 
                const settingsControlsMarginLeft = parseFloat(getComputedStyle(settingsControls).marginLeft); 

                const availableWidth = window.innerWidth - bodyPaddingX - gridPaddingX - motionControlsWidth - motionControlsMarginRight - settingsControlsWidth - settingsControlsMarginLeft; 
                const availableHeight = window.innerHeight - bodyPaddingY - gridPaddingY - h1Height - h1MarginBottom;
                
                const MIN_BASE_GRID_UNIT = 20;
                const unitFromWidth = availableWidth / (maxColumnIndexUsed > 0 ? maxColumnIndexUsed : 1); 
                const unitFromHeight = availableHeight / (maxRowIndexUsed > 0 ? maxRowIndexUsed : 1);   
                const newBaseGridUnit = Math.max(MIN_BASE_GRID_UNIT, Math.min(unitFromWidth, unitFromHeight));

                document.documentElement.style.setProperty('--base-grid-unit', newBaseGridUnit + 'px');

                servoGrid.style.gridTemplateColumns = `repeat(${maxColumnIndexUsed}, var(--base-grid-unit))`;
                servoGrid.style.gridTemplateRows = `repeat(${maxRowIndexUsed}, var(--base-grid-unit))`;
            }

            function initializeServoSettings(tileId, originalCol) {
                const isFirstOrLastCol = (originalCol === 1 || originalCol === totalOriginalCols);
                const isSpecificallyDeactivated = deactivatedModules.has(tileId);
                
                // A tile is active if it's not in the first/last column AND not specifically deactivated,
                // AND its numeric ID is within the TOTAL_SERVOS_TO_DISPLAY limit.
                const isActive = !(isFirstOrLastCol || isSpecificallyDeactivated) && servoIdCounter < TOTAL_SERVOS_TO_DISPLAY; 

                let numericId = null; 
                if (isActive) { // Only assign numeric IDs to active tiles
                    numericId = servoIdCounter++; 
                }
                
                servoIdToNumericIdMap.set(tileId, numericId); 
                if (numericId !== null) {
                    numericIdToServoIdMap.set(numericId, tileId);
                }

                servoSettings[tileId] = servoSettings[tileId] || { 
                    lowerLimit: 0, 
                    upperLimit: 180, 
                    moveToPosition: 90, 
                    isActive: isActive, 
                    numericId: numericId,
                    boardNumber: null, 
                    pinNumber: null    
                };
            }

            function openPopup(tileId) {
                activeTile = tileId;
                const numericId = servoSettings[activeTile].numericId;
                currentTileIdSpan.textContent = numericId !== null ? `${numericId}` : ''; 

                // Display current known values or 'Loading...'
                moduleNumberInput.value = servoSettings[activeTile].boardNumber !== null ? servoSettings[activeTile].boardNumber : 'Loading...';
                pinNumberInput.value = servoSettings[activeTile].pinNumber !== null ? servoSettings[activeTile].pinNumber : 'Loading...';
                lowerLimitInput.value = servoSettings[activeTile].lowerLimit !== null ? servoSettings[activeTile].lowerLimit : 'Loading...';
                upperLimitInput.value = servoSettings[activeTile].upperLimit !== null ? servoSettings[activeTile].upperLimit : 'Loading...';
                moveToPositionInput.value = servoSettings[activeTile].moveToPosition || 90; 

                popup.style.display = 'flex';
                // Request fresh data from backend for this specific servo
                sendDataViaWebSocket('fetchLimits', null, numericId); 
            }

            function closePopup() {
                popup.style.display = 'none';
                activeTile = null;
            }

            sendBoardPinButton.addEventListener('click', () => {
                if (activeTile) {
                    const boardValue = parseInt(moduleNumberInput.value);
                    const pinValue = parseInt(pinNumberInput.value);
                    const numericId = servoSettings[activeTile].numericId;

                    if (!isNaN(boardValue) && !isNaN(pinValue) && boardValue >= 0 && boardValue < TOTAL_BOARDS && pinValue >= 0 && pinValue < SERVOS_PER_BOARD) {
                        servoSettings[activeTile].boardNumber = boardValue;
                        servoSettings[activeTile].pinNumber = pinValue;
                        sendDataViaWebSocket('setBoardPin', { board: boardValue, pin: pinValue }, numericId);
                    } else {
                        console.error("Board and Pin values must be valid numbers within range.");
                    }
                }
            });

            sendLowerLimitButton.addEventListener('click', () => {
                if (activeTile) {
                    const value = parseInt(lowerLimitInput.value);
                    const numericId = servoSettings[activeTile].numericId;
                    if (value >= 0 && value <= 180) {
                        servoSettings[activeTile].lowerLimit = value;
                        sendDataViaWebSocket('lowerLimit', value, numericId);
                    } else {
                        console.error("Lower Limit must be between 0 and 180.");
                    }
                }
            });

            sendUpperLimitButton.addEventListener('click', () => {
                if (activeTile) {
                    const value = parseInt(upperLimitInput.value);
                    const numericId = servoSettings[activeTile].numericId;
                    if (value >= 0 && value <= 180) {
                        servoSettings[activeTile].upperLimit = value;
                        sendDataViaWebSocket('upperLimit', value, numericId);
                    } else {
                        console.error("Upper Limit must be between 0 and 180.");
                    }
                }
            });

            sendToPositionButton.addEventListener('click', () => {
                if (activeTile) {
                    const value = parseInt(moveToPositionInput.value);
                    const numericId = servoSettings[activeTile].numericId;
                    if (value >= 0 && value <= 180) {
                        servoSettings[activeTile].moveToPosition = value;
                        if (!animationFrameId && !isPaintMode) { // Only update if not in animation or paint mode
                            updateTileColor(activeTile);
                        }
                        sendDataViaWebSocket('moveToPosition', value, numericId);
                    } else {
                        console.error("Move To Position must be between 0 and 180.");
                    }
                }
            });

            testButton.addEventListener('click', () => {
                if (activeTile) {
                    const settings = servoSettings[activeTile];
                    const lower = settings.lowerLimit;
                    const upper = settings.upperLimit;
                    const numericId = settings.numericId;

                    if (lower <= upper && lower >= 0 && upper <= 180) {
                        if (settings.isActive) {
                            sendDataViaWebSocket('test', { lower: lower, upper: upper }, numericId);
                            // Frontend no longer simulates the full test motion, Arduino does.
                        } else {
                            console.log(`Servo ${activeTile} is inactive, test motion skipped.`);
                        }
                    } else {
                        console.error("Invalid limits for testing. Ensure Lower Limit <= Upper Limit and within 0-180 range.");
                    }
                }
            });

            function stopCurrentAnimation() {
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }
                if (animationIntervalId) { 
                    clearInterval(animationIntervalId);
                    animationIntervalId = null;
                }
                animationStartTime = null;
                
                waveMotionButton.textContent = "Wave";
                pulseMotionButton.textContent = "Pulse"; 
                sweepMotionButton.textContent = "Sweep"; 
                rippleMotionButton.textContent = "Ripple";
                currentAnimationFunction = null; 
                
                Object.keys(servoSettings).forEach(tileId => {
                    updateTileColor(tileId);
                });
            }

            function sendActiveServoPositions() {
                const activeServoData = new Uint8Array(TOTAL_SERVOS_TO_DISPLAY); 
                
                for (let i = 0; i < TOTAL_SERVOS_TO_DISPLAY; i++) {
                    const tileId = numericIdToServoIdMap.get(i);
                    if (tileId && servoSettings[tileId].isActive) {
                        const position = servoSettings[tileId].moveToPosition;
                        activeServoData[i] = Math.max(0, Math.min(180, position)); 
                    } else {
                        // If a servo is inactive or not found in map, send its default position (90)
                        activeServoData[i] = 90; 
                    }
                }
                sendDataViaWebSocket('streamPositions', activeServoData);
            }

            let animationStartTime = null; 

            function animateWave(currentTime) {
                if (!animationStartTime) {
                    animationStartTime = currentTime;
                }
                const elapsedTime = currentTime - animationStartTime;
                const timeProgress = (elapsedTime % animationDuration) / animationDuration;
                const globalPhase = timeProgress * 2 * Math.PI;

                const tiles = document.querySelectorAll('.servo-tile');
                let maxGridColumn = 0;
                tiles.forEach(tile => {
                    const col = parseInt(tile.style.gridColumnStart);
                    if (!isNaN(col) && col > maxGridColumn) {
                        maxGridColumn = col;
                    }
                });

                if (maxGridColumn === 0) {
                    animationFrameId = requestAnimationFrame(animateWave);
                    return;
                }
                const phaseShiftPerColumn = (2 * Math.PI) / maxGridColumn;

                tiles.forEach(tile => {
                    const tileId = tile.dataset.tileId;
                    if (servoSettings[tileId].isActive) {
                        const parts = tileId.match(/\[(\d+),(\d+)\]/);
                        let tileColIndex = 0;
                        if (parts && parts.length > 2) {
                            tileColIndex = parseInt(parts[2]);
                        }
                        const tilePhaseOffset = (tileColIndex - 1) * phaseShiftPerColumn;
                        const currentTilePhase = globalPhase + tilePhaseOffset;

                        const sineValue = Math.sin(currentTilePhase);
                        const normalizedValue = (sineValue + 1) / 2;
                        const newMoveToPosition = Math.round(normalizedValue * 180);

                        servoSettings[tileId].moveToPosition = newMoveToPosition;
                    }
                    updateTileColor(tileId); 
                });
                animationFrameId = requestAnimationFrame(animateWave);
            }

            function animateFullSweep(currentTime) {
                if (!animationStartTime) {
                    animationStartTime = currentTime;
                }
                const elapsedTime = currentTime - animationStartTime;
                const progress = (elapsedTime % animationDuration) / animationDuration; 

                let position;
                if (progress <= 0.5) {
                    position = progress * 2 * 180;
                } else {
                    position = 180 - ((progress - 0.5) * 2 * 180);
                }
                position = Math.round(position); 

                Object.keys(servoSettings).forEach(tileId => {
                    if (servoSettings[tileId].isActive) {
                        servoSettings[tileId].moveToPosition = position;
                    }
                    updateTileColor(tileId); 
                });
                animationFrameId = requestAnimationFrame(animateFullSweep);
            }

            function animatePulse(currentTime) {
                if (!animationStartTime) {
                    animationStartTime = currentTime;
                }
                const elapsedTime = currentTime - animationStartTime;
                const timeProgress = (elapsedTime % animationDuration) / animationDuration; 
                const globalPhase = timeProgress * 2 * Math.PI;

                const tiles = document.querySelectorAll('.servo-tile');
                
                let minTileRow = Infinity, maxTileRow = -Infinity;
                tiles.forEach(tile => {
                    const parts = tile.dataset.tileId.match(/\[(\d+),(\d+)\]/);
                    const row = parseInt(parts[1]); 
                    minTileRow = Math.min(minTileRow, row);
                    maxTileRow = Math.max(maxTileRow, row);
                });

                const totalVerticalSpan = maxTileRow - minTileRow + 1; 
                const phaseShiftPerRow = (2 * Math.PI) / totalVerticalSpan;

                tiles.forEach(tile => {
                    const tileId = tile.dataset.tileId;
                    if (servoSettings[tileId].isActive) {
                        const parts = tileId.match(/\[(\d+),(\d+)\]/);
                        let tileRowIndex = 0;
                        if (parts && parts.length > 1) {
                            tileRowIndex = parseInt(parts[1]); 
                        }

                        const currentTilePhase = globalPhase + (tileRowIndex - minTileRow) * phaseShiftPerRow;
                        
                        const sineValue = Math.sin(currentTilePhase); 
                        const normalizedValue = (sineValue + 1) / 2; 
                        const newMoveToPosition = Math.round(normalizedValue * 180); 

                        servoSettings[tileId].moveToPosition = newMoveToPosition;
                    }
                    updateTileColor(tileId); 
                });
                animationFrameId = requestAnimationFrame(animatePulse); 
            }

            function animateRipple(currentTime) {
                if (!animationStartTime) {
                    animationStartTime = currentTime;
                }
                const elapsedTime = currentTime - animationStartTime;
                const timeProgress = (elapsedTime % animationDuration) / animationDuration; 
                
                const tiles = document.querySelectorAll('.servo-tile');
                
                let minTileCol = Infinity, maxTileCol = -Infinity;
                let minTileRow = Infinity, maxTileRow = -Infinity;
                tiles.forEach(tile => {
                    const col = parseInt(tile.style.gridColumnStart);
                    const row = parseInt(tile.style.gridRowStart);
                    minTileCol = Math.min(minTileCol, col);
                    maxTileCol = Math.max(maxTileCol, col);
                    minTileRow = Math.min(minTileRow, row);
                    maxTileRow = Math.max(maxTileRow, row);
                });

                const centerCol = (minTileCol + maxTileCol) / 2;
                const centerRow = (minTileRow + maxTileRow) / 2;

                const cornerDistances = [
                    Math.sqrt(Math.pow(minTileCol - centerCol, 2) + Math.pow(minTileRow - centerRow, 2)),
                    Math.sqrt(Math.pow(maxTileCol - centerCol, 2) + Math.pow(minTileRow - centerRow, 2)),
                    Math.sqrt(Math.pow(minTileCol - centerCol, 2) + Math.pow(maxTileRow - centerRow, 2)),
                    Math.sqrt(Math.pow(maxTileCol - centerCol, 2) + Math.pow(maxTileRow - centerRow, 2)) 
                ];
                const maxSpreadDistance = Math.max(...cornerDistances);

                const wavelength = 2 * Math.PI; 

                tiles.forEach(tile => {
                    const tileId = tile.dataset.tileId;
                    if (servoSettings[tileId].isActive) {
                        const tileCol = parseInt(tile.style.gridColumnStart);
                        const tileRow = parseInt(tile.style.gridRowStart);

                        const distance = Math.sqrt(Math.pow(tileCol - centerCol, 2) + Math.pow(tileRow - centerRow, 2));
                        
                        const phase = (distance / maxSpreadDistance) * wavelength + (timeProgress * wavelength);
                        
                        const sineValue = Math.sin(phase); 
                        const normalizedValue = (sineValue + 1) / 2; 
                        const newMoveToPosition = Math.round(normalizedValue * 180); 

                        servoSettings[tileId].moveToPosition = newMoveToPosition;
                    }
                    updateTileColor(tileId); 
                });
                animationFrameId = requestAnimationFrame(animateRipple);
            }

            setFpsButton.addEventListener('click', () => {
                let newFPS = parseInt(fpsInput.value);
                if (isNaN(newFPS) || newFPS < 1 || newFPS > 60) {
                    console.warn("Invalid FPS value. Please enter a number between 1 and 60.");
                    fpsInput.value = currentFPS;
                    return;
                }
                currentFPS = newFPS;
                streamFrequency = 1000 / currentFPS;

                if (animationIntervalId) {
                    clearInterval(animationIntervalId);
                    animationIntervalId = setInterval(sendActiveServoPositions, streamFrequency);
                }
            });


            waveMotionButton.addEventListener('click', () => { 
                stopTestAllServosAnimation(); 
                stopPaintMode(); // Stop paint mode if active
                if (currentAnimationFunction === animateWave) { 
                    stopCurrentAnimation(); 
                } else {
                    stopCurrentAnimation(); 
                    currentAnimationFunction = animateWave; 
                    animationStartTime = null; 
                    animateWave(); 
                    animationIntervalId = setInterval(sendActiveServoPositions, streamFrequency); 
                    waveMotionButton.textContent = "Stop Wave";
                }
                updateButtonStates();
            });

            pulseMotionButton.addEventListener('click', () => { 
                stopTestAllServosAnimation(); 
                stopPaintMode(); // Stop paint mode if active
                if (currentAnimationFunction === animatePulse) { 
                    stopCurrentAnimation(); 
                } else {
                    stopCurrentAnimation(); 
                    currentAnimationFunction = animatePulse; 
                    animationStartTime = null; 
                    animatePulse(); 
                    animationIntervalId = setInterval(sendActiveServoPositions, streamFrequency); 
                    pulseMotionButton.textContent = "Stop Pulse"; 
                }
                updateButtonStates();
            });

            sweepMotionButton.addEventListener('click', () => { 
                stopTestAllServosAnimation(); 
                stopPaintMode(); // Stop paint mode if active
                if (currentAnimationFunction === animateFullSweep) { 
                    stopCurrentAnimation(); 
                } else {
                    stopCurrentAnimation(); 
                    currentAnimationFunction = animateFullSweep; 
                    animationStartTime = null; 
                    animateFullSweep(); 
                    animationIntervalId = setInterval(sendActiveServoPositions, streamFrequency); 
                    sweepMotionButton.textContent = "Stop Sweep"; 
                }
                updateButtonStates();
            });

            rippleMotionButton.addEventListener('click', () => { 
                stopTestAllServosAnimation(); 
                stopPaintMode(); // Stop paint mode if active
                if (currentAnimationFunction === animateRipple) {
                    stopCurrentAnimation();
                } else {
                    stopCurrentAnimation();
                    currentAnimationFunction = animateRipple;
                    animationStartTime = null;
                    animateRipple();
                    animationIntervalId = setInterval(sendActiveServoPositions, streamFrequency); 
                    rippleMotionButton.textContent = "Stop Ripple";
                }
                updateButtonStates();
            });

            closePopupButton.addEventListener('click', closePopup);

            // --- Configure All Logic ---
            configureAllButton.addEventListener('click', () => {
                closePopup(); 
                stopPaintMode(); // Stop paint mode if active
                stopInteractiveSetup(); // Stop interactive setup if active
                stopTestAllServosAnimation(); // Stop test all servos if active
                openConfigureAllPopup();
            });

            closeConfigureAllPopupButton.addEventListener('click', () => {
                configureAllPopup.style.display = 'none';
            });

            function openConfigureAllPopup() {
                configureAllPopup.style.display = 'flex';
                populateConfigTable(null); 
                sendDataViaWebSocket('fetchAllLimits', null); 
            }

            function populateConfigTable(servosData) {
                configTableBody.innerHTML = ''; 
                const fetchedMap = new Map();
                if (servosData) {
                    servosData.forEach(servo => fetchedMap.set(servo.numericServoId, servo));
                }

                for (let i = 0; i < TOTAL_SERVOS_TO_DISPLAY; i++) {
                    const row = configTableBody.insertRow();
                    row.dataset.numericServoId = i; 

                    const servo = fetchedMap.get(i); 

                    row.insertCell().textContent = i; 

                    const boardCell = row.insertCell();
                    const boardInput = document.createElement('input');
                    boardInput.type = 'number';
                    boardInput.value = servo ? servo.boardNumber : 'Loading...';
                    boardInput.min = 0;
                    boardInput.max = TOTAL_BOARDS - 1; 
                    boardInput.dataset.originalValue = boardInput.value; 
                    boardInput.dataset.fieldName = 'boardNumber'; 
                    boardInput.addEventListener('change', (event) => handleConfigInputChange(event, i, 'boardNumber', boardInput));
                    boardCell.appendChild(boardInput);

                    const pinCell = row.insertCell();
                    const pinInput = document.createElement('input');
                    pinInput.type = 'number';
                    pinInput.value = servo ? servo.pinNumber : 'Loading...';
                    pinInput.min = 0;
                    pinInput.max = SERVOS_PER_BOARD - 1; 
                    pinInput.dataset.originalValue = pinInput.value; 
                    pinInput.dataset.fieldName = 'pinNumber'; 
                    pinInput.addEventListener('change', (event) => handleConfigInputChange(event, i, 'pinNumber', pinInput));
                    pinCell.appendChild(pinInput);

                    const lowerLimitCell = row.insertCell();
                    const lowerLimitInput = document.createElement('input');
                    lowerLimitInput.type = 'number';
                    lowerLimitInput.value = servo ? servo.lowerLimit : 'Loading...';
                    lowerLimitInput.min = 0;
                    lowerLimitInput.max = 180;
                    lowerLimitInput.dataset.originalValue = lowerLimitInput.value; 
                    lowerLimitInput.dataset.fieldName = 'lowerLimit'; 
                    lowerLimitInput.addEventListener('change', (event) => handleConfigInputChange(event, i, 'lowerLimit', lowerLimitInput));
                    lowerLimitCell.appendChild(lowerLimitInput);

                    const upperLimitCell = row.insertCell();
                    const upperLimitInput = document.createElement('input');
                    upperLimitInput.type = 'number';
                    upperLimitInput.value = servo ? servo.upperLimit : 'Loading...';
                    upperLimitInput.min = 0;
                    upperLimitInput.max = 180;
                    upperLimitInput.dataset.originalValue = upperLimitInput.value; 
                    upperLimitInput.dataset.fieldName = 'upperLimit'; 
                    upperLimitInput.addEventListener('change', (event) => handleConfigInputChange(event, i, 'upperLimit', upperLimitInput));
                    upperLimitCell.appendChild(upperLimitInput);

                    const actionsCell = row.insertCell();
                    const testConfigButton = document.createElement('button');
                    testConfigButton.classList.add('test-config-button');
                    testConfigButton.textContent = 'Test';
                    testConfigButton.addEventListener('click', () => {
                        const currentLower = parseInt(lowerLimitInput.value);
                        const currentUpper = parseInt(upperLimitInput.value);

                        if (!isNaN(currentLower) && !isNaN(currentUpper) && currentLower <= currentUpper && currentLower >= 0 && currentUpper <= 180) {
                            sendDataViaWebSocket('test', { lower: currentLower, upper: currentUpper }, i);
                            // Frontend no longer simulates the full test motion, Arduino does.
                        } else {
                            console.error("Invalid limits for testing this servo. Ensure Lower Limit <= Upper Limit and within 0-180 range.");
                        }
                    });
                    actionsCell.appendChild(testConfigButton);
                }
            }

            // Custom Confirmation Dialog Logic
            let currentConfirmCallback = null;
            let currentCancelCallback = null;

            function showConfirmationDialog(message, onConfirm, onCancel) {
                confirmationMessage.textContent = message;
                currentConfirmCallback = onConfirm;
                currentCancelCallback = onCancel;
                confirmationDialog.style.display = 'flex';
            }

            confirmYesButton.addEventListener('click', () => {
                if (currentConfirmCallback) {
                    currentConfirmCallback();
                }
                confirmationDialog.style.display = 'none';
            });

            confirmNoButton.addEventListener('click', () => {
                if (currentCancelCallback) {
                    currentCancelCallback();
                }
                confirmationDialog.style.display = 'none';
            });

            // Function to handle input changes and confirmations in the config table
            function handleConfigInputChange(event, numericServoId, fieldName, inputElement) {
                const newValue = parseInt(inputElement.value);
                const originalValue = parseInt(inputElement.dataset.originalValue);

                if (isNaN(newValue) || (fieldName === 'lowerLimit' || fieldName === 'upperLimit') && (newValue < 0 || newValue > 180)) {
                    console.warn(`Invalid input for ${fieldName} for servo ${numericServoId}. Reverting.`);
                    inputElement.value = originalValue; 
                    return;
                }
                if ((fieldName === 'boardNumber') && (newValue < 0 || newValue >= TOTAL_BOARDS)) {
                    console.warn(`Invalid board number for servo ${numericServoId}. Reverting.`);
                    inputElement.value = originalValue; 
                    return;
                }
                if ((fieldName === 'pinNumber') && (newValue < 0 || newValue >= SERVOS_PER_BOARD)) {
                    console.warn(`Invalid pin number for servo ${numericServoId}. Reverting.`);
                    inputElement.value = originalValue; 
                    return;
                }

                if (newValue === originalValue) {
                    return;
                }

                showConfirmationDialog(`Update ${fieldName} for Servo ${numericServoId} to ${newValue}?`, () => {
                    let messageType;
                    let messageValue;

                    const row = inputElement.closest('tr');
                    const boardInputInRow = row.querySelector('input[data-field-name="boardNumber"]');
                    const pinInputInRow = row.querySelector('input[data-field-name="pinNumber"]');

                    if (fieldName === 'boardNumber' || fieldName === 'pinNumber') {
                        messageType = 'setBoardPin';
                        messageValue = {
                            board: parseInt(boardInputInRow.value),
                            pin: parseInt(pinInputInRow.value)
                        };
                    } else if (fieldName === 'lowerLimit') {
                        messageType = 'lowerLimit';
                        messageValue = newValue;
                    } else if (fieldName === 'upperLimit') {
                        messageType = 'upperLimit';
                        messageValue = newValue;
                    } else {
                        console.error(`Unknown fieldName: ${fieldName}`);
                        return;
                    }

                    sendDataViaWebSocket(messageType, messageValue, numericServoId); 
                    inputElement.dataset.originalValue = newValue;

                    const tileId = numericIdToServoIdMap.get(numericId);
                    if (tileId) {
                        if (fieldName === 'boardNumber') servoSettings[tileId].boardNumber = newValue;
                        else if (fieldName === 'pinNumber') servoSettings[tileId].pinNumber = newValue;
                        else if (fieldName === 'lowerLimit') servoSettings[tileId].lowerLimit = newValue;
                        else if (fieldName === 'upperLimit') servoSettings[tileId].upperLimit = newValue; 
                        if (popup.style.display === 'none' && !isPaintMode) { // Only update if not in animation or paint mode
                            updateTileColor(tileId);
                        }
                    }

                }, () => {
                    inputElement.value = originalValue;
                });
            }

            // Export CSV Functionality 
            exportCsvButton.addEventListener('click', () => {
                const allServoConfigs = [];
                const rows = configTableBody.querySelectorAll('tr');
                rows.forEach(row => {
                    const numericServoId = parseInt(row.dataset.numericServoId);
                    const inputs = row.querySelectorAll('input[type="number"]');
                    const boardNumber = parseInt(inputs[0].value);
                    const pinNumber = parseInt(inputs[1].value);
                    const lowerLimit = parseInt(inputs[2].value);
                    const upperLimit = parseInt(inputs[3].value);

                    allServoConfigs.push({
                        numericServoId: numericServoId,
                        boardNumber: boardNumber,
                        pinNumber: pinNumber,
                        lowerLimit: lowerLimit,
                        upperLimit: upperLimit
                    });
                });

                let csvContent = "ID,Board,Pin,Lower Limit,Upper Limit\n"; 
                allServoConfigs.forEach(servo => {
                    csvContent += `${servo.numericServoId},${servo.boardNumber},${servo.pinNumber},${servo.lowerLimit},${servo.upperLimit}\n`;
                });

                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'servo_configurations.csv'; 
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });

            // Import CSV Functionality 
            importCsvButton.addEventListener('click', () => {
                importCsvInput.click(); 
            });

            importCsvInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (!file) {
                    return;
                }

                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const csvText = e.target.result;
                        const parsedData = parseCsv(csvText); 

                        if (!Array.isArray(parsedData) || parsedData.length === 0) {
                            throw new Error("Invalid CSV format: No data rows found or incorrect structure.");
                        }

                        const isValid = parsedData.every(servo =>
                            typeof servo.numericServoId === 'number' &&
                            typeof servo.boardNumber === 'number' &&
                            typeof servo.pinNumber === 'number' &&
                            typeof servo.lowerLimit === 'number' &&
                            typeof servo.upperLimit === 'number' &&
                            servo.numericServoId >= 0 && servo.numericServoId < TOTAL_SERVOS_TO_DISPLAY &&
                            servo.boardNumber >= 0 && servo.boardNumber < TOTAL_BOARDS &&
                            servo.pinNumber >= 0 && servo.pinNumber < SERVOS_PER_BOARD &&
                            servo.lowerLimit >= 0 && servo.lowerLimit <= 180 &&
                            servo.upperLimit >= 0 && servo.upperLimit <= 180
                        );

                        if (!isValid) {
                            throw new Error("Invalid CSV data: Missing or incorrect servo properties, or values out of range.");
                        }

                        showConfirmationDialog("Are you sure you want to import this configuration? This will overwrite existing settings and send updates to the device EEPROM. This operation cannot be undone.", () => {
                            applyImportedData(parsedData);
                            sendDataViaWebSocket('batchUpdateConfig', parsedData);
                        }, () => {
                        });

                    } catch (error) {
                        alert(`Error importing CSV: ${error.message}`); 
                    } finally {
                        event.target.value = ''; 
                    }
                };
                reader.readAsText(file);
            });

            function parseCsv(csvText) {
                const lines = csvText.trim().split('\n');
                if (lines.length < 2) { 
                    throw new Error("CSV file is empty or missing data rows.");
                }

                const headers = lines[0].split(',').map(h => h.trim());
                const expectedHeaders = ["ID", "Board", "Pin", "Lower Limit", "Upper Limit"];

                if (headers.length !== expectedHeaders.length || !headers.every((h, i) => h === expectedHeaders[i])) {
                    throw new Error("Invalid CSV header. Expected: ID,Board,Pin,Lower Limit,Upper Limit");
                }

                const result = [];
                for (let i = 1; i < lines.length; i++) {
                    const values = lines[i].split(',').map(v => v.trim());
                    if (values.length !== expectedHeaders.length) {
                        console.warn(`Skipping malformed row ${i + 1}: "${lines[i]}". Incorrect number of columns.`);
                        continue; 
                    }

                    const servo = {
                        numericServoId: parseInt(values[0]),
                        boardNumber: parseInt(values[1]),
                        pinNumber: parseInt(values[2]),
                        lowerLimit: parseInt(values[3]),
                        upperLimit: parseInt(values[4])
                    };

                    if (isNaN(servo.numericServoId) || isNaN(servo.boardNumber) || isNaN(servo.pinNumber) || isNaN(servo.lowerLimit) || isNaN(servo.upperLimit) ||
                        servo.numericServoId < 0 || servo.numericServoId >= TOTAL_SERVOS_TO_DISPLAY ||
                        servo.boardNumber < 0 || servo.boardNumber >= TOTAL_BOARDS ||
                        servo.pinNumber < 0 || servo.pinNumber >= SERVOS_PER_BOARD ||
                        servo.lowerLimit < 0 || servo.lowerLimit > 180 ||
                        servo.upperLimit < 0 || servo.upperLimit > 180) {
                        console.warn(`Skipping row ${i + 1} due to invalid data: ${JSON.stringify(servo)}`);
                        continue;
                    }
                    result.push(servo);
                }
                return result;
            }


            function applyImportedData(data) {
                data.forEach(importedServo => {
                    const numericId = importedServo.numericServoId;
                    const row = configTableBody.querySelector(`tr[data-numeric-servo-id="${numericId}"]`);
                    if (row) {
                        const inputs = row.querySelectorAll('input[type="number"]');
                        if (inputs.length >= 4) {
                            inputs[0].value = importedServo.boardNumber;
                            inputs[0].dataset.originalValue = importedServo.boardNumber;
                            inputs[1].value = importedServo.pinNumber;
                            inputs[1].dataset.originalValue = importedServo.pinNumber;
                            inputs[2].value = importedServo.lowerLimit;
                            inputs[2].dataset.originalValue = importedServo.lowerLimit;
                            inputs[3].value = importedServo.upperLimit;
                            inputs[3].dataset.originalValue = importedServo.upperLimit;

                            const tileId = numericIdToServoIdMap.get(numericId);
                            if (tileId) {
                                servoSettings[tileId].boardNumber = importedServo.boardNumber;
                                servoSettings[tileId].pinNumber = importedServo.pinNumber;
                                servoSettings[tileId].lowerLimit = importedServo.lowerLimit;
                                servoSettings[tileId].upperLimit = importedServo.upperLimit; 
                                if (popup.style.display === 'none' && !isPaintMode) { // Only update if not in animation or paint mode
                                    updateTileColor(tileId);
                                }
                            }
                        }
                    } else {
                        console.warn(`Servo with numeric ID ${numericId} not found in table during import.`);
                    }
                });
            }

            // --- Interactive Setup Logic ---
            interactiveSetupButton.addEventListener('click', () => {
                if (isInteractiveSetupMode) {
                    // When stopping, send command to Arduino and let Arduino confirm the stop
                    // Frontend will update its state upon receiving 'interactiveSetupComplete' from Arduino
                    sendDataViaWebSocket('stopInteractiveSetup');
                    interactiveSetupMessage.innerHTML = "Stopping interactive setup... Please wait.";
                    interactiveSetupMessage.style.display = 'block';
                    interactiveSetupButtonsContainer.style.display = 'none'; // Hide buttons immediately
                    updateButtonStates(); // Update button states to reflect "stopping" state
                } else {
                    startInteractiveSetup();
                }
            });

            // Event listener for repeat button (renamed from blinkCurrentServoButton)
            repeatBlinkServoButton.addEventListener('click', () => {
                if (isInteractiveSetupMode && currentBlinkingPhysicalServo.board !== -1 && currentBlinkingPhysicalServo.pin !== -1) {
                    sendDataViaWebSocket('reBlinkServo', {
                        board: currentBlinkingPhysicalServo.board,
                        pin: currentBlinkingPhysicalServo.pin
                    });
                } else {
                    console.warn("No servo to re-blink or not in interactive setup mode.");
                }
            });

            // Event listener for back button
            previousServoButton.addEventListener('click', () => {
                if (isInteractiveSetupMode) {
                    sendDataViaWebSocket('previousServo');
                    interactiveSetupMessage.innerHTML = "Requesting previous servo... Please wait.";
                    interactiveSetupButtonsContainer.style.display = 'flex'; // Keep container visible
                    interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                    repeatBlinkServoButton.disabled = true; // Disable until new servo blinks
                    previousServoButton.disabled = true; // Disable until new servo blinks
                } else {
                    console.warn("Not in interactive setup mode to go back.");
                }
            });


            function startInteractiveSetup() {
                isInteractiveSetupMode = true;
                isTestAllServosMode = false; 
                isPaintMode = false; // Stop paint mode if active
                stopCurrentAnimation(); 
                stopTestAllServosAnimation(); 
                stopPaintMode(); // Ensure paint mode is off

                currentBlinkingPhysicalServo = { board: -1, pin: -1 }; 
                
                // Show the button container immediately when interactive setup starts
                interactiveSetupButtonsContainer.style.display = 'flex'; 
                interactiveSetupMessage.innerHTML = "Starting interactive setup... Please wait for the first servo to blink.";
                interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                
                // Show buttons but disable them initially
                repeatBlinkServoButton.style.display = 'inline-block';
                previousServoButton.style.display = 'inline-block';
                repeatBlinkServoButton.disabled = true;
                previousServoButton.disabled = true;

                interactiveSetupMessage.style.display = 'block';

                sendDataViaWebSocket('startInteractiveSetup', {
                    totalBoards: TOTAL_BOARDS,
                    servosPerBoard: SERVOS_PER_BOARD,
                    meanPosition: 90, 
                    testRange: 40    
                });
                updateButtonStates(); // Update button states after showing container
            }

            function stopInteractiveSetup() {
                isInteractiveSetupMode = false;
                interactiveSetupMessage.textContent = "Interactive setup stopped.";
                interactiveSetupMessage.style.display = 'none'; 
                interactiveSetupButtonsContainer.style.display = 'none'; // Hide the button container
                repeatBlinkServoButton.style.display = 'none'; // Hide the button
                previousServoButton.style.display = 'none'; // Hide the button
                repeatBlinkServoButton.disabled = true; // Ensure disabled
                previousServoButton.disabled = true; // Ensure disabled
                currentBlinkingPhysicalServo = { board: -1, pin: -1 };
                nextServoToMapIndex = 0;
                updateButtonStates(); 
                sendDataViaWebSocket('stopInteractiveSetup'); // This line is removed from here and moved to the button listener

                // Remove 'mapped' class from all tiles when interactive setup stops
                document.querySelectorAll('.servo-tile.mapped').forEach(tile => {
                    tile.classList.remove('mapped');
                    updateTileColor(tile.dataset.tileId); // Revert color based on current position
                });
            }

            function handleInteractiveTileClick(tileId) {
                if (!isInteractiveSetupMode) {
                    openPopup(tileId);
                    return;
                }

                if (currentBlinkingPhysicalServo.board === -1 || currentBlinkingPhysicalServo.pin === -1) {
                    interactiveSetupMessage.innerHTML = "Please wait for a physical servo to start blinking before clicking a tile.";
                    interactiveSetupButtonsContainer.style.display = 'flex'; // Keep container visible
                    interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                    repeatBlinkServoButton.disabled = true; // Keep disabled
                    previousServoButton.disabled = true; // Keep disabled
                    return;
                }

                const clickedNumericId = servoIdToNumericIdMap.get(tileId);
                if (clickedNumericId === null) {
                    interactiveSetupMessage.innerHTML = "This tile is deactivated and cannot be mapped.";
                    interactiveSetupButtonsContainer.style.display = 'flex'; // Keep container visible
                    interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                    repeatBlinkServoButton.disabled = false; // Keep enabled if a servo is blinking
                    previousServoButton.disabled = false; // Keep enabled if a servo is blinking
                    return;
                }

                showConfirmationDialog(
                    `Map physical servo (Board: ${currentBlinkingPhysicalServo.board}, Pin: ${currentBlinkingPhysicalServo.pin}) to interface Servo ID ${clickedNumericId}?`,
                    () => {
                        servoSettings[tileId].boardNumber = currentBlinkingPhysicalServo.board;
                        servoSettings[tileId].pinNumber = currentBlinkingPhysicalServo.pin;
                        servoSettings[tileId].lowerLimit = 50; 
                        servoSettings[tileId].upperLimit = 130; 
                        servoSettings[tileId].moveToPosition = 90; 

                        sendDataViaWebSocket('mapServo', {
                            numericServoId: clickedNumericId,
                            boardNumber: currentBlinkingPhysicalServo.board,
                            pinNumber: currentBlinkingPhysicalServo.pin,
                            lowerLimit: 50,
                            upperLimit: 130
                        });

                        sendDataViaWebSocket('nextServo');
                        
                        currentBlinkingPhysicalServo = { board: -1, pin: -1 };
                        interactiveSetupMessage.innerHTML = "Mapping confirmed. Waiting for the next servo to blink...";
                        interactiveSetupButtonsContainer.style.display = 'flex'; // Keep container visible
                        interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                        repeatBlinkServoButton.disabled = true; // Disable until next servo blinks
                        previousServoButton.disabled = true; // Disable until next servo blinks
                        
                        // Add 'mapped' class and update color for the successfully mapped tile
                        const tileElement = document.querySelector(`.servo-tile[data-tile-id="${tileId}"]`);
                        if (tileElement) {
                            tileElement.classList.add('mapped'); 
                            updateTileColor(tileId); // Re-apply color logic to show mapped state
                        }
                    },
                    () => {
                        interactiveSetupMessage.innerHTML = "Mapping cancelled. Click another tile or the 'Repeat' button if available.";
                        interactiveSetupButtonsContainer.style.display = 'flex'; // Keep container visible
                        interactiveSetupMessage.appendChild(interactiveSetupButtonsContainer);
                        repeatBlinkServoButton.disabled = false; // Keep button visible if user cancels
                        previousServoButton.disabled = false; // Keep button visible if user cancels
                    }
                );
            }

            // --- Test All Servos Logic ---
            startTestButton.addEventListener('click', () => {
                startTestAllServosAnimation();
            });

            playPauseButton.addEventListener('click', () => {
                if (isTestAllServosMode) { 
                    if (isTestPaused) {
                        resumeTestAllServosAnimation();
                    } else {
                        pauseTestAllServosAnimation();
                    }
                }
            });

            exitTestButton.addEventListener('click', () => {
                stopTestAllServosAnimation();
            });

            forwardButton.addEventListener('click', () => {
                if (isTestAllServosMode) {
                    if (currentTestingServoTileId) {
                        const prevTile = document.querySelector(`.servo-tile[data-tile-id="${currentTestingServoTileId}"]`);
                        if (prevTile) prevTile.classList.remove('highlighted');
                    }
                    currentTestServoIndex++;
                    if (currentTestServoIndex >= activeNumericIds.length) {
                        currentTestServoIndex = 0; 
                    }
                    testAllServosMessageDiv.textContent = `Skipping forward to Servo ID: ${activeNumericIds[currentTestServoIndex]}`;
                    
                    if (testStepPromiseResolve) { 
                        testStepPromiseResolve();
                    } else if (isTestPaused) { 
                    } else {
                    }
                }
            });

            backwardButton.addEventListener('click', () => {
                if (isTestAllServosMode) {
                    if (currentTestingServoTileId) {
                        const prevTile = document.querySelector(`.servo-tile[data-tile-id="${currentTestingServoTileId}"]`);
                        if (prevTile) prevTile.classList.remove('highlighted');
                    }
                    currentTestServoIndex--;
                    if (currentTestServoIndex < 0) {
                        currentTestServoIndex = activeNumericIds.length - 1; 
                    }
                    testAllServosMessageDiv.textContent = `Skipping backward to Servo ID: ${activeNumericIds[currentTestServoIndex]}`;
                    
                    if (testStepPromiseResolve) { 
                        testStepPromiseResolve();
                    } else if (isTestPaused) { 
                    } else {
                    }
                }
            });


            async function startTestAllServosAnimation() {
                isTestAllServosMode = true;
                isInteractiveSetupMode = false; 
                isPaintMode = false; // Stop paint mode if active
                isTestPaused = false;
                stopCurrentAnimation(); 
                stopPaintMode(); // Ensure paint mode is off
                
                updateButtonStates(); 
                
                testAllServosMessageDiv.style.display = 'block';
                testAllServosMessageDiv.textContent = "Moving all active servos to mean position (90 degrees)...";

                for (const tileId in servoSettings) {
                    if (servoSettings[tileId].isActive) {
                        const numericId = servoSettings[tileId].numericId;
                        servoSettings[tileId].moveToPosition = 90;
                        updateTileColor(tileId);
                        sendDataViaWebSocket('moveToPosition', 90, numericId);
                    }
                }
                await delay(2000); 

                activeNumericIds = Array.from(numericIdToServoIdMap.keys()).filter(numericId => {
                    const tileId = numericIdToServoIdMap.get(numericId);
                    return servoSettings[tileId] && servoSettings[tileId].isActive;
                }).sort((a,b) => a - b); 

                currentTestServoIndex = 0; 

                while (isTestAllServosMode && currentTestServoIndex < activeNumericIds.length) {
                    if (isTestPaused) {
                        await waitForResume(); 
                        if (!isTestAllServosMode) break; 
                    }
                    
                    const numericId = activeNumericIds[currentTestServoIndex];
                    const tileId = numericIdToServoIdMap.get(numericId);
                    const tileElement = document.querySelector(`.servo-tile[data-tile-id="${tileId}"]`);
                    const servo = servoSettings[tileId];

                    if (tileElement && servo) {
                        currentTestingServoTileId = tileId;
                        tileElement.classList.add('highlighted');
                        testAllServosMessageDiv.textContent = `Testing Servo ID: ${numericId}`;
                        
                        servo.moveToPosition = servo.lowerLimit;
                        updateTileColor(tileId);
                        sendDataViaWebSocket('moveToPosition', servo.lowerLimit, numericId);
                        try { await delay(TEST_SERVO_INTERVAL_MS); } catch (e) { } 
                        if (!isTestAllServosMode) break; 
                        if (isTestPaused) { await waitForResume(); if (!isTestAllServosMode) break; }

                        servo.moveToPosition = servo.upperLimit;
                        updateTileColor(tileId);
                        sendDataViaWebSocket('moveToPosition', servo.upperLimit, numericId);
                        try { await delay(TEST_SERVO_INTERVAL_MS); } catch (e) { } 
                        if (!isTestAllServosMode) break; 
                        if (isTestPaused) { await waitForResume(); if (!isTestAllServosMode) break; }

                        servo.moveToPosition = 90;
                        updateTileColor(tileId);
                        sendDataViaWebSocket('moveToPosition', 90, numericId);
                        try { await delay(TEST_SERVO_INTERVAL_MS); } catch (e) { } 
                        if (!isTestAllServosMode) break; 
                        if (isTestPaused) { await waitForResume(); if (!isTestAllServosMode) break; }

                        tileElement.classList.remove('highlighted');
                    }
                    if (testStepPromiseResolve === null && testStepPromiseReject === null && !isTestPaused) {
                        currentTestServoIndex++; 
                    } else {
                        testStepPromiseResolve = null;
                        testStepPromiseReject = null;
                    }
                }
                if (isTestAllServosMode) { 
                    stopTestAllServosAnimation(); 
                    testAllServosMessageDiv.textContent = "All servos test complete.";
                    testAllServosMessageDiv.style.display = 'block';
                }
            }

            function pauseTestAllServosAnimation() {
                isTestPaused = true;
                playIcon.style.display = 'block'; 
                pauseIcon.style.display = 'none'; 
                testAllServosMessageDiv.textContent = "Test Paused.";
                if (testStepPromiseResolve) { 
                    testStepPromiseResolve();
                }
            }

            function resumeTestAllServosAnimation() {
                isTestPaused = false;
                playIcon.style.display = 'none'; 
                pauseIcon.style.display = 'block'; 
                testAllServosMessageDiv.textContent = "Resuming Test...";
                if (resolvePausePromise) {
                    resolvePausePromise(); 
                    resolvePausePromise = null;
                }
            }

            function stopTestAllServosAnimation() {
                isTestAllServosMode = false;
                isTestPaused = false;
                currentTestServoIndex = 0; 
                if (currentTestingServoTileId) {
                    const prevTile = document.querySelector(`.servo-tile[data-tile-id="${currentTestingServoTileId}"]`);
                    if (prevTile) prevTile.classList.remove('highlighted');
                    currentTestingServoTileId = null;
                }
                testAllServosMessageDiv.textContent = "Test All Servos stopped.";
                setTimeout(() => { testAllServosMessageDiv.style.display = 'none'; }, 3000); 
                updateButtonStates(); 

                if (resolvePausePromise) { 
                    resolvePausePromise();
                    resolvePausePromise = null;
                }
                if (testStepPromiseReject) { 
                    testStepPromiseReject("Test stopped");
                }
            }

            function delay(ms, signal = null) {
                return new Promise((resolve, reject) => {
                    const timeoutId = setTimeout(() => {
                        testStepPromiseResolve = null;
                        testStepPromiseReject = null;
                        resolve();
                    }, ms);

                    if (signal) {
                        signal.addEventListener('abort', () => {
                            clearTimeout(timeoutId);
                            reject(new Error("Operation aborted")); 
                            testStepPromiseResolve = null;
                            testStepPromiseReject = null;
                        }, { once: true });
                    }

                    testStepPromiseResolve = () => {
                        clearTimeout(timeoutId);
                        resolve();
                        testStepPromiseResolve = null;
                        testStepPromiseReject = null;
                    };
                    testStepPromiseReject = (reason) => {
                        clearTimeout(timeoutId);
                        reject(reason);
                        testStepPromiseResolve = null;
                        testStepPromiseReject = null;
                    };
                });
            }

            function waitForResume() {
                return new Promise(resolve => {
                    resolvePausePromise = resolve;
                });
            }

            // --- Paint Mode Logic ---
            paintModeButton.addEventListener('click', () => {
                if (isPaintMode) {
                    stopPaintMode();
                } else {
                    startPaintMode();
                }
            });

            function startPaintMode() {
                isPaintMode = true;
                isInteractiveSetupMode = false;
                isTestAllServosMode = false;
                stopCurrentAnimation();
                stopTestAllServosAnimation();
                stopInteractiveSetup(); // Ensure interactive setup is off
                paintModeButton.textContent = "Exit Paint Mode";
                updateButtonStates();
                // Optional: Move all active servos to a default paint position (e.g., 90)
                Object.keys(servoSettings).forEach(tileId => {
                    if (servoSettings[tileId].isActive) {
                        servoSettings[tileId].moveToPosition = 90; // Or a specific 'paint' position
                        updateTileColor(tileId);
                        sendDataViaWebSocket('moveToPosition', 90, servoSettings[tileId].numericId);
                    }
                });
            }

            function stopPaintMode() {
                isPaintMode = false;
                paintModeButton.textContent = "Paint Mode";
                updateButtonStates();
            }

            function handlePaintModeClick(tileId) {
                if (!servoSettings[tileId].isActive) {
                    console.log(`Servo ${tileId} is inactive, cannot paint.`);
                    return;
                }
                // Toggle position between two states, e.g., 0 and 180
                const currentPos = servoSettings[tileId].moveToPosition;
                const newPos = (currentPos === 0) ? 180 : 0; // Or 90 and 180, or 0 and 90
                servoSettings[tileId].moveToPosition = newPos;
                updateTileColor(tileId);
                sendDataViaWebSocket('moveToPosition', newPos, servoSettings[tileId].numericId);
            }


            // Centralized function to manage button states based on active mode
            function updateButtonStates() {
                const isAnyModeActive = isInteractiveSetupMode || isTestAllServosMode || isPaintMode; 

                waveMotionButton.disabled = isAnyModeActive;
                pulseMotionButton.disabled = isAnyModeActive;
                sweepMotionButton.disabled = isAnyModeActive;
                rippleMotionButton.disabled = isAnyModeActive;
                setFpsButton.disabled = isAnyModeActive;
                configureAllButton.disabled = isAnyModeActive;

                interactiveSetupButton.disabled = isTestAllServosMode || isPaintMode;
                startTestButton.disabled = isInteractiveSetupMode || isPaintMode; 
                paintModeButton.disabled = isInteractiveSetupMode || isTestAllServosMode; // Disable if other modes are active
                
                // New Toggle Buttons
                toggleIdVisibilityButton.disabled = isAnyModeActive;
                toggleHiddenModulesButton.disabled = isAnyModeActive;

                startTestButton.style.display = isTestAllServosMode ? 'none' : 'block';
                testControlIcons.style.display = isTestAllServosMode ? 'flex' : 'none';
                exitTestButton.style.display = isTestAllServosMode ? 'block' : 'none';

                if (isTestAllServosMode) {
                    playIcon.style.display = isTestPaused ? 'block' : 'none';
                    pauseIcon.style.display = isTestPaused ? 'none' : 'block';
                } else {
                    playIcon.style.display = 'block'; 
                    pauseIcon.style.display = 'none';
                }

                const isPopupOpen = popup.style.display === 'flex';
                sendBoardPinButton.disabled = isAnyModeActive || !isPopupOpen;
                sendLowerLimitButton.disabled = isAnyModeActive || !isPopupOpen;
                sendUpperLimitButton.disabled = isAnyModeActive || !isPopupOpen;
                sendToPositionButton.disabled = isAnyModeActive || !isPopupOpen;
                testButton.disabled = isAnyModeActive || !isPopupOpen;

                // Interactive Setup specific button visibility
                interactiveSetupMessage.style.display = isInteractiveSetupMode ? 'block' : 'none'; 
                // The container is now always flex when in interactiveSetupMode
                interactiveSetupButtonsContainer.style.display = isInteractiveSetupMode ? 'flex' : 'none';
                
                // Individual repeat/back buttons enabled only when a servo is blinking
                const isServoBlinking = currentBlinkingPhysicalServo.board !== -1;
                repeatBlinkServoButton.style.display = isInteractiveSetupMode ? 'inline-block' : 'none';
                previousServoButton.style.display = isInteractiveSetupMode ? 'inline-block' : 'none';
                
                // Control disabled state based on whether a servo is blinking
                repeatBlinkServoButton.disabled = !isServoBlinking;
                previousServoButton.disabled = !isServoBlinking;
                
                testAllServosMessageDiv.style.display = isTestAllServosMode ? 'block' : 'none';
            }

            // Event listeners for new toggle buttons
            toggleIdVisibilityButton.addEventListener('click', () => {
                showModuleIds = !showModuleIds;
                toggleIdVisibilityButton.textContent = showModuleIds ? "Hide Module IDs" : "Show Module IDs";
                // Re-render all tiles to apply ID visibility changes
                Object.keys(servoSettings).forEach(tileId => updateTileColor(tileId));
            });

            toggleHiddenModulesButton.addEventListener('click', () => {
                showHiddenModules = !showHiddenModules;
                toggleHiddenModulesButton.textContent = showHiddenModules ? "Hide Inactive Modules" : "Show Inactive Modules";
                // Re-render all tiles to apply display changes
                Object.keys(servoSettings).forEach(tileId => updateTileColor(tileId));
            });


            generateGrid();
            connectWebSocket(); 
            updateButtonStates(); 

            // Set initial button text for new toggles
            toggleIdVisibilityButton.textContent = showModuleIds ? "Hide Module IDs" : "Show Module IDs";
            toggleHiddenModulesButton.textContent = showHiddenModules ? "Hide Inactive Modules" : "Show Inactive Modules";

            window.addEventListener('resize', generateGrid);
        });
    </script>
</body>
</html>
)rawliteral";

// --- Servo Control Functions Implementation ---

// Initializes all PCA9685 drivers
void bloom_init() {
    unresponsive_boards.clear(); // Clear previous state
    for (int i = 0; i < NO_OF_BOARD; i++) {
        // Only attempt to initialize if the board exists at the address
        Wire.beginTransmission(pca9685_addresses[i]);
        if (Wire.endTransmission() == 0) { // 0 means success
            pwm_drivers[i] = Adafruit_PWMServoDriver(pca9685_addresses[i]);
            pwm_drivers[i].begin();
            pwm_drivers[i].setPWMFreq(60); // Standard servo frequency (60Hz)
            // Serial.printf("PCA9685 driver at 0x%02X initialized.\n", pca9685_addresses[i]); // Too verbose
        } else {
            Serial.printf("No PCA9685 driver found at 0x%02X. Skipping initialization.\n", pca9685_addresses[i]);
            unresponsive_boards.push_back(i); // Add unresponsive board index
        }
    }

    // After attempting to initialize all boards, send status to frontend
    DynamicJsonDocument doc(256); // Small enough for this message
    doc["type"] = "i2cStatus";
    JsonArray unresponsiveArray = doc.createNestedArray("unresponsiveBoards");
    for (uint8_t board_idx : unresponsive_boards) {
        unresponsiveArray.add(board_idx);
    }
    String response;
    serializeJson(doc, response);
    webSocket.broadcastTXT(response);
}

// Loads all servo configurations from EEPROM into RAM
void load_all_servo_configs_from_eeprom() {
    // Initialize EEPROM with enough space for the version byte + all servo data
    EEPROM.begin(EEPROM_DATA_START_ADDRESS + sizeof(motor_eeprom_data));

    byte eeprom_version = EEPROM.read(EEPROM_VERSION_ADDRESS);

    if (eeprom_version == EEPROM_CURRENT_VERSION) {
        // If version matches, load existing data
        EEPROM.get(EEPROM_DATA_START_ADDRESS, motor_eeprom_data);
        Serial.println("Servo configurations loaded from EEPROM.");

        // Populate RAM struct from EEPROM data
        for (int i = 0; i < NO_OF_MODULE; i++) {
            all_servos_ram[i].lower_limit_pwm = angle_to_pwm_pulse(motor_eeprom_data.lower_limit_angle[i]);
            all_servos_ram[i].upper_limit_pwm = angle_to_pwm_pulse(motor_eeprom_data.upper_limit_angle[i]);
            all_servos_ram[i].board_idx = motor_eeprom_data.board_idx[i];
            all_servos_ram[i].servo_pin_idx = motor_eeprom_data.servo_pin_idx[i];
            all_servos_ram[i].current_pwm_value = angle_to_pwm_pulse(90); // Default to center on startup
        }
    } else {
        // If EEPROM is empty or version mismatch, initialize with defaults
        Serial.println("EEPROM empty or invalid version. Initializing default servo configurations.");
        for (int i = 0; i < NO_OF_MODULE; i++) {
            motor_eeprom_data.lower_limit_angle[i] = 0;   // Default lower angle
            motor_eeprom_data.upper_limit_angle[i] = 180; // Default upper angle
            motor_eeprom_data.board_idx[i] = i / NO_OF_SERVOS_IN_A_BOARD; // Default board assignment
            motor_eeprom_data.servo_pin_idx[i] = i % NO_OF_SERVOS_IN_A_BOARD; // Default pin assignment

            // Also populate RAM struct with these defaults
            all_servos_ram[i].lower_limit_pwm = angle_to_pwm_pulse(0);
            all_servos_ram[i].upper_limit_pwm = angle_to_pwm_pulse(180);
            all_servos_ram[i].board_idx = motor_eeprom_data.board_idx[i];
            all_servos_ram[i].servo_pin_idx = motor_eeprom_data.servo_pin_idx[i];
            all_servos_ram[i].current_pwm_value = angle_to_pwm_pulse(90); // Default to center on startup
        }
        // Save these default configurations to EEPROM and write version
        EEPROM.put(EEPROM_DATA_START_ADDRESS, motor_eeprom_data);
        EEPROM.write(EEPROM_VERSION_ADDRESS, EEPROM_CURRENT_VERSION);
        EEPROM.commit();
        Serial.println("Default servo configurations saved to EEPROM.");
    }
}

// Saves a single servo's configuration to EEPROM, only if changed.
void save_single_servo_config_to_eeprom(int numeric_servo_id, int board, int pin, int lower_limit, int upper_limit) {
    if (numeric_servo_id < 0 || numeric_servo_id >= NO_OF_MODULE) {
        Serial.println("Invalid numeric_servo_id for saving single config.");
        return;
    }

    bool changed = false;
    // Compare new values with current values in EEPROM struct before updating
    if (motor_eeprom_data.board_idx[numeric_servo_id] != (uint8_t)board) {
        motor_eeprom_data.board_idx[numeric_servo_id] = (uint8_t)board;
        all_servos_ram[numeric_servo_id].board_idx = (uint8_t)board;
        changed = true;
    }
    if (motor_eeprom_data.servo_pin_idx[numeric_servo_id] != (uint8_t)pin) {
        motor_eeprom_data.servo_pin_idx[numeric_servo_id] = (uint8_t)pin;
        all_servos_ram[numeric_servo_id].servo_pin_idx = (uint8_t)pin;
        changed = true;
    }
    if (motor_eeprom_data.lower_limit_angle[numeric_servo_id] != (uint8_t)lower_limit) {
        motor_eeprom_data.lower_limit_angle[numeric_servo_id] = (uint8_t)lower_limit;
        all_servos_ram[numeric_servo_id].lower_limit_pwm = angle_to_pwm_pulse(lower_limit);
        changed = true;
    }
    if (motor_eeprom_data.upper_limit_angle[numeric_servo_id] != (uint8_t)upper_limit) {
        motor_eeprom_data.upper_limit_angle[numeric_servo_id] = (uint8_t)upper_limit;
        all_servos_ram[numeric_servo_id].upper_limit_pwm = angle_to_pwm_pulse(upper_limit);
        changed = true;
    }

    if (changed) {
        EEPROM.put(EEPROM_DATA_START_ADDRESS, motor_eeprom_data);
        EEPROM.commit();
        Serial.printf("Servo %d config updated in EEPROM.\n", numeric_servo_id);
    }
}

// Converts angle (SERVO_MIN_ANGLE-SERVO_MAX_ANGLE) to raw PWM pulse length (PWM_MIN-PWM_MAX)
int angle_to_pwm_pulse(int angle) {
    return map(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE, PWM_MIN, PWM_MAX);
}

// Converts raw PWM pulse length (PWM_MIN-PWM_MAX) to angle (SERVO_MIN_ANGLE-SERVO_MAX_ANGLE)
int pwm_pulse_to_angle(int pwm_pulse) {
    return map(pwm_pulse, PWM_MIN, PWM_MAX, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
}

// Maps a desired angle (SERVO_MIN_ANGLE-SERVO_MAX_ANGLE) to a PWM pulse, respecting the servo's configured pwm limits.
int map_angle_to_constrained_pwm(int servo_idx, int angle) {
    return map(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE, all_servos_ram[servo_idx].lower_limit_pwm, all_servos_ram[servo_idx].upper_limit_pwm);
}

// Maps a constrained PWM pulse back to a (SERVO_MIN_ANGLE-SERVO_MAX_ANGLE) angle.
int map_constrained_pwm_to_angle(int servo_idx, int pwm) {
    return map(pwm, all_servos_ram[servo_idx].lower_limit_pwm, all_servos_ram[servo_idx].upper_limit_pwm, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
}

// Sets a specific servo to a target PWM pulse.
void set_servo_pwm_pulse(int servo_idx, int pulse) {
    if (servo_idx < 0 || servo_idx >= NO_OF_MODULE) {
        // Serial.printf("Error: Invalid servo_idx %d for set_servo_pwm_pulse.\n", servo_idx);
        return;
    }
    uint8_t board = all_servos_ram[servo_idx].board_idx;
    uint8_t pin = all_servos_ram[servo_idx].servo_pin_idx;

    if (board >= NO_OF_BOARD || pin >= NO_OF_SERVOS_IN_A_BOARD) {
        // Serial.printf("Error: Invalid board/pin for servo %d (Board: %u, Pin: %u). Cannot set PWM.\n", servo_idx, board, pin);
        return;
    }

    Wire.beginTransmission(pca9685_addresses[board]);
    if (Wire.endTransmission() == 0) { // Check if the board is responsive
        pwm_drivers[board].setPWM(pin, 0, pulse);
    } else {
        // Serial.printf("Warning: PCA9685 board at 0x%02X not responsive for servo %d. Cannot set PWM.\n", pca9685_addresses[board], servo_idx);
    }
}

// Iterates through all servos in RAM and applies their current_pwm_value
void update_all_servos_from_ram() {
    for (int i = 0; i < NO_OF_MODULE; i++) {
        set_servo_pwm_pulse(i, all_servos_ram[i].current_pwm_value);
    }
}

// Starts the interactive setup process
void start_interactive_setup() {
    interactive_setup_running = true;
    current_physical_servo_index = 0; // Start from the first physical servo (0-indexed)
    Serial.println("Starting interactive setup...");

    // Move all servos to the mean position before starting the blink sequence
    for (int i = 0; i < NO_OF_MODULE; i++) {
        set_servo_pwm_pulse(i, angle_to_pwm_pulse(interactive_setup_mean_position));
    }
    delay(500); // Give time for servos to reach position

    // Immediately try to find and blink the first responsive servo
    next_interactive_servo(); 
}

// Moves to the next physical servo in the interactive setup sequence (non-recursive)
void next_interactive_servo() {
    if (!interactive_setup_running) return;

    // Loop to find the next responsive physical servo
    // Note: current_physical_servo_index can go up to (NO_OF_BOARD * NO_OF_SERVOS_IN_A_BOARD - 1)
    // which is 111 for 7 boards * 16 pins.
    while (current_physical_servo_index < (NO_OF_BOARD * NO_OF_SERVOS_IN_A_BOARD)) {
        int board = current_physical_servo_index / NO_OF_SERVOS_IN_A_BOARD;
        int pin = current_physical_servo_index % NO_OF_SERVOS_IN_A_BOARD;

        // Check if the board is responsive
        Wire.beginTransmission(pca9685_addresses[board]);
        if (Wire.endTransmission() == 0) { // If board is responsive
            // Perform blinking motion for the current physical servo
            pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position - interactive_setup_test_range));
            delay(500);
            pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position + interactive_setup_test_range));
            delay(500);
            pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position));

            // Send the physical board and pin info to the frontend
            DynamicJsonDocument doc(128);
            doc["type"] = "blinkingServoInfo";
            doc["board"] = board;
            doc["pin"] = pin;
            String response;
            serializeJson(doc, response);
            webSocket.broadcastTXT(response);
            Serial.printf("Blinking physical servo: Board %d, Pin %d (physical index %d)\n", board, pin, current_physical_servo_index);
            return; // Found a servo to blink, exit function and wait for frontend interaction
        } else {
            Serial.printf("PCA9685 board at 0x%02X not responsive. Skipping to next physical servo.\n", pca9685_addresses[board]);
            current_physical_servo_index++; // Increment to check the next physical servo
            // Loop continues to the next iteration
        }
    }

    // If the loop finishes, it means all physical servos have been checked (or skipped)
    // and no responsive board was found, or we've reached the end of the sequence.
    interactive_setup_running = false;
    DynamicJsonDocument doc(64);
    doc["type"] = "interactiveSetupComplete";
    String response;
    serializeJson(doc, response);
    webSocket.broadcastTXT(response);
    Serial.println("Interactive setup complete (or no more responsive servos found).");
    // Move all servos to the mean position after setup
    for(int i=0; i<NO_OF_MODULE; i++) {
        all_servos_ram[i].current_pwm_value = angle_to_pwm_pulse(interactive_setup_mean_position);
    }
    update_all_servos_from_ram();
}

// Stops the interactive setup process
void stop_interactive_setup() {
    interactive_setup_running = false;
    current_physical_servo_index = 0; // Reset index
    Serial.println("Interactive setup manually stopped.");
    // Optionally move all servos back to a default state (e.g., 90 degrees)
    for(int i=0; i<NO_OF_MODULE; i++) {
        all_servos_ram[i].current_pwm_value = angle_to_pwm_pulse(90);
    }
    update_all_servos_from_ram();
    DynamicJsonDocument doc(64);
    doc["type"] = "interactiveSetupComplete"; // Use same message type to signal end
    String response;
    serializeJson(doc, response);
    webSocket.broadcastTXT(response);
}

// --- I2C Scanner Function ---
void i2c_scanner() {
    byte error, address;
    int nDevices;

    Serial.println("Scanning I2C addresses...");
    nDevices = 0;
    for (address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();

        if (error == 0) {
            Serial.printf("I2C device found at address 0x%02X\n", address);
            nDevices++;
        } else if (error == 4) {
            Serial.printf("Unknown error at address 0x%02X\n", address);
        }
    }
    if (nDevices == 0) {
        Serial.println("No I2C devices found.\n");
    } else {
        Serial.printf("Scan complete. Found %d I2C devices.\n", nDevices);
    }
}


// --- WebSocket Event Handler ---
void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length)
{
    switch (type)
    {
    case WStype_DISCONNECTED:
    {
        Serial.printf("[%u] Disconnected!\n", num);
        break;
    }
    case WStype_CONNECTED:
    {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d url: %s\n", num, ip[0], ip[1], ip[2], ip[3], payload);
        webSocket.sendTXT(num, "{\"type\":\"connected\"}"); // Send connection confirmation
        break;
    }
    case WStype_TEXT:
    {
        DynamicJsonDocument doc(8192); // Adjusted size for larger messages
        DeserializationError error = deserializeJson(doc, payload);

        if (error)
        {
            Serial.print(F("deserializeJson() failed: "));
            Serial.println(error.f_str());
            webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid JSON format\"}");
            return;
        }

        const char *msg_type = doc["type"];
        int numeric_servo_id = doc["numericServoId"].as<int>(); 

        if (strcmp(msg_type, "fetchLimits") == 0)
        {
            DynamicJsonDocument responseDoc(256);
            responseDoc["type"] = "data_fetched";
            responseDoc["numericServoId"] = numeric_servo_id;

            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE)
            {
                responseDoc["lowerLimit"] = motor_eeprom_data.lower_limit_angle[numeric_servo_id];
                responseDoc["upperLimit"] = motor_eeprom_data.upper_limit_angle[numeric_servo_id];
                responseDoc["boardNumber"] = motor_eeprom_data.board_idx[numeric_servo_id];
                responseDoc["pinNumber"] = motor_eeprom_data.servo_pin_idx[numeric_servo_id];
            }
            else
            {
                responseDoc["lowerLimit"] = 0; // Default to safe values
                responseDoc["upperLimit"] = 180;
                responseDoc["boardNumber"] = 0; 
                responseDoc["pinNumber"] = 0;   
                Serial.printf("  Warning: fetchLimits requested for out-of-bounds servo ID: %d\n", numeric_servo_id);
            }

            String response;
            serializeJson(responseDoc, response);
            webSocket.sendTXT(num, response);
        }
        else if (strcmp(msg_type, "setBoardPin") == 0)
        {
            int board_value = doc["value"]["board"].as<int>();
            int pin_value = doc["value"]["pin"].as<int>();
            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE && board_value >= 0 && board_value < NO_OF_BOARD && pin_value >=0 && pin_value < NO_OF_SERVOS_IN_A_BOARD)
            {
                save_single_servo_config_to_eeprom(numeric_servo_id, board_value, pin_value, motor_eeprom_data.lower_limit_angle[numeric_servo_id], motor_eeprom_data.upper_limit_angle[numeric_servo_id]);
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Board/Pin set and saved\"}");
            }
            else
            {
                Serial.printf("  Invalid servo ID, board, or pin for setBoardPin: %d, board %d, pin %d\n", numeric_servo_id, board_value, pin_value);
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid servo ID, board, or pin\"}");
            }
        }
        else if (strcmp(msg_type, "lowerLimit") == 0)
        {
            int value = doc["value"].as<int>();
            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE && value >= SERVO_MIN_ANGLE && value <= SERVO_MAX_ANGLE)
            {
                save_single_servo_config_to_eeprom(numeric_servo_id, all_servos_ram[numeric_servo_id].board_idx, all_servos_ram[numeric_servo_id].servo_pin_idx, value, motor_eeprom_data.upper_limit_angle[numeric_servo_id]);
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"lowerLimit acknowledged and saved\"}");
            }
            else
            {
                Serial.printf("  Invalid servo ID or limit for lowerLimit: %d, value %d\n", numeric_servo_id, value);
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid servo ID or limit\"}");
            }
        }
        else if (strcmp(msg_type, "upperLimit") == 0)
        {
            int value = doc["value"].as<int>();
            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE && value >= SERVO_MIN_ANGLE && value <= SERVO_MAX_ANGLE)
            {
                save_single_servo_config_to_eeprom(numeric_servo_id, all_servos_ram[numeric_servo_id].board_idx, all_servos_ram[numeric_servo_id].servo_pin_idx, motor_eeprom_data.lower_limit_angle[numeric_servo_id], value);
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"upperLimit acknowledged and saved\"}");
            }
            else
            {
                Serial.printf("  Invalid servo ID or limit for upperLimit: %d, value %d\n", numeric_servo_id, value);
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid servo ID or limit\"}");
            }
        }
        else if (strcmp(msg_type, "moveToPosition") == 0)
        {
            int value = doc["value"].as<int>();
            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE && value >= SERVO_MIN_ANGLE && value <= SERVO_MAX_ANGLE)
            {
                all_servos_ram[numeric_servo_id].current_pwm_value = map_angle_to_constrained_pwm(numeric_servo_id, value);
                set_servo_pwm_pulse(numeric_servo_id, all_servos_ram[numeric_servo_id].current_pwm_value); // Move the servo
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"moveToPosition acknowledged\"}");
            }
            else
            {
                Serial.printf("  Invalid servo ID or position for moveToPosition: %d, value %d\n", numeric_servo_id, value);
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid servo ID or position\"}");
            }
        }
        else if (strcmp(msg_type, "test") == 0)
        {
            int lower = doc["value"]["lower"].as<int>();
            int upper = doc["value"]["upper"].as<int>();
            if (numeric_servo_id >= 0 && numeric_servo_id < NO_OF_MODULE && lower >= SERVO_MIN_ANGLE && lower <= SERVO_MAX_ANGLE && upper >= SERVO_MIN_ANGLE && upper <= SERVO_MAX_ANGLE)
            {
                Serial.printf("  Received test command for servo %d: Lower=%d, Upper=%d\n", numeric_servo_id, lower, upper);
                // Perform the full test swing on the Arduino
                set_servo_pwm_pulse(numeric_servo_id, map_angle_to_constrained_pwm(numeric_servo_id, lower));
                delay(500);
                set_servo_pwm_pulse(numeric_servo_id, map_angle_to_constrained_pwm(numeric_servo_id, upper));
                delay(500);
                set_servo_pwm_pulse(numeric_servo_id, map_angle_to_constrained_pwm(numeric_servo_id, 90)); // Return to center
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"test acknowledged\"}");
            }
            else
            {
                Serial.printf("  Invalid servo ID or limits for test: %d, lower %d, upper %d\n", numeric_servo_id, lower, upper);
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid servo ID or limits\"}");
            }
        }
        else if (strcmp(msg_type, "fetchAllLimits") == 0)
        {
            DynamicJsonDocument responseDoc(8192); 
            responseDoc["type"] = "all_data_fetched";
            JsonArray servosArray = responseDoc.createNestedArray("servos");

            for (int i = 0; i < NO_OF_MODULE; i++) {
                JsonObject servoObj = servosArray.createNestedObject();
                servoObj["numericServoId"] = i;
                servoObj["boardNumber"] = all_servos_ram[i].board_idx;
                servoObj["pinNumber"] = all_servos_ram[i].servo_pin_idx;
                servoObj["lowerLimit"] = motor_eeprom_data.lower_limit_angle[i];
                servoObj["upperLimit"] = motor_eeprom_data.upper_limit_angle[i];
            }

            String response;
            serializeJson(responseDoc, response);
            webSocket.sendTXT(num, response);
            Serial.println("Sent all servo data to frontend.");
        }
        else if (strcmp(msg_type, "batchUpdateConfig") == 0)
        {
            JsonArray importedServos = doc["value"].as<JsonArray>();
            for (JsonObject servo : importedServos) {
                int id = servo["numericServoId"];
                int board = servo["boardNumber"];
                int pin = servo["pinNumber"];
                int lower = servo["lowerLimit"];
                int upper = servo["upperLimit"];
                
                if (id >= 0 && id < NO_OF_MODULE && board >= 0 && board < NO_OF_BOARD && pin >= 0 && pin < NO_OF_SERVOS_IN_A_BOARD && lower >= SERVO_MIN_ANGLE && lower <= SERVO_MAX_ANGLE && upper >= SERVO_MIN_ANGLE && upper <= SERVO_MAX_ANGLE) {
                    save_single_servo_config_to_eeprom(id, board, pin, lower, upper);
                } else {
                    Serial.printf("  Warning: Invalid data in batch update for servo ID %d. Skipping.\n", id);
                }
            }
            webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Batch update complete\"}");
            Serial.println("Batch update from frontend processed.");
        }
        else if (strcmp(msg_type, "startInteractiveSetup") == 0)
        {
            interactive_setup_mean_position = doc["value"]["meanPosition"].as<int>();
            interactive_setup_test_range = doc["value"]["testRange"].as<int>();
            start_interactive_setup(); // Call the function to begin the sequence
            webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Interactive setup started\"}");
        }
        else if (strcmp(msg_type, "mapServo") == 0)
        {
            if (interactive_setup_running) {
                int id = doc["numericServoId"].as<int>();
                int board = doc["boardNumber"].as<int>();
                int pin = doc["pinNumber"].as<int>();
                int lower = doc["lowerLimit"].as<int>(); 
                int upper = doc["upperLimit"].as<int>(); 
                
                if (id >= 0 && id < NO_OF_MODULE && board >= 0 && board < NO_OF_BOARD && pin >= 0 && pin < NO_OF_SERVOS_IN_A_BOARD && lower >= SERVO_MIN_ANGLE && lower <= SERVO_MAX_ANGLE && upper >= SERVO_MIN_ANGLE && upper <= SERVO_MAX_ANGLE) {
                    save_single_servo_config_to_eeprom(id, board, pin, lower, upper);
                    Serial.printf("Mapped numeric ID %d to physical Board %d, Pin %d\n", id, board, pin);
                    webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Servo mapped\"}");
                } else {
                    Serial.printf("  Warning: Invalid data in mapServo command for servo ID %d. Skipping.\n", id);
                    webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Invalid mapServo data\"}");
                }
            } else {
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Interactive setup not running\"}");
            }
        }
        else if (strcmp(msg_type, "nextServo") == 0)
        {
            if (interactive_setup_running) {
                current_physical_servo_index++; // Increment the index to move to the next physical servo
                next_interactive_servo(); // Call to find and blink the next responsive servo
                webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Next servo requested\"}");
            } else {
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Interactive setup not running\"}");
            }
        }
        else if (strcmp(msg_type, "reBlinkServo") == 0)
        {
            // Re-blink the current servo without advancing the index
            if (interactive_setup_running) {
                int board = current_physical_servo_index / NO_OF_SERVOS_IN_A_BOARD;
                int pin = current_physical_servo_index % NO_OF_SERVOS_IN_A_BOARD;
                
                Wire.beginTransmission(pca9685_addresses[board]);
                if (Wire.endTransmission() == 0) { // If board is responsive
                    pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position - interactive_setup_test_range));
                    delay(500);
                    pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position + interactive_setup_test_range));
                    delay(500);
                    pwm_drivers[board].setPWM(pin, 0, angle_to_pwm_pulse(interactive_setup_mean_position));

                    DynamicJsonDocument doc(128);
                    doc["type"] = "blinkingServoInfo";
                    doc["board"] = board;
                    doc["pin"] = pin;
                    String response;
                    serializeJson(doc, response);
                    webSocket.broadcastTXT(response);
                    Serial.printf("Re-blinking physical servo: Board %d, Pin %d (physical index %d)\n", board, pin, current_physical_servo_index);
                } else {
                    Serial.printf("Re-blink failed: PCA9685 board at 0x%02X not responsive.\n", pca9685_addresses[board]);
                    webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Re-blink failed: Board not responsive\"}");
                }
            } else {
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Interactive setup not running\"}");
            }
        }
        else if (strcmp(msg_type, "previousServo") == 0)
        {
            // Move back to the previous servo
            if (interactive_setup_running) {
                if (current_physical_servo_index > 0) {
                    current_physical_servo_index--; // Decrement the index
                    Serial.printf("Moving to previous physical servo index: %d\n", current_physical_servo_index);
                    next_interactive_servo(); // Call to find and blink the new (previous) servo
                    webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Previous servo requested\"}");
                } else {
                    Serial.println("Already at the first physical servo.");
                    webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Already at first servo\"}");
                }
            } else {
                webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Interactive setup not running\"}");
            }
        }
        else if (strcmp(msg_type, "stopInteractiveSetup") == 0)
        {
            stop_interactive_setup();
            webSocket.sendTXT(num, "{\"type\":\"received\", \"status\":\"Interactive setup stopped\"}");
        }
        else
        {
            Serial.printf("  Unknown command type: %s\n", msg_type);
            webSocket.sendTXT(num, "{\"type\":\"error\", \"message\":\"Unknown command type\"}");
        }
        break;
    }
    case WStype_BIN:
    {
        if (length == NO_OF_MODULE) { // Ensure the binary payload matches expected size
            for (size_t i = 0; i < length; i++) {
                uint8_t numeric_id = i;
                uint8_t position_angle = payload[i]; // Value is the angle (0-180)

                if (numeric_id < NO_OF_MODULE) { // Ensure within bounds
                    all_servos_ram[numeric_id].current_pwm_value =  map_angle_to_constrained_pwm(numeric_id, position_angle);
                }
            }
            update_all_servos_from_ram(); 
        } else {
            Serial.printf("Received binary stream with unexpected length: %u (expected %u)\n", length, NO_OF_MODULE);
        }
        break;
    }
    case WStype_ERROR:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT_FIN:
    {
        break;
    }
    }
}

// Handler for the root path ("/") - serves the embedded HTML content
void handleRoot() {
    server.send_P(200, "text/html", html_content);
}

// Handler for any unknown requests (captive portal redirection)
void handleNotFound() {
    if (server.method() == HTTP_GET) {
        Serial.println("Redirecting to captive portal...");
        server.sendHeader("Location", String("http://") + WiFi.softAPIP().toString());
        server.send(302, "text/plain", ""); // 302 Found (Temporary Redirect)
    } else {
        server.send(404, "text/plain", "404 Not Found");
    }
}

void setup() {
    Serial.begin(115200);
    Serial.println();
    Serial.println("Starting ESP32 Servo Controller...");

    // Set up ESP32 as an Access Point
    WiFi.softAP(ap_ssid, ap_password);
    IPAddress myIP = WiFi.softAPIP();
    Serial.print("AP IP Address: ");
    Serial.println(myIP);

    // Start DNS server for captive portal
    dnsServer.start(53, "*", myIP);
    Serial.println("DNS server started.");

    // Start Web Server
    server.on("/", handleRoot);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("HTTP server started.");

    // Start WebSocket server
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
    Serial.printf("WebSocket server started on ws://%s:%u\n", myIP.toString().c_str(), 8765);

    // Initialize I2C communication
    Wire.begin(); 
    // Run I2C scanner to help diagnose hardware issues
    i2c_scanner();

    // Initialize PCA9685 drivers (now includes a check for responsiveness and reports to frontend)
    bloom_init();
    Serial.println("Attempted PCA9685 drivers initialization.");

    // Read EEPROM for servo limits and addresses
    load_all_servo_configs_from_eeprom();
    Serial.println("EEPROM data loaded/initialized.");

    // Move all servos to their initial current_pwm_value (e.g., 90 degrees)
    update_all_servos_from_ram();
    Serial.println("Servos moved to initial positions.");
}

void loop() {
    dnsServer.processNextRequest(); // Process DNS requests (crucial for captive portal)
    server.handleClient();          // Process HTTP client requests
    webSocket.loop();               // Process WebSocket events
}
