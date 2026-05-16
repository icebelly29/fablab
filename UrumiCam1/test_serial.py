import serial
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    print("SUCCESS: Opened /dev/ttyACM0")
    ser.close()
except Exception as e:
    print(f"FAILURE: {e}")
