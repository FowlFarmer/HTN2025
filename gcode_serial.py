import serial
import serial.tools.list_ports

def find_serial_port(pattern="usbserial"):
    """Find the first serial port whose name contains the pattern."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if pattern.lower() in port.device.lower():
            return port.device
    return None

def activate():
    # Example: auto-find a /dev/tty.usbserial* device
    port_name = find_serial_port("usbserial")
    if not port_name:
        raise RuntimeError("No tty.usbserial device found")
    ser = serial.Serial(port_name, 115200, timeout=1)
    print(f"Using port: {port_name}")
    print("Move the two rails to the beginning to zero motors (The two arm axle joints should be close to the motors).")
    print("Press Enter to continue...")
    input()

# quick test
ser.write(b"$$\n")  # ask GRBL for settings
print(ser.readline().decode())