import serial
import serial.tools.list_ports

def find_serial_port(pattern="usbserial"):
    """Find the first serial port whose name contains the pattern."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if pattern.lower() in port.device.lower():
            return port.device
    return None

# Example: auto-find a /dev/tty.usbserial* device
port_name = find_serial_port("usbserial")
if not port_name:
    raise RuntimeError("No tty.usbserial device found")

print(f"Using port: {port_name}")
ser = serial.Serial(port_name, 115200, timeout=1)

# quick test
ser.write(b"$$\n")  # ask GRBL for settings
print(ser.readline().decode())