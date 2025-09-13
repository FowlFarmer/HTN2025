import serial
import serial.tools.list_ports

def find_serial_port(pattern="usbserial"):
    """Find the first serial port whose name contains the pattern."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if pattern.lower() in port.device.lower():
            return port.device
    return None

def wait_for_ok():
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue  # timeout, just keep looping
        print("GRBL:", line)  # optional debug
        if line.lower() == "ok":
            return
        if line.startswith("error"):
            raise RuntimeError(f"GRBL error: {line}")

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
    ser.write(b"G21\n") # set units to mm
    wait_for_ok()
    ser.write(b"G90\n") # set to absolute positioning
    wait_for_ok()
    ser.write(b"G92 X0 Y0\n")  # home axis
    wait_for_ok()


if __name__ == "__main__":
    activate()