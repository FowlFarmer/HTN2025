import serial
import serial.tools.list_ports

class GRBLController:
    def __init__(self, pattern="usbserial", baud=115200, timeout=1):
        self._pattern = pattern
        self._baud = baud
        self._timeout = timeout
        self._ser = None

    def _find_serial_port(self):
        """Find the first serial port whose name contains the pattern."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if self._pattern.lower() in port.device.lower():
                return port.device
        return None

    def connect(self):
        """Locate the port and open a serial connection to GRBL."""
        port_name = self._find_serial_port()
        if not port_name:
            raise RuntimeError("No tty.usbserial device found")

        self._ser = serial.Serial(port_name, self._baud, timeout=self._timeout)
        print(f"Using port: {port_name}")

    def wait_for_ok(self):
        """Block until GRBL replies 'ok' or error."""
        if not self._ser:
            raise RuntimeError("Serial connection not established. Call connect() first.")
        while True:
            line = self._ser.readline().decode().strip()
            if not line:
                continue  # timeout, just keep looping
            print("GRBL:", line)
            if line.lower() == "ok":
                return
            if line.startswith("error"):
                raise RuntimeError(f"GRBL error: {line}")

    def activate(self):
        """Set up GRBL units, positioning, and home zero."""
        if not self._ser:
            self.connect()

        print("Move the two rails to the beginning to zero motors "
              "(The two arm axle joints should be close to the motors).")
        input("Press Enter to continue...")

        self._ser.write(b"G21\n")  # set units to mm
        self.wait_for_ok()
        self._ser.write(b"G90\n")  # absolute positioning
        self.wait_for_ok()
        self._ser.write(b"G92 X0 Y0\n")  # home axis
        self.wait_for_ok()

if __name__ == "__main__":
    controller = GRBLController()
    controller.activate()