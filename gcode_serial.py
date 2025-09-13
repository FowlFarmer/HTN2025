import serial
import serial.tools.list_ports

import time
import threading # add commands to buffer and it will send them in order in background
import queue

class GRBLController:
    def __init__(self, pattern="usbserial", baud=115200, timeout=1):
        self._pattern = pattern
        self._baud = baud
        self._timeout = timeout
        self._ser = None
        self._t = None
        self._command_queue = queue.Queue()

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
                time.sleep(0.005)
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
        self._ser.write(b"$100=14.29\n")  # feedrate settings
        self.wait_for_ok()
        self._ser.write(b"$101=14.29\n")  
        self.wait_for_ok()

        self._ser.write(b"$110=20000\n")  # max speed settings
        self.wait_for_ok()
        self._ser.write(b"$111=20000\n")  
        self.wait_for_ok()

        self._ser.write(b"$120=200\n")  # max accel settings
        self.wait_for_ok()
        self._ser.write(b"$121=200\n")  # max accel settings
        self.wait_for_ok()

        self._ser.write(b"G21\n")  # set units to mm
        self.wait_for_ok()
        self._ser.write(b"G90\n")  # absolute positioning
        self.wait_for_ok()
        self._ser.write(b"G92 X0 Y0\n")  # home axis
        self.wait_for_ok()
        self._t = threading.Thread(target=self._command_sender, daemon=True)
        self._t.start()
        
    def _command_sender(self):
        """Background thread to send commands from the queue."""
        while True:
            cmd = self._command_queue.get()
            if cmd is None:
                time.sleep(0.005)
                continue
            self._ser.write(cmd)
            self.wait_for_ok()
            # self._command_queue.task_done() # not necessary
    
    def set_position(self, x_mm, y_mm):
        """Move to (x_mm, y_mm) in mm."""
        if not self._ser:
            raise RuntimeError("Serial connection not established. Call connect() first.")
        cmd = f"G1 X{x_mm:.3f} Y{y_mm:.3f} F20000\n".encode()
        self._command_queue.put(cmd)

if __name__ == "__main__":
    controller = GRBLController()
    controller.activate()
    print("Moving to (60, 60) mm")
    controller.set_position(60, 60)
    print("Moving to (0, 0) mm")
    controller.set_position(0, 0)