import serial
import serial.tools.list_ports
import time

class GRBLController:
    def __init__(self, pattern="usb", baud=9600, timeout=1.0):
        self._pattern = pattern
        self._baud = baud
        self._timeout = timeout
        self._ser = None

    # ---- serial helpers ----
    def _find_serial_port(self):
        for p in serial.tools.list_ports.comports():
            if self._pattern.lower() in p.device.lower():
                return p.device
        return None

    def connect(self):
        port = self._find_serial_port()
        if not port:
            raise RuntimeError(f"No serial device matching '{self._pattern}' found")
        self._ser = serial.Serial(port, self._baud, timeout=self._timeout)
        print(f"Using port: {port}")
        # Small pause so GRBL can reset & print its banner
        time.sleep(0.2)
        self._flush_input()

    def _flush_input(self):
        """Clear any pending input (boot banner, junk)."""
        if not self._ser: return
        self._ser.reset_input_buffer()
        # Some adapters ignore reset_input_buffer; do a timed drain:
        t0 = time.time()
        while time.time() - t0 < 0.15 and self._ser.in_waiting:
            _ = self._ser.read(self._ser.in_waiting)

    def _wait_for_ok(self):
        """Block until we read 'ok' or 'error:' line. Returns when ok; raises on error."""
        if not self._ser:
            raise RuntimeError("Not connected")
        while True:
            line = self._ser.readline()
            if not line:
                # timeout—loop again
                continue
            try:
                s = line.decode(errors="ignore").strip()
            except Exception:
                s = line.decode("latin1", errors="ignore").strip()

            if not s:
                continue
            # Uncomment if you want to see GRBL responses while debugging:
            # print("GRBL:", s)

            low = s.lower()
            if low == "ok":
                return
            if low.startswith("error"):
                raise RuntimeError(f"GRBL error: {s}")
            # Otherwise it's a status/info line; keep reading.

    def _send_and_wait(self, cmd: str):
        """Write a single G-code/command line and wait for ok."""
        if not self._ser:
            self.connect()
        if not cmd.endswith("\n"):
            cmd += "\n"
        self._ser.write(cmd.encode())
        self._wait_for_ok()

    # ---- high-level API ----
    def activate(self):
        if not self._ser:
            self.connect()

        print("Move rails to zero position manually, then press Enter.")
        input()

        # Example machine settings (tune to your machine)
        self._send_and_wait("$100=14.29")   # X steps/mm
        self._send_and_wait("$101=14.29")   # Y steps/mm
        self._send_and_wait("$110=20000")   # X max rate
        self._send_and_wait("$111=20000")   # Y max rate
        self._send_and_wait("$120=200")     # X accel
        self._send_and_wait("$121=200")     # Y accel

        self._send_and_wait("G21")          # mm
        self._send_and_wait("G90")          # absolute
        self._send_and_wait("G92 X0 Y0")    # set work zero

        # Your servo bridge expects these plain lines:
        self._send_and_wait("M3 S0")          # servo1 to 0
        # self._send_and_wait("2 0")          # servo2 to 0

    def upCold(self):
        self._send_and_wait("M3 S0")

    def downCold(self):
        self._send_and_wait("M3 S180")

    def upHot(self):
        self._send_and_wait("M4 S0")

    def downHot(self):
        self._send_and_wait("M4 S180")

    def set_position(self, x_mm: float, y_mm: float, feed: int = 20000):
        """Blocking move: returns after GRBL acknowledges the command line (not after motion completes)."""
        self._send_and_wait(f"G1 X{x_mm:.3f} Y{y_mm:.3f} F{feed}")

if __name__ == "__main__":
    ctl = GRBLController(pattern="usb", baud=115200, timeout=1.0)
    ctl.activate()
    print("Moving to (60,60)")
    ctl.set_position(60, 60)
    print("Moving to (0,0)")
    ctl.set_position(0, 0)