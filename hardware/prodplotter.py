# delta_actuator.py
# Production actuator for TwoRailDelta: consume (x, y) mm targets and drive GRBL.
# Command space: 150x150 mm centered at (0, 0). Effective EE center is (0, -20mm).

from __future__ import annotations
import math
import threading
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from gcode_serial import GRBLController
from driver import TwoRailDelta, Bounds, RailLimits

class DeltaActuator:
    """
    Consumes (x, y) commands in mm (command frame), maps to robot frame with
    effective-center offset (0, -20 mm), validates reachability, computes IK,
    and sends rail positions to GRBL.

    Coordinate frames:
      - Command frame: user sends (x_cmd, y_cmd) in a square 150x150 mm centered at (0,0).
      - Robot frame: EE position; effective center is (0, -20 mm).
        We map (x_cmd, y_cmd) -> (x_cmd, y_cmd - 20).

    Rail outputs:
      controller.set_position(left_mm, right_mm) where rails are 0..270 mm.
      From your test: left = 270 - yR, right = 270 - yL (note the swap & flip).
    """

    # ------------------ MACHINE CONFIG (adjust if needed) ------------------
    RAIL_SEP = 210.0           # Distance between rails D (mm)
    ROD_LEN  = 210.0           # Rod length L (mm)

    # End-effector workspace in robot frame (mm)
    WORKSPACE = Bounds(
        x_min=-80.0, x_max=80.0,
        y_min=-220.0, y_max=100.0
    )

    # Rail travel limits (slider Y in robot frame, mm)
    RAIL_LIMITS = RailLimits(
        y_left_min=0.0,  y_left_max=270.0,
        y_right_min=0.0, y_right_max=270.0
    )

    # Choose IK branch with sliders above EE
    PREFER_ABOVE = True

    # Command space (what you feed to .command(x,y)), in mm
    CMD_HALF = 75.0            # so command range is [-75, +75] on each axis (150x150)
    CMD_OFFSET_Y = -20.0       # effective center offset: (0, -20 mm)
    # ----------------------------------------------------------------------

    def __init__(self, controller: Optional[GRBLController] = None, observe: bool = True):
        self.controller = controller or GRBLController()
        self.controller.activate()

        self.robot = TwoRailDelta(
            rail_sep_mm=self.RAIL_SEP,
            rod_len_mm=self.ROD_LEN,
            workspace=self.WORKSPACE,
            rail_limits=self.RAIL_LIMITS,
            prefer_above=self.PREFER_ABOVE
        )

        # Rail X positions for plotting
        self.rail_x1 = -self.RAIL_SEP / 2.0
        self.rail_x2 =  self.RAIL_SEP / 2.0

        self.observe = observe
        if self.observe:
            self._init_plot()

    # ---------- Public API ----------

    def command(self, x_mm: float, y_mm: float) -> bool:
        """
        Receive a (x, y) command in mm in the 150x150 command space centered at (0,0).
        Returns True if move sent to controller; False if rejected (out of bounds).
        """
        # Validate command envelope
        if not (-self.CMD_HALF <= x_mm <= self.CMD_HALF and -self.CMD_HALF <= y_mm <= self.CMD_HALF):
            # Outside command space
            return False

        # Map to robot frame (apply effective center offset)
        x_robot = x_mm
        y_robot = y_mm + self.CMD_OFFSET_Y  # (0, -20) effective center -> add -20 to incoming y

        # Check reachability and compute IK
        if not self.robot.reachable(x_robot, y_robot):
            self._update_plot(None, None, None, None, x_robot, y_robot, infeasible=True)
            return False

        try:
            yL, yR = self.robot.ik(x_robot, y_robot)
        except ValueError:
            self._update_plot(None, None, None, None, x_robot, y_robot, infeasible=True)
            return False

        # Convert slider positions -> GRBL rail commands (note flip/swap from your script)
        left_cmd  = 270.0 - yR
        right_cmd = 270.0 - yL

        # Safety clamp to rail limits
        left_cmd  = float(min(max(left_cmd,  self.RAIL_LIMITS.y_left_min),  self.RAIL_LIMITS.y_left_max))
        right_cmd = float(min(max(right_cmd, self.RAIL_LIMITS.y_right_min), self.RAIL_LIMITS.y_right_max))

        # Send to machine
        self.controller.set_position(left_cmd, right_cmd)

        # Update observer
        self._update_plot(x_robot, y_robot, yL, yR, x_robot, y_robot, infeasible=False)
        return True

    # ---------- Internal helpers ----------

    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        # Show a comfortable view around workspace
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-220, 250)

        # Rails
        self.rail1_line, = self.ax.plot([self.rail_x1, self.rail_x1], [-300, 300], '--', linewidth=1)
        self.rail2_line, = self.ax.plot([self.rail_x2, self.rail_x2], [-300, 300], '--', linewidth=1)

        # Artists to update
        self.ee_dot,   = self.ax.plot([], [], 'ko', markersize=6, label='End Effector')
        self.r1_dot,   = self.ax.plot([], [], 'ro', markersize=6, label='Left Rail Slider')
        self.r2_dot,   = self.ax.plot([], [], 'bo', markersize=6, label='Right Rail Slider')
        self.link1_ln, = self.ax.plot([], [], '-', linewidth=2, alpha=0.9)
        self.link2_ln, = self.ax.plot([], [], '-', linewidth=2, alpha=0.9)
        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va='top', ha='left',
                                fontsize=10, family='monospace',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
        self.ax.legend(loc='lower right')
        self.fig.canvas.draw_idle()
        self.fig.show()

    def _update_plot(self,
                     x_plot: Optional[float],
                     y_plot: Optional[float],
                     yL: Optional[float],
                     yR: Optional[float],
                     x_robot: float,
                     y_robot: float,
                     infeasible: bool):
        if not self.observe:
            return
        if infeasible or x_plot is None or y_plot is None or yL is None or yR is None:
            # Clear markers on infeasible
            self.ee_dot.set_data([], [])
            self.r1_dot.set_data([], [])
            self.r2_dot.set_data([], [])
            self.link1_ln.set_data([], [])
            self.link2_ln.set_data([], [])
            self.txt.set_text(f"Out of bounds / IK failed at ({x_robot:.2f}, {y_robot:.2f}) mm")
        else:
            self.ee_dot.set_data([x_plot], [y_plot])
            self.r1_dot.set_data([self.rail_x1], [yL])
            self.r2_dot.set_data([self.rail_x2], [yR])
            self.link1_ln.set_data([x_plot, self.rail_x1], [y_plot, yL])
            self.link2_ln.set_data([x_plot, self.rail_x2], [y_plot, yR])
            self.txt.set_text(
                f"x,y (robot): ({x_plot:7.3f}, {y_plot:7.3f}) mm\n"
                f"yL,yR:       ({yL:7.3f}, {yR:7.3f}) mm\n"
                f"D/L:         {self.RAIL_SEP:.1f}/{self.ROD_LEN:.1f}  |  branch: {'above' if self.PREFER_ABOVE else 'below'}"
            )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Build actuator with observer on (matplotlib only visualizes; no manual controls)
    actuator = DeltaActuator(observe=True)

    # Example moves in the 150x150 command space (centered at 0,0):
    # These will internally offset to (x, y-20) in the robot frame.
    demo_points = [
        (0.0, 0.0),
        (40.0,  40.0),
        (-40.0, 40.0),
        (60.0, -30.0),
        (0.0,  70.0),
        (0.0,   0.0),
    ]

    for (x, y) in demo_points:
        ok = actuator.command(x, y)
        print(f"command({x:.1f}, {y:.1f}) -> {'sent' if ok else 'rejected'}")

    # Keep the observer window open until closed by user
    if actuator.observe:
        print("Observer active. Close the window to exit.")
        plt.show(block=True)