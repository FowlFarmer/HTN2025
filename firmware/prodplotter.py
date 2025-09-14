# delta_actuator.py (top-left input version)

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Optional

from gcode_serial import GRBLController
from driver import TwoRailDelta, Bounds, RailLimits

class DeltaActuator:
    RAIL_SEP = 210.0
    ROD_LEN  = 210.0

    WORKSPACE = Bounds(x_min=-80.0, x_max=80.0, y_min=-220.0, y_max=100.0)
    RAIL_LIMITS = RailLimits(y_left_min=0.0, y_left_max=270.0, y_right_min=0.0, y_right_max=270.0)
    PREFER_ABOVE = True

    # Incoming command space in mm (screen-like, top-left origin)
    CMD_SIZE = 150.0
    CMD_MIN  = 0.0
    CMD_MAX  = 150.0
    HALF     = 75.0

    # Effective EE center offset in robot frame (mm)
    OFFSET_Y = -20.0

    # --- trail config ---
    TRAIL_MAX_POINTS = 20000  # cap memory; adjust as you like

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

        self.rail_x1 = -self.RAIL_SEP / 2.0
        self.rail_x2 =  self.RAIL_SEP / 2.0

        self.observe = observe
        # trail buffers
        self.trail_x = []
        self.trail_y = []

        if self.observe:
            self._init_plot()

    def upCold(self): self.controller.upCold()
    def downCold(self): self.controller.downCold()
    def upHot(self): self.controller.upHot()
    def downHot(self): self.controller.downHot()

    def return_home(self):
        self.command(0, 0)

    def command(self, x_tl_mm: float, y_tl_mm: float) -> bool:
        """
        Accepts top-left-origin coordinates in mm: (0,0) at top-left of 150x150 area.
        +x right, +y down. Maps to robot-centered frame (+y up), applies EE offset,
        checks reachability, runs IK, and drives GRBL.
        """
        # 1) Validate top-left bounds
        if not (self.CMD_MIN <= x_tl_mm <= self.CMD_MAX and self.CMD_MIN <= y_tl_mm <= self.CMD_MAX):
            self._update_plot(None, None, None, None, 0.0, 0.0, infeasible=True,
                              note="Outside 0..150 mm top-left command area")
            return False

        # 2) Map top-left → centered (robot convention, +y up)
        x_c = x_tl_mm - self.HALF
        y_c = self.HALF - y_tl_mm

        # 3) Apply effective center offset (0, -20 mm)
        x_robot = x_c
        y_robot = y_c + self.OFFSET_Y  # OFFSET_Y = -20 → subtract 20

        # 4) Reachability + IK
        if not self.robot.reachable(x_robot, y_robot):
            self._update_plot(None, None, None, None, x_robot, y_robot, infeasible=True,
                              note="Unreachable by workspace/rail limits")
            return False

        try:
            yL, yR = self.robot.ik(x_robot, y_robot)
        except ValueError:
            self._update_plot(None, None, None, None, x_robot, y_robot, infeasible=True, note="IK failed")
            return False

        # 5) Map sliders → GRBL rails (keep your tested flip/swap)
        left_cmd  = 270.0 - yR
        right_cmd = 270.0 - yL

        # Clamp to rail limits
        left_cmd  = float(min(max(left_cmd,  self.RAIL_LIMITS.y_left_min),  self.RAIL_LIMITS.y_left_max))
        right_cmd = float(min(max(right_cmd, self.RAIL_LIMITS.y_right_min), self.RAIL_LIMITS.y_right_max))

        # 6) Send to GRBL
        self.controller.set_position(left_cmd, right_cmd)

        # 7) Observer (and trail update)
        self._update_plot(x_robot, y_robot, yL, yR, x_robot, y_robot, infeasible=False)
        return True

    # -------- plotting (observer only) ----------
    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-220, 250)
        self.ax.plot([self.rail_x1, self.rail_x1], [-300, 300], '--', linewidth=1)
        self.ax.plot([self.rail_x2, self.rail_x2], [-300, 300], '--', linewidth=1)

        # Trail line (thin, semi-transparent)
        (self.trail_ln,) = self.ax.plot([], [], '-', linewidth=1.0, alpha=0.35, color='k', label='Trail')

        (self.ee_dot,)   = self.ax.plot([], [], 'ko', markersize=6, label='End Effector')
        (self.r1_dot,)   = self.ax.plot([], [], 'ro', markersize=6, label='Left Rail Slider')
        (self.r2_dot,)   = self.ax.plot([], [], 'bo', markersize=6, label='Right Rail Slider')
        (self.link1_ln,) = self.ax.plot([], [], '-', linewidth=2, alpha=0.9)
        (self.link2_ln,) = self.ax.plot([], [], '-', linewidth=2, alpha=0.9)

        self.txt = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va='top', ha='left',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85)
        )
        self.ax.legend(loc='lower right')
        self.fig.canvas.draw_idle()
        self.fig.show()

    def clear_trail(self):
        """Erase the accumulated trail from the plot (non-destructive to state)."""
        self.trail_x.clear()
        self.trail_y.clear()
        if self.observe:
            self.trail_ln.set_data([], [])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _update_plot(self, x_plot, y_plot, yL, yR, x_robot, y_robot, infeasible: bool, note: str = ""):
        if not self.observe:
            return

        if infeasible or None in (x_plot, y_plot, yL, yR):
            # Don’t add to trail on infeasible updates
            self.ee_dot.set_data([], [])
            self.r1_dot.set_data([], [])
            self.r2_dot.set_data([], [])
            self.link1_ln.set_data([], [])
            self.link2_ln.set_data([], [])
            msg = note or "Out of bounds / IK failed."
            self.txt.set_text(f"{msg}\n(robot) ({x_robot:.2f}, {y_robot:.2f}) mm")
        else:
            # Append to trail (cap to max points to avoid unbounded memory)
            self.trail_x.append(x_plot)
            self.trail_y.append(y_plot)
            if len(self.trail_x) > self.TRAIL_MAX_POINTS:
                # Drop oldest in chunks for efficiency
                drop = len(self.trail_x) - self.TRAIL_MAX_POINTS
                del self.trail_x[:drop]
                del self.trail_y[:drop]

            # Update trail line
            self.trail_ln.set_data(self.trail_x, self.trail_y)

            # Update current geometry
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

# Example (remove in production)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    act = DeltaActuator(observe=True)
    # Draw a quick box to demo the trail
    for p in [(0,0), (150,0), (150,150), (0,150), (0,0)]:
        act.command(*p)
    if act.observe:
        plt.show(block=True)