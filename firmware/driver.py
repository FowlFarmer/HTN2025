import math
from dataclasses import dataclass

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

@dataclass
class RailLimits:
    y_left_min: float
    y_left_max: float
    y_right_min: float
    y_right_max: float

class TwoRailDelta:
    """
    2-DOF planar delta with two vertical rails at x = ± D/2.
    Each rail has a slider at y_L and y_R; end-effector is at (x, y).
    Distances from (−D/2, y_L) and (+D/2, y_R) to (x, y) are both L.
    """
    def __init__(self,
                 rail_sep_mm: float = 150.0,   # D
                 rod_len_mm: float = 150.0,    # L
                 workspace: Bounds = Bounds(-50.0, 50.0, 0.0, 120.0),
                 rail_limits: RailLimits = RailLimits(0.0, 200.0, 0.0, 200.0),
                 prefer_above: bool = True):
        self.D = float(rail_sep_mm)
        self.L = float(rod_len_mm)
        self.workspace = workspace
        self.rail_limits = rail_limits
        self.prefer_above = prefer_above

        if self.L <= 0 or self.D <= 0:
            raise ValueError("rod_len_mm and rail_sep_mm must be positive.")
        # Geometry sanity: you can operate with any D and L, but reachable x must satisfy |x ± D/2| ≤ L.

    def _rail_y_from_xy(self, x: float, y: float, side: str) -> float:
        """
        Compute rail slider y for a given (x, y).
        side: 'L' (left rail at x=-D/2) or 'R' (right rail at x=+D/2).
        Uses the 'above' branch by default (y_slider >= y), else the 'below' branch.
        """
        if side not in ('L', 'R'):
            raise ValueError("side must be 'L' or 'R'.")

        x_base = -self.D / 2.0 if side == 'L' else self.D / 2.0
        dx = x - x_base
        # Feasibility check for this side: horizontal offset must be ≤ L
        if abs(dx) > self.L:
            raise ValueError(f"Unreachable: |x - x_{side}| = {abs(dx):.3f} > L = {self.L:.3f}")

        # (x - x_base)^2 + (y - y_slider)^2 = L^2
        # => (y_slider - y)^2 = L^2 - dx^2
        root = math.sqrt(max(0.0, self.L**2 - dx**2))
        if self.prefer_above:
            y_slider = y + root       # slider above the effector
        else:
            y_slider = y - root       # slider below the effector

        return y_slider

    def ik(self, x: float, y: float):
        """
        Inverse kinematics: (x,y) -> (y_L, y_R).
        Checks workspace and rail limits; raises ValueError if invalid/unreachable.
        """
        # Workspace check (optional but helpful)
        if not (self.workspace.x_min <= x <= self.workspace.x_max and
                self.workspace.y_min <= y <= self.workspace.y_max):
            raise ValueError(
                f"Target (x={x:.3f}, y={y:.3f}) is outside workspace "
                f"[x:{self.workspace.x_min}..{self.workspace.x_max}], "
                f"[y:{self.workspace.y_min}..{self.workspace.y_max}]."
            )

        # Compute rail positions
        y_L = self._rail_y_from_xy(x, y, 'L')
        y_R = self._rail_y_from_xy(x, y, 'R')

        # Rail limit checks
        if not (self.rail_limits.y_left_min <= y_L <= self.rail_limits.y_left_max):
            raise ValueError(
                f"Left rail y={y_L:.3f} mm out of limits "
                f"[{self.rail_limits.y_left_min}..{self.rail_limits.y_left_max}] mm."
            )
        if not (self.rail_limits.y_right_min <= y_R <= self.rail_limits.y_right_max):
            raise ValueError(
                f"Right rail y={y_R:.3f} mm out of limits "
                f"[{self.rail_limits.y_right_min}..{self.rail_limits.y_right_max}] mm."
            )

        return y_L, y_R

    def reachable(self, x: float, y: float) -> bool:
        """Quick boolean check without exceptions."""
        try:
            self.ik(x, y)
            return True
        except ValueError:
            return False

if __name__ == "__main__":
    # Example usage:
    # Default geometry: D = L = 150 mm (your note), rails run from y=0..200 mm,
    # workspace x in [-50, 50] mm, y in [0, 120] mm.
    robot = TwoRailDelta()

    tests = [
        (0.0, 60.0),
        (30.0, 80.0),
        (-40.0, 40.0),
    ]

    for (x, y) in tests:
        try:
            yL, yR = robot.ik(x, y)
            print(f"Target (x={x:.1f}, y={y:.1f}) -> Left y={yL:.2f} mm, Right y={yR:.2f} mm")
        except ValueError as e:
            print(f"Target (x={x:.1f}, y={y:.1f}) unreachable: {e}")