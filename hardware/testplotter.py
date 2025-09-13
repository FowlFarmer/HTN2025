# delta_driver_loop.py
# Interactive driver-in-the-loop test using TwoRailDelta.
# Controls: W/A/S/D (or arrow keys) to move the end-effector.

import math
import matplotlib.pyplot as plt

from gcode_serial import GRBLController
from driver import TwoRailDelta, Bounds, RailLimits

controller = GRBLController()
controller.activate()
print("Moving to (60, 60) mm")
controller.set_position(60, 60)
print("Moving to (0, 0) mm")
controller.set_position(0, 0)

# ------------------ CONFIGURE YOUR MACHINE HERE ------------------
# Geometry (mm)
RAIL_SEP = 210.0          # Distance between rails (D)
ROD_LEN  = 210.0          # Rod length (L); you said "L = D = 150 mm"

# Workspace limits (end-effector XY bounds, mm)
WORKSPACE = Bounds(
    x_min = -80.0, x_max = 80.0,
    y_min =  -220.0, y_max = 100.0
)

# Rail travel limits (slider Y bounds, mm)
RAIL_LIMITS = RailLimits(
    y_left_min  = 0.0,  y_left_max  = 270,
    y_right_min = 0.0,  y_right_max = 270
)

# UI tuning
STEP = 3.0                # WASD step size (mm)
START_XY = [0.0, -50.0]    # initial end-effector position (mm)
AXES_LIMITS = (-150, 150, -200, 250)   # xmin, xmax, ymin, ymax for view
PREFER_ABOVE = True       # choose the "above" IK branch (sliders above EE)
# ----------------------------------------------------------------

# Build the driver
robot = TwoRailDelta(rail_sep_mm=RAIL_SEP,
                     rod_len_mm=ROD_LEN,
                     workspace=WORKSPACE,
                     rail_limits=RAIL_LIMITS,
                     prefer_above=PREFER_ABOVE)

# Convenience: rail X positions
rail_x1, rail_x2 = -RAIL_SEP/2.0, RAIL_SEP/2.0

# End-effector state
EE = [float(START_XY[0]), float(START_XY[1])]

def ik_or_none(x, y):
    """Run IK; return (yL, yR) or None if unreachable/violates limits."""
    try:
        return robot.ik(x, y)
    except ValueError:
        return None

# ---- Plot setup ----
plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(AXES_LIMITS[0], AXES_LIMITS[1])
ax.set_ylim(AXES_LIMITS[2], AXES_LIMITS[3])

# Rails
rail1_line, = ax.plot([rail_x1, rail_x1], [AXES_LIMITS[2], AXES_LIMITS[3]], '--', linewidth=1)
rail2_line, = ax.plot([rail_x2, rail_x2], [AXES_LIMITS[2], AXES_LIMITS[3]], '--', linewidth=1)

# Artists to update
ee_dot, = ax.plot([], [], 'ko', markersize=6, label='End Effector')
r1_dot, = ax.plot([], [], 'ro', markersize=6, label='Left Rail Slider')
r2_dot, = ax.plot([], [], 'bo', markersize=6, label='Right Rail Slider')
link1_line, = ax.plot([], [], '-', linewidth=2, alpha=0.9)
link2_line, = ax.plot([], [], '-', linewidth=2, alpha=0.9)
txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left',
              fontsize=10, family='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
ax.legend(loc='lower right')

def redraw():
    res = ik_or_none(EE[0], EE[1])

    if res is not None:
        yL, yR = res
        controller.set_position(270-yR,270-yL)
        # Update points/links
        ee_dot.set_data([EE[0]], [EE[1]])
        r1_dot.set_data([rail_x1], [yL])
        r2_dot.set_data([rail_x2], [yR])
        link1_line.set_data([EE[0], rail_x1], [EE[1], yL])
        link2_line.set_data([EE[0], rail_x2], [EE[1], yR])

        txt.set_text(
            f"x,y:   ({EE[0]:7.3f}, {EE[1]:7.3f}) mm\n"
            f"yL,yR: ({yL:7.3f}, {yR:7.3f}) mm\n"
            f"D/L:   {RAIL_SEP:.1f}/{ROD_LEN:.1f} mm  |  branch: {'above' if PREFER_ABOVE else 'below'}"
        )
    else:
        # Clear on infeasible (should be rare; we block moves)
        ee_dot.set_data([], [])
        r1_dot.set_data([], [])
        r2_dot.set_data([], [])
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        txt.set_text("Out of bounds (IK/limits failed).")

    fig.canvas.draw_idle()
    # return yL, yR

def try_move(dx, dy):
    """Attempt to move the end-effector; apply only if new pos is feasible and within WS/rail limits."""
    nx, ny = EE[0] + dx, EE[1] + dy
    if robot.reachable(nx, ny):
        EE[0], EE[1] = nx, ny
        redraw()
    # else: ignore move

def on_key(event):
    k = (event.key or "").lower()
    if k in ('w', 'up'):
        try_move(0.0, STEP)
    elif k in ('x', 'down', 's'):
        try_move(0.0, -STEP)
    elif k in ('a', 'left'):
        try_move(-STEP, 0.0)
    elif k in ('d', 'right'):
        try_move(STEP, 0.0)

if __name__ == "__main__":

    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial draw
    redraw()
    print("Controls: W/A/S/D (or arrow keys) to move the end-effector. Close the window to quit.")
    plt.show(block=True)

