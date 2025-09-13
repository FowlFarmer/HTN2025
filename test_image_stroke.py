# run_stroke_player.py
import argparse
import glob
import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from stroke_extractor import extract_strokes  # <-- your function file

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def load_binary_0_255(path: str) -> np.ndarray:
    """Load as grayscale uint8 {0,255}."""
    im = Image.open(path).convert("L")  # grayscale
    arr = np.array(im, dtype=np.uint8)
    # Threshold to binary
    bw = (arr >= 128).astype(np.uint8) * 255
    return bw

def compute_mm_dims(h: int, w: int, max_size_mm: float) -> Tuple[float, float, float]:
    """Return (px_to_mm, width_mm, height_mm) using fit-to-max scaling."""
    px_to_mm = min(max_size_mm / float(w), max_size_mm / float(h))
    return px_to_mm, w * px_to_mm, h * px_to_mm

def play_strokes_mm(
    strokes_mm: List[List[Tuple[float, float]]],
    width_mm: float,
    height_mm: float,
    point_delay_s: float = 0.1,
    title: str = ""
):
    """Animate drawing strokes (mm coords) with Matplotlib."""
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)  # origin at top-left like image coords
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(False)

    # We draw one line per stroke so each stroke remains visible when the next begins.
    for s_idx, stroke in enumerate(strokes_mm):
        if not stroke:
            continue
        xs, ys = [], []
        (line,) = ax.plot([], [], linewidth=2)  # default color, solid line
        # Optionally show a moving marker:
        marker, = ax.plot([], [], "o", markersize=4)

        for (x, y) in stroke:
            xs.append(x); ys.append(y)
            line.set_data(xs, ys)
            marker.set_data([x], [y])
            fig.canvas.draw_idle()
            plt.pause(point_delay_s)

        # small pause at end of each stroke to visually separate
        plt.pause(0.2)

    # hold final frame until user closes
    plt.ioff()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Play extracted strokes from images in a directory.")
    ap.add_argument("--img_dir", type=str, required=True, help="Directory containing input images.")
    ap.add_argument("--step_px", type=float, default=4.0, help="Sampling step along centerline (pixels).")
    ap.add_argument("--erase_radius", type=int, default=3, help="Erase radius (pixels) around sampled points.")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Neighbor connectivity.")
    ap.add_argument("--max_size_mm", type=float, default=160.0, help="Fit-to-max square size in mm (default 160 = 16 cm).")
    ap.add_argument("--delay", type=float, default=0.1, help="Delay (s) between plotted points.")
    args = ap.parse_args()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.img_dir, "*"))
        if os.path.splitext(p.lower())[1] in IMG_EXTS
    )
    if not paths:
        print("No images found. Supported extensions:", ", ".join(IMG_EXTS))
        return

    for path in paths:
        print(f"\nProcessing: {os.path.basename(path)}")
        img = load_binary_0_255(path)
        H, W = img.shape
        px_to_mm, width_mm, height_mm = compute_mm_dims(H, W, args.max_size_mm)

        strokes_mm = extract_strokes(
            img,
            step_px=args.step_px,
            erase_radius=args.erase_radius,
            connectivity=args.connectivity,
            max_size_mm=args.max_size_mm,
        )

        if not strokes_mm:
            print("No strokes found.")
            continue

        play_strokes_mm(
            strokes_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            point_delay_s=args.delay,
            title=os.path.basename(path),
        )

if __name__ == "__main__":
    main()
