import numpy as np
from typing import List, Tuple, Optional

# Optional fast skeletonization via scikit-image
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def _zhang_suen_thinning(bin_img: np.ndarray) -> np.ndarray:
    """
    Minimal Zhang–Suen thinning. Input: boolean array (True=foreground).
    Returns boolean skeleton with single-pixel thickness where possible.
    """
    img = bin_img.copy().astype(np.uint8)
    changed = True
    H, W = img.shape

    def neighbors(y, x):
        # clockwise P2..P9 around P1 (y,x)
        return [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
                img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]

    def transitions(nb):
        # number of 0->1 transitions in circular sequence
        s = 0
        for i in range(8):
            if nb[i] == 0 and nb[(i+1) % 8] == 1:
                s += 1
        return s

    # Pad to simplify neighbor checks
    img = np.pad(img, 1, mode='constant', constant_values=0)

    while changed:
        changed = False
        to_del = []

        # Step 1
        for y in range(1, H+1):
            for x in range(1, W+1):
                if img[y, x] != 1:
                    continue
                nb = neighbors(y, x)
                A = transitions(nb)
                B = sum(nb)
                if (2 <= B <= 6 and A == 1 and
                    (nb[0] * nb[2] * nb[4] == 0) and
                    (nb[2] * nb[4] * nb[6] == 0)):
                    to_del.append((y, x))
        if to_del:
            changed = True
            for (y, x) in to_del:
                img[y, x] = 0

        to_del = []
        # Step 2
        for y in range(1, H+1):
            for x in range(1, W+1):
                if img[y, x] != 1:
                    continue
                nb = neighbors(y, x)
                A = transitions(nb)
                B = sum(nb)
                if (2 <= B <= 6 and A == 1 and
                    (nb[0] * nb[2] * nb[6] == 0) and
                    (nb[0] * nb[4] * nb[6] == 0)):
                    to_del.append((y, x))
        if to_del:
            changed = True
            for (y, x) in to_del:
                img[y, x] = 0

    # Remove padding
    img = img[1:-1, 1:-1]
    return img.astype(bool)

def _skeletonize_bool(bw: np.ndarray) -> np.ndarray:
    """bw (bool): True for foreground. Returns bool skeleton."""
    if _HAS_SKIMAGE:
        # skimage expects 1 for foreground
        return _sk_skeletonize(bw).astype(bool)
    return _zhang_suen_thinning(bw)

def extract_strokes(
    img: np.ndarray,
    step_px: float = 5.0,
    erase_radius: int = 3,
    connectivity: int = 8,
    max_walk: Optional[int] = None,
    max_size_mm: float = 160.0,  # 16 cm
) -> List[List[Tuple[float, float]]]:
    """
    Stroke extractor for thick lines:
      - Skeletonizes current image each iteration
      - Follows centerline on the skeleton
      - Samples every `step_px`
      - Erases disks of radius `erase_radius` in the ORIGINAL image
      - Repeats until no white pixels remain

    Returns: list of strokes, each stroke is list[(x_mm, y_mm)].
    """
    assert img.ndim == 2 and img.dtype == np.uint8
    assert connectivity in (4, 8)

    H, W = img.shape
    px_to_mm = min(max_size_mm / float(W), max_size_mm / float(H))

    # Work copy of original image (we erase here)
    work = img.copy()

    neigh_4 = [(0,1),(1,0),(0,-1),(-1,0)]
    neigh_8 = [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if not (dy==0 and dx==0)]
    neighbors = neigh_8 if connectivity == 8 else neigh_4

    def in_bounds(y, x): return (0 <= y < H) and (0 <= x < W)

    def erase_disk_on_work(yc: float, xc: float, r: int):
        y0, x0 = int(round(yc)), int(round(xc))
        r2 = r*r
        y_min, y_max = max(0, y0 - r), min(H-1, y0 + r)
        x_min, x_max = max(0, x0 - r), min(W-1, x0 + r)
        for yy in range(y_min, y_max+1):
            dy = yy - y0
            max_dx = int(np.floor(np.sqrt(max(0, r2 - dy*dy))))
            xx1, xx2 = max(x_min, x0 - max_dx), min(x_max, x0 + max_dx)
            if xx2 >= xx1:
                work[yy, xx1:xx2+1] = 0

    def sample_by_step(path_ij, step):
        if not path_ij: return []
        pts = np.array([[x, y] for (y, x) in path_ij], dtype=float)  # (x,y) in px
        diffs = pts[1:] - pts[:-1]
        seglen = np.sqrt((diffs**2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total = cum[-1]
        if total == 0.0:
            x, y = pts[0]
            return [(float(x), float(y))]
        samples = [0.0]
        s = step
        while s < total:
            samples.append(s); s += step
        samples.append(total)
        out, j = [], 0
        for s in samples:
            while j+1 < len(cum) and cum[j+1] < s:
                j += 1
            if j+1 >= len(cum):
                out.append(tuple(pts[-1])); continue
            t = 0.0 if cum[j+1] == cum[j] else (s - cum[j]) / (cum[j+1] - cum[j])
            p = pts[j] * (1 - t) + pts[j+1] * t
            out.append((float(p[0]), float(p[1])))
        # de-dup boundary duplicates
        dedup = [out[0]]
        for p in out[1:]:
            if abs(p[0]-dedup[-1][0]) > 1e-6 or abs(p[1]-dedup[-1][1]) > 1e-6:
                dedup.append(p)
        return dedup

    def line_follow_on_skel(skel: np.ndarray, seed: Tuple[int,int]) -> List[Tuple[int,int]]:
        """
        Greedy direction-preserving walk over 1-px skeleton.
        Handles forks by choosing the 'straightest' continuation.
        """
        def degree(y, x):
            d = 0
            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if in_bounds(ny, nx) and skel[ny, nx]:
                    d += 1
            return d

        def next_step(y, x, py, px):
            cand = []
            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if not in_bounds(ny, nx): continue
                if not skel[ny, nx]: continue
                if (ny == py and nx == px): continue
                cand.append((ny, nx, dy, dx))
            if not cand: return None
            if py is None or px is None:
                return cand[0][:2]
            v_dir = np.array([y - py, x - px], dtype=float)
            best, best_dot = None, -1e9
            for ny, nx, dy, dx in cand:
                v_n = np.array([dy, dx], dtype=float)
                dot = float(v_dir @ v_n)
                if dot > best_dot:
                    best_dot, best = dot, (ny, nx)
            return best

        # go from seed to one endpoint
        def to_endpoint(start_y, start_x):
            path = [(start_y, start_x)]
            prev = (None, None)
            steps = 0
            cap = max_walk if max_walk is not None else H * W
            while steps < cap:
                y, x = path[-1]
                if degree(y, x) <= 1 and len(path) > 1:
                    break
                nxt = next_step(y, x, *prev)
                if nxt is None: break
                prev = (y, x)
                path.append(nxt)
                steps += 1
            return path, path[-1]

        legA, endA = to_endpoint(*seed)

        # then continue to the opposite endpoint
        path = legA[:]
        visited = set(path)
        prev = (None, None)
        if len(legA) >= 2:
            prev = legA[-2]
        y, x = endA
        steps = 0
        cap = max_walk if max_walk is not None else H * W

        while steps < cap:
            # stop if we reached another endpoint
            # (endpoint iff degree <=1 and not the very start)
            def deg_at(p): 
                return sum(
                    in_bounds(p[0]+dy, p[1]+dx) and skel[p[0]+dy, p[1]+dx]
                    for dy, dx in neighbors
                )
            if deg_at((y, x)) <= 1 and (y, x) != endA:
                break

            cand = []
            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if not in_bounds(ny, nx): continue
                if not skel[ny, nx]: continue
                if (ny, nx) == prev: continue
                cand.append((ny, nx, dy, dx))
            if not cand: break

            if prev[0] is None:
                ny, nx = cand[0][:2]
            else:
                v_dir = np.array([y - prev[0], x - prev[1]], dtype=float)
                best, best_dot = None, -1e9
                for ny_, nx_, dy, dx in cand:
                    v_n = np.array([dy, dx], dtype=float)
                    dot = float(v_dir @ v_n)
                    if dot > best_dot and (ny_, nx_) not in visited:
                        best_dot, best = dot, (ny_, nx_)
                if best is None:
                    best = cand[0][:2]
                ny, nx = best

            prev = (y, x)
            y, x = ny, nx
            if (y, x) not in visited:
                path.append((y, x))
                visited.add((y, x))
            steps += 1

        return path

    strokes_mm: List[List[Tuple[float, float]]] = []

    while True:
        # stop if nothing left
        if not np.any(work == 255):
            break

        # Skeletonize current foreground (True=foreground)
        skel = _skeletonize_bool(work == 255)
        if not skel.any():
            # Nothing structurally traceable; clear leftovers to avoid infinite loop
            work[:] = 0
            break

        # Find a seed pixel on the skeleton
        ys, xs = np.where(skel)
        seed = (int(ys[0]), int(xs[0]))

        # Trace along skeleton to get centerline path (in pixel indices)
        path_ij = line_follow_on_skel(skel, seed)
        if not path_ij:
            # Defensive: blank the seed pixel from work and continue
            work[seed[0], seed[1]] = 0
            continue

        # Sample along the centerline at fixed arc-length in pixels
        stroke_px = sample_by_step(path_ij, step_px)

        # Convert to mm (fit-to-max scaling)
        stroke_mm = [(x * px_to_mm, y * px_to_mm) for (x, y) in stroke_px]
        strokes_mm.append(stroke_mm)

        # Erase thickness on the ORIGINAL work image around the sampled points
        for x, y in stroke_px:
            erase_disk_on_work(y, x, erase_radius)

    return strokes_mm
