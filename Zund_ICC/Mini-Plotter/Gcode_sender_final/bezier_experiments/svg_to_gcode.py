import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# SVG -> GCODE (polyline/path lines only)
# Supports path commands: M, L, H, V, Z (absolute + relative)
# Applies SVG transforms on groups and paths
# Outputs format like:
# G21G90;
# ;path#polyline1
# G0 X... Y... Z1;
# G1 X... Y... Z0;
# ============================================================


# ---------- affine transform helpers ----------
# Matrix format: (a, b, c, d, e, f)
# [a c e]
# [b d f]
# [0 0 1]

def mat_mul(M1, M2):
    a1, b1, c1, d1, e1, f1 = M1
    a2, b2, c2, d2, e2, f2 = M2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def parse_transform(s):
    if not s:
        return (1, 0, 0, 1, 0, 0)

    s = s.strip()
    pat = re.compile(r'([a-zA-Z]+)\s*\(([^)]*)\)')
    M = (1, 0, 0, 1, 0, 0)

    for name, args in pat.findall(s):
        vals = [float(v) for v in re.split(r'[,\s]+', args.strip()) if v]

        if name == "matrix":
            if len(vals) != 6:
                raise ValueError("matrix() must have 6 values")
            T = tuple(vals)

        elif name == "translate":
            tx = vals[0]
            ty = vals[1] if len(vals) > 1 else 0.0
            T = (1, 0, 0, 1, tx, ty)

        elif name == "scale":
            sx = vals[0]
            sy = vals[1] if len(vals) > 1 else sx
            T = (sx, 0, 0, sy, 0, 0)

        elif name == "rotate":
            ang = math.radians(vals[0])
            c = math.cos(ang)
            s_ = math.sin(ang)
            R = (c, s_, -s_, c, 0, 0)
            if len(vals) == 3:
                cx, cy = vals[1], vals[2]
                T = mat_mul(mat_mul((1, 0, 0, 1, cx, cy), R), (1, 0, 0, 1, -cx, -cy))
            else:
                T = R

        elif name == "skewX":
            T = (1, 0, math.tan(math.radians(vals[0])), 1, 0, 0)

        elif name == "skewY":
            T = (1, math.tan(math.radians(vals[0])), 0, 1, 0, 0)

        else:
            raise ValueError(f"Unsupported transform: {name}")

        M = mat_mul(M, T)

    return M


def apply_transform(M, x, y):
    a, b, c, d, e, f = M
    return a * x + c * y + e, b * x + d * y + f


# ---------- SVG path parser (M/L/H/V/Z only) ----------
_token_re = re.compile(
    r'[MmLlHhVvZz]|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?'
)
_cmd_re = re.compile(r'^[MmLlHhVvZz]$')


def parse_path_to_subpaths(d):
    """
    Returns list of subpaths; each subpath is a list of (x,y) points.
    Supports M/L/H/V/Z only. (No curves/arcs in this version)
    """
    tokens = _token_re.findall(d)
    i = 0
    cmd = None
    x = y = 0.0
    sx = sy = 0.0
    subpaths = []
    cur = []

    def getnum():
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        t = tokens[i]
        if _cmd_re.match(t):
            cmd = t
            i += 1

        if cmd in ("M", "m"):
            first = True
            while i < len(tokens) and not _cmd_re.match(tokens[i]):
                nx = getnum()
                ny = getnum()
                if cmd == "m":
                    nx += x
                    ny += y

                if first:
                    if cur:
                        subpaths.append(cur)
                    cur = [(nx, ny)]
                    sx, sy = nx, ny
                    first = False
                else:
                    # extra coords after M are treated as L
                    cur.append((nx, ny))

                x, y = nx, ny

            # implicit line mode after M
            cmd = "L" if cmd == "M" else "l"

        elif cmd in ("L", "l"):
            while i < len(tokens) and not _cmd_re.match(tokens[i]):
                nx = getnum()
                ny = getnum()
                if cmd == "l":
                    nx += x
                    ny += y
                cur.append((nx, ny))
                x, y = nx, ny

        elif cmd in ("H", "h"):
            while i < len(tokens) and not _cmd_re.match(tokens[i]):
                nx = getnum()
                if cmd == "h":
                    nx += x
                x = nx
                cur.append((x, y))

        elif cmd in ("V", "v"):
            while i < len(tokens) and not _cmd_re.match(tokens[i]):
                ny = getnum()
                if cmd == "v":
                    ny += y
                y = ny
                cur.append((x, y))

        elif cmd in ("Z", "z"):
            if cur and (abs(x - sx) > 1e-12 or abs(y - sy) > 1e-12):
                cur.append((sx, sy))
                x, y = sx, sy
            cmd = None

        else:
            raise ValueError(
                f"Unsupported path command '{cmd}'. "
                "This script supports only M/L/H/V/Z. "
                "Convert curves to polylines first if needed."
            )

    if cur:
        subpaths.append(cur)

    return subpaths


# ---------- formatting ----------
def fmt(v):
    return f"{v:.6f}"


# ---------- main converter ----------
def svg_to_gcode(
    svg_file,
    gcode_file,
    page_w=210.0,
    page_h=297.0,
    clamp_to_page=False
):
    svg_path = Path(svg_file)
    out_path = Path(gcode_file)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Handle namespace or no-namespace SVGs
    def local_name(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    all_subpaths = []
    all_names = []

    bounds = [float("inf"), float("inf"), -float("inf"), -float("inf")]  # minx,miny,maxx,maxy

    def walk(el, parent_M=(1, 0, 0, 1, 0, 0)):
        M2 = mat_mul(parent_M, parse_transform(el.attrib.get("transform")))

        if local_name(el.tag) == "path":
            d = el.attrib.get("d", "")
            if d.strip():
                subpaths = parse_path_to_subpaths(d)
                pid = el.attrib.get("id", "path")

                for idx, sub in enumerate(subpaths, start=1):
                    tsub = []
                    for x, y in sub:
                        tx, ty = apply_transform(M2, x, y)
                        tsub.append((tx, ty))

                        bounds[0] = min(bounds[0], tx)
                        bounds[1] = min(bounds[1], ty)
                        bounds[2] = max(bounds[2], tx)
                        bounds[3] = max(bounds[3], ty)

                    if len(tsub) >= 2:
                        all_subpaths.append(tsub)
                        all_names.append(pid if len(subpaths) == 1 else f"{pid}_{idx}")

        for child in list(el):
            walk(child, M2)

    walk(root)

    if not all_subpaths:
        raise ValueError("No drawable path segments found.")

    # Shift only if negative coordinates exist (to satisfy 0,0 minimum)
    shift_x = -bounds[0] if bounds[0] < 0 else 0.0
    shift_y = -bounds[1] if bounds[1] < 0 else 0.0

    # Optional clamp check / warning
    final_minx = bounds[0] + shift_x
    final_miny = bounds[1] + shift_y
    final_maxx = bounds[2] + shift_x
    final_maxy = bounds[3] + shift_y

    if final_maxx > page_w or final_maxy > page_h:
        msg = (
            f"Warning: Drawing exceeds page after shift. "
            f"Final bounds x[{final_minx:.3f},{final_maxx:.3f}] "
            f"y[{final_miny:.3f},{final_maxy:.3f}] vs page {page_w}x{page_h} mm."
        )
        if clamp_to_page:
            print(msg + " (clamp_to_page=True: points will be clamped)")
        else:
            print(msg)

    # Generate G-code
    lines = ["G21G90;"]  # mm + absolute

    for name, sub in zip(all_names, all_subpaths):
        pts = []
        last = None

        for x, y in sub:
            xx = x + shift_x
            yy = y + shift_y

            if clamp_to_page:
                xx = max(0.0, min(page_w, xx))
                yy = max(0.0, min(page_h, yy))

            q = (xx, yy)

            # remove consecutive duplicate points
            if last is None or abs(q[0] - last[0]) > 1e-9 or abs(q[1] - last[1]) > 1e-9:
                pts.append(q)
                last = q

        if len(pts) < 2:
            continue

        lines.append(f";path#{name}")

        # Move to start (pen up)
        x0, y0 = pts[0]
        lines.append(f"G0 X{fmt(x0)} Y{fmt(y0)} Z1;")

        # Draw path (pen down)
        for x, y in pts[1:]:
            lines.append(f"G1 X{fmt(x)} Y{fmt(y)} Z0;")

        # Lift pen at end
        xe, ye = pts[-1]
        lines.append(f"G0 X{fmt(xe)} Y{fmt(ye)} Z1;")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Summary
    print(f"Saved: {out_path}")
    print(f"Paths exported: {len(all_subpaths)}")
    print(f"Original bounds (mm): x[{bounds[0]:.6f}, {bounds[2]:.6f}] y[{bounds[1]:.6f}, {bounds[3]:.6f}]")
    print(f"Applied shift (mm): dx={shift_x:.6f}, dy={shift_y:.6f}")
    print(f"Final bounds (mm): x[{final_minx:.6f}, {final_maxx:.6f}] y[{final_miny:.6f}, {final_maxy:.6f}]")

    print("\nPreview:")
    for line in lines[:12]:
        print(line)


# ---------- run example ----------
if __name__ == "__main__":
    svg_to_gcode(
        svg_file="chins.svg",
        gcode_file="chins_page_210x297.gcode",
        page_w=210.0,
        page_h=297.0,
        clamp_to_page=False
    )