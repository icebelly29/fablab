"""
===============================================================================
ARUCO FRAME EXTRACTOR - LEARNING VERSION
===============================================================================
This script is a "Smart Digital Scanner." It finds a physical frame in a photo,
fixes the camera tilt and lens distortion, and saves a perfectly flat, 
scale-accurate digital image.

LEARNING RESOURCES FOR BEGINNERS:
---------------------------------
1. OpenCV (cv2) - The "Eyes": 
   - ArUco Markers: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
   - Image Transformations: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

2. NumPy (np) - The "Math Brain":
   - Beginner's Guide: https://numpy.org/doc/stable/user/absolute_beginners.html
   - Visual Guide to Arrays: https://jalammar.github.io/visual-numpy/

3. Python "Clean Code" Tools:
   - Command Line Arguments (argparse): https://docs.python.org/3/howto/argparse.html
   - Config Files (JSON): https://realpython.com/python-json/
===============================================================================
"""

import datetime
import os
import json
import sys
import argparse # Tool for reading terminal commands: https://docs.python.org/3/library/argparse.html

import cv2 # OpenCV Library (Computer Vision): https://opencv.org/
import numpy as np # NumPy Library (Math on Lists): https://numpy.org/

# Helper tools created specifically for this project (found in the /utils folder)
from utils import solve_lens, misc

# --- 1. SETTING UP THE "ORDER" ---
# This function handles what you type in the terminal (like -i photo.jpg).
# Ref: https://docs.python.org/3/howto/argparse.html
def parse_arguments():
    usage_text = (
        "Extracts the image from an image containing an aruco frame"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-i", "--input", type=str,
                        help="Input filename (The photo you took).")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output filename (Where to save the result).")
    parser.add_argument("-d", "--dpi", type=int, default=-1,
                        help="Manual output DPI (Quality setting).")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show debug window while processing.")
    parser.add_argument("-c", "--config", type=str, default="./config/config.json",
                        help="Frame configuration file (The 'rulebook').")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print extra updates to the screen.")
    return parser.parse_known_args()


# --- 2. SHOWING THE IMAGE ---
# Simple function to pop up a window on your desktop.
# Ref: https://docs.opencv.org/4.x/df/d24/group__highgui__opengl.html
def imshow(img, h_view=700, win_name="debug"):
    h, w = img.shape[:2]
    w_view = int(h_view * w / h)
    cv2.imshow(win_name, cv2.resize(img, (w_view, h_view), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0) # Waits for you to press a key before closing


# --- 3. THE "FLATTENER" ---
# This uses math to turn a tilted trapezoid (photo) into a perfect square (scan).
# Ref: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
def extract_image(img, proj, config, dots_per_mm, dist_params=None):
    h, w, c = img.shape

    # Get the "cut-out" area from the config file (in millimeters)
    m = config["margins"]["inner_content"]
    xmin, xmax = m, config["width"] - m
    ymin, ymax = m, config["height"] - m

    # Calculate final pixel size: (mm * DPI)
    h_out = int(dots_per_mm * (ymax - ymin))
    w_out = int(dots_per_mm * (xmax - xmin))

    # Create a perfectly flat grid (Graph Paper)
    # Ref: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    x = np.linspace(xmin, xmax, w_out)
    y = np.linspace(ymax, ymin, h_out)
    xx, yy = np.meshgrid(x, y)

    # Use the projection math to find these grid points in the original photo
    xy_list = np.ones((h_out * w_out, 2))
    xy_list[:, 0] = xx.flatten()
    xy_list[:, 1] = yy.flatten()
    uv_src = apply_affine(proj, xy_list)

    # Fix lens distortion (remove the "curviness" of the camera lens)
    if dist_params is not None:
        k1, k2, uc, vc = dist_params[:]
        mat = np.array([[w, 0, uc], [0, w, vc], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([[0, 0, 0, 0, 0, k1, k2, 0]], dtype=np.float32)
        out = cv2.undistortPoints(uv_src, mat, dist_coeffs, P=mat)
        uv_src = out[:, 0, :]

    # Move pixels from the tilted photo to the new flat grid
    # Ref: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gab75bc239040066270c1795bc2369c56e
    map1 = uv_src[:, 0].reshape((h_out, w_out)).astype(np.float32)
    map2 = uv_src[:, 1].reshape((h_out, w_out)).astype(np.float32)
    img_out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC)

    return img_out


# --- 4. FINDING THE STICKERS ---
# Scans for ArUco markers (square barcodes).
# Ref: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
def find_aruco(img):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMax = 40
    params.useAruco3Detection = True
    corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary=aruco_dict, parameters=params)
    if ids is None:
        return {}
    else:
        # Save as a dictionary: {MarkerID: Pixel Coordinates}
        corners_dict = {ids[k][0]: corners[k][0, :, :] for k in range(len(ids))}
        return corners_dict

# --- 5. IDENTIFYING THE FRAME ---
# Checks the IDs of the markers to see if you're using 'Small', 'Medium', or 'Large'.
def identify_frame(img, config_frames, debug=False):
    corners_dict = find_aruco(img)

    if debug:
        img_view = np.copy(img)
        for c in corners_dict.values():
            for uv in c[:, :]:
                cv2.circle(img_view, uv.astype(np.int32), radius=30, color=(0, 0, 255), thickness=cv2.FILLED)
        imshow(img_view)

    name_found = None
    for name in config_frames:
        match = True
        for aruco_id in config_frames[name]["aruco_id"]:
            if aruco_id not in corners_dict:
                match = False
                break
        if match:
            name_found = name
            break
    return name_found


# --- 6. PINPOINTING THE DOTS ---
# Uses the rough ArUco location to find the exact center of dots with sub-pixel precision.
# Ref: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e4d78a8d167dfcc175d7903dbfa27
def get_corner_features(img_gray, proj, config):
    n_points = sum(len(edge) for edge in config["corner_pos"])
    xy_feats = np.zeros((n_points, 2))
    uv_feats_approx = np.zeros((n_points, 2))

    k = 0
    for edge in config["corner_pos"]:
        n_edge = len(edge)
        xy_feats[k:k + n_edge, :] = np.array(edge)
        uv_feats_approx[k:k + n_edge] = apply_affine(proj, xy_feats[k:k + n_edge, :])
        k += n_edge

    # Search window size for the dot center
    search_mm = 0.7 * config["corner_size"] / 2
    cross_xy = np.zeros((4 * n_points, 2), dtype=np.float32)
    cross_xy[0::4, :] = xy_feats - np.array([search_mm, 0])
    cross_xy[1::4, :] = xy_feats + np.array([search_mm, 0])
    cross_xy[2::4, :] = xy_feats - np.array([0, search_mm])
    cross_xy[3::4, :] = xy_feats + np.array([0, search_mm])

    cross_uv = apply_affine(proj, cross_xy)
    cross_uv_r = cross_uv.reshape(n_points, 4, 2)
    span_uv = (np.max(cross_uv_r, axis=1) - np.min(cross_uv_r, axis=1)) / 2
    search_uv = np.mean(span_uv, axis=0).astype(np.int32)

    # The "Digital Magnifying Glass"
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 0.001)
    ret = cv2.cornerSubPix(img_gray,
                           uv_feats_approx[:, np.newaxis, :].astype(np.float32),
                           (search_uv[0], search_uv[1]),
                           (-1, -1),
                           criteria)
    uv_feats = ret[:, 0, :]

    return xy_feats, uv_feats


# --- 7. MAIN ASSEMBLY LINE ---
# This orchestrates the whole process from start to finish.
def process_image(img, config_frames, solve_dist=False, view=False, view_radius=16, verbose=False, dpi=None):
    # Convert image to grayscale (B&W) so the computer can 'see' shapes better
    if len(img.shape) == 2:
        img_gray = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = img

    # Step A: Find the frame
    frame_name = identify_frame(img_rgb, config_frames)
    if frame_name is None:
        raise RuntimeError("No ArUco frame found in photo!")

    config = config_frames[frame_name]

    # Step B: Get rough perspective
    xy_a, uv_a = get_aruco_features(img_rgb, config)
    proj = misc.solve_affine(xy_a, uv_a)

    # Step C: Calculate Quality (DPI)
    if dpi is None:
        dpi = int(get_dots_per_mm(xy_a, uv_a) * 25.4)
    dots_per_mm = dpi / 25.4

    # Step D: Get precise perspective (Corner Dots)
    xy_c, uv_c = get_corner_features(img_gray, proj, config)
    proj_fine = misc.solve_affine(xy_c, uv_c)

    # Step E: Fix lens distortion and "Flatten" (The magic part)
    if solve_dist:
        params = solve_lens.solve_distortion(xy_c, uv_c, proj_fine, img.shape[1], img.shape[1], img.shape[0])
        img_out = extract_image(img_rgb, proj_fine, config, dots_per_mm, dist_params=params)
    else:
        img_out = extract_image(img_rgb, proj_fine, config, dots_per_mm)

    # Step F: Auto-rotate if it's upside down
    if uv_a[0][1] < uv_a[2][1]:
        img_out = cv2.rotate(img_out, cv2.ROTATE_180)

    return img_out, dpi


# --- 8. CONFIG LOADER ---
# Loads the rulebooks (settings) from JSON files.
def load_config_frames(filename):
    head, tail = os.path.split(filename)
    with open(filename, "r") as f:
        config_all = json.load(f)
    config = {}
    for frame_name, frame_filename in config_all.items():
        with open(os.path.join(head, frame_filename), "r") as f:
            config[frame_name] = json.load(f)
    return config


# --- 9. STARTING THE ENGINE ---
# This is where the code actually starts running.
def main():
    args, _ = parse_arguments()
    if args.input is None:
        print("Error: Please provide a photo! Use '-i <filename>'")
        return

    # Load everything
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    config_frames = load_config_frames(args.config)

    # Run the factory
    img_out, dpi = process_image(img, config_frames, solve_dist=True, view=args.show, verbose=args.verbose, dpi=args.dpi if args.dpi != -1 else None)

    # Decide where to save the result
    filename_out = args.output if args.output != "" else os.path.splitext(args.input)[0] + f"_{dpi}_DPI.png"
    
    # Save the file with scale info embedded
    misc.writePNGwithdpi(filename_out, img_out, dpi=(dpi, dpi))
    print(f"Done! Saved to: {filename_out}")


if __name__ == "__main__":
    main()
