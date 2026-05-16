"""
============================================================================
                    URUMICAM — OFFLINE SCAN STITCHER
============================================================================

Standalone CLI tool to stitch scan tiles using gantry coordinates.
Supports feathering and alpha blending for seamless joins.

Usage:
    python stitch_scan.py path/to/scan_folder
    python stitch_scan.py scans/scan_2026-05-15 --scale 0.5

============================================================================
"""

import os
import json
import argparse
import logging
import time
import numpy as np
import cv2
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("stitcher")

def stitch_folder(scan_dir, scale=1.0, quality=95, feather=20, fov_x_override=None, fov_y_override=None, flip_x=False, flip_y=False, use_cv=False):
    scan_dir = Path(scan_dir)
    manifest_path = scan_dir / "scan_manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return None

    logger.info(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    tiles = manifest.get("tiles", [])
    
    # Filter only completed tiles
    completed_tiles = [t for t in tiles if t.get("status") == "complete"]
    if not completed_tiles:
        logger.error("No completed tiles found in manifest")
        return None

    logger.info(f"Processing {len(completed_tiles)} tiles...")

    # Load first image to get dimensions
    first_tile_path = scan_dir / Path(completed_tiles[0]["image_path"]).name
    first_img = cv2.imread(str(first_tile_path))
    if first_img is None:
        logger.error(f"Could not load first tile: {first_tile_path}")
        return None
    
    tile_h, tile_w = first_img.shape[:2]
    
    # Determine FOV
    fov_x = fov_x_override if fov_x_override else config.get("tile_fov_x_mm", 10.0)
    fov_y = fov_y_override if fov_y_override else config.get("tile_fov_y_mm", 7.5)
    logger.info(f"Using FOV: {fov_x:.2f} x {fov_y:.2f} mm")

    px_per_mm_x = tile_w / fov_x
    px_per_mm_y = tile_h / fov_y
    px_per_mm_x *= scale
    px_per_mm_y *= scale
    scaled_w = int(tile_w * scale)
    scaled_h = int(tile_h * scale)

    # Find bounds
    xs = [t["center_x_mm"] for t in completed_tiles]
    ys = [t["center_y_mm"] for t in completed_tiles]
    if flip_x: xs = [-x for x in xs]
    if flip_y: ys = [-y for y in ys]
    
    min_x_mm, min_y_mm = min(xs), min(ys)
    
    canvas_w = int((max(xs) - min_x_mm) * px_per_mm_x) + scaled_w + 100
    canvas_h = int((max(ys) - min_y_mm) * px_per_mm_y) + scaled_h + 100
    logger.info(f"Canvas size: {canvas_w}x{canvas_h}")
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # ORB detector for CV mode
    orb = cv2.ORB_create(nfeatures=2000) if use_cv else None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if use_cv else None

    # Create feathering mask
    mask = np.ones((scaled_h, scaled_w), dtype=np.float32)
    if feather > 0:
        f = int(feather * scale)
        if f > 0:
            for i in range(f):
                v = i / f
                mask[i, :] *= v
                mask[-1-i, :] *= v
                mask[:, i] *= v
                mask[:, -1-i] *= v

    for i, t in enumerate(completed_tiles):
        img_path = scan_dir / Path(t["image_path"]).name
        img = cv2.imread(str(img_path))
        if img is None: continue
        if scale != 1.0:
            img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        cx = -t["center_x_mm"] if flip_x else t["center_x_mm"]
        cy = -t["center_y_mm"] if flip_y else t["center_y_mm"]

        # 1. Start with coordinate-based position
        x_start = int((cx - min_x_mm) * px_per_mm_x) + 50
        y_start = int((cy - min_y_mm) * px_per_mm_y) + 50

        # 2. Refine with Computer Vision if enabled
        if use_cv and i > 0:
            # Check overlap with existing canvas
            overlap_x1 = max(0, x_start)
            overlap_y1 = max(0, y_start)
            overlap_x2 = min(canvas_w, x_start + scaled_w)
            overlap_y2 = min(canvas_h, y_start + scaled_h)
            
            # Extract current background
            bg_region = canvas[overlap_y1:overlap_y2, overlap_x1:overlap_x2].astype(np.uint8)
            bg_weight = weight_map[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
            
            # Only attempt CV match if there's significant overlap and existing data
            if bg_weight.mean() > 0.1:
                kp1, des1 = orb.detectAndCompute(img, None)
                kp2, des2 = orb.detectAndCompute(bg_region, None)
                
                if des1 is not None and des2 is not None:
                    matches = matcher.match(des1, des2)
                    if len(matches) > 15:
                        # Compute median offset
                        dxs = [kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in matches]
                        dys = [kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in matches]
                        
                        # Add relative offset to coordinate position
                        # (Matches are relative to overlap region origin)
                        x_start += int(np.median(dxs)) - (overlap_x1 - x_start)
                        y_start += int(np.median(dys)) - (overlap_y1 - y_start)

        # 3. Draw to canvas (with updated coordinates)
        th, tw = img.shape[:2]
        dst_x1, dst_y1 = max(0, x_start), max(0, y_start)
        dst_x2, dst_y2 = min(canvas_w, x_start + tw), min(canvas_h, y_start + th)
        
        src_x1, src_y1 = dst_x1 - x_start, dst_y1 - y_start
        src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            region = img[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
            m_slice = mask[src_y1:src_y2, src_x1:src_x2, np.newaxis]
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] += region * m_slice
            weight_map[dst_y1:dst_y2, dst_x1:dst_x2] += m_slice.squeeze()

    # Normalize and save
    nonzero = weight_map > 0
    canvas[nonzero] /= weight_map[nonzero][..., np.newaxis]
    final_mosaic = np.clip(canvas, 0, 255).astype(np.uint8)
    
    prefix = "mosaic_cv" if use_cv else "mosaic_calibrated"
    out_path = scan_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(out_path), final_mosaic, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    logger.info(f"Stitching complete! Saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinate-based scan stitcher with CV refinement")
    parser.add_argument("folder", help="Path to the scan folder")
    parser.add_argument("--scale", type=float, default=1.0, help="Downscale factor (0.1 to 1.0)")
    parser.add_argument("--feather", type=int, default=30, help="Feathering edge size in pixels")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--fov_x", type=float, help="Manual Field of View X (mm)")
    parser.add_argument("--fov_y", type=float, help="Manual Field of View Y (mm)")
    parser.add_argument("--flip_x", action="store_true", help="Invert X axis coordinates")
    parser.add_argument("--flip_y", action="store_true", help="Invert Y axis coordinates")
    parser.add_argument("--auto", action="store_true", help="Use Computer Vision to snap tiles into alignment")
    
    args = parser.parse_args()
    stitch_folder(args.folder, 
                  scale=args.scale, 
                  feather=args.feather, 
                  quality=args.quality,
                  fov_x_override=args.fov_x,
                  fov_y_override=args.fov_y,
                  flip_x=args.flip_x,
                  flip_y=args.flip_y,
                  use_cv=args.auto)
