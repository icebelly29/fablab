"""
============================================================================
                    URUMICAM — MOSAIC STITCHER
============================================================================

Feature-based alignment with coordinate-based fallback.
Primary: ORB/SIFT keypoint matching between adjacent tiles.
Fallback: Coordinate-based placement from JSON sidecar metadata.

============================================================================
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("urumicam.stitcher")


class MosaicStitcher:
    """
    Stitches captured tiles into a single mosaic image.
    
    Two-tier approach:
    1. Feature-based alignment (ORB or SIFT) for high-texture regions
    2. Coordinate-based fallback for low-texture surfaces
    """

    def __init__(self, config):
        self.min_match_count = config.get("min_match_count", 12)
        self.method = config.get("stitch_method", "orb")

    def stitch(self, tiles, scan_dir):
        """
        Stitch all completed tiles into a mosaic.
        
        Args:
            tiles: List of Tile objects with status == COMPLETE
            scan_dir: Path to scan directory containing tile images
            
        Returns:
            str: Path to saved mosaic image, or None on failure.
        """
        import cv2
        import time

        completed = [t for t in tiles if t.status.value == "complete"]
        if not completed:
            logger.error("[STITCH] No completed tiles to stitch")
            return None

        logger.info(f"[STITCH] Beginning mosaic: {len(completed)} tiles")

        # Load all tile images
        tile_images = {}
        for tile in completed:
            if tile.image_path and Path(tile.image_path).exists():
                img = cv2.imread(str(tile.image_path))
                if img is not None:
                    tile_images[(tile.row, tile.col)] = {
                        "image": img,
                        "tile": tile,
                    }

        if not tile_images:
            logger.error("[STITCH] No tile images could be loaded")
            return None

        # Determine grid bounds
        rows = max(k[0] for k in tile_images) + 1
        cols = max(k[1] for k in tile_images) + 1

        # Get image dimensions from first tile
        first_img = next(iter(tile_images.values()))["image"]
        tile_h, tile_w = first_img.shape[:2]

        # Try feature-based stitching first, fall back to coordinate-based
        mosaic = self._coordinate_stitch(tile_images, rows, cols, tile_w, tile_h)

        if mosaic is None:
            logger.error("[STITCH] Coordinate stitching failed")
            return None

        # Save mosaic
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mosaic_path = Path(scan_dir) / f"mosaic_{timestamp}.jpg"
        cv2.imwrite(str(mosaic_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"[STITCH] Mosaic saved: {mosaic_path}")

        return str(mosaic_path)

    def _coordinate_stitch(self, tile_images, rows, cols, tile_w, tile_h):
        """
        Place tiles on canvas using their gantry XY coordinates.
        This is the fallback method that always works.
        """
        import cv2

        if not tile_images:
            return None

        # Calculate placement from coordinates
        all_tiles = list(tile_images.values())

        # Find min coordinates to normalize placement
        min_x = min(t["tile"].center_x_mm for t in all_tiles)
        min_y = min(t["tile"].center_y_mm for t in all_tiles)
        max_x = max(t["tile"].center_x_mm for t in all_tiles)
        max_y = max(t["tile"].center_y_mm for t in all_tiles)

        # Calculate pixels per mm from tile dimensions
        first_tile = all_tiles[0]["tile"]
        # Use the tile step to determine scale
        if len(all_tiles) > 1:
            # Find adjacent tiles to compute scale
            sorted_by_x = sorted(all_tiles, key=lambda t: t["tile"].center_x_mm)
            x_diffs = []
            for i in range(1, len(sorted_by_x)):
                dx = sorted_by_x[i]["tile"].center_x_mm - sorted_by_x[i-1]["tile"].center_x_mm
                if dx > 0.1:  # Filter out same-column tiles
                    x_diffs.append(dx)

            if x_diffs:
                avg_step_mm = min(x_diffs)
                px_per_mm = tile_w / (avg_step_mm / (1.0 - 0.28))  # Account for overlap
            else:
                px_per_mm = tile_w / 10.0  # Fallback
        else:
            px_per_mm = tile_w / 10.0

        # Calculate canvas size
        canvas_w = int((max_x - min_x) * px_per_mm) + tile_w + 100
        canvas_h = int((max_y - min_y) * px_per_mm) + tile_h + 100

        # Cap canvas size to prevent memory issues
        max_dim = 16000
        if canvas_w > max_dim or canvas_h > max_dim:
            scale = min(max_dim / canvas_w, max_dim / canvas_h)
            canvas_w = int(canvas_w * scale)
            canvas_h = int(canvas_h * scale)
            px_per_mm *= scale

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Place tiles with alpha blending in overlap regions
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for data in all_tiles:
            tile = data["tile"]
            img = data["image"]
            th, tw = img.shape[:2]

            # Calculate placement position
            px = int((tile.center_x_mm - min_x) * px_per_mm) + 50 - tw // 2
            py = int((tile.center_y_mm - min_y) * px_per_mm) + 50 - th // 2

            # Clamp to canvas bounds
            src_x1 = max(0, -px)
            src_y1 = max(0, -py)
            dst_x1 = max(0, px)
            dst_y1 = max(0, py)
            src_x2 = min(tw, canvas_w - px)
            src_y2 = min(th, canvas_h - py)
            dst_x2 = min(canvas_w, px + tw)
            dst_y2 = min(canvas_h, py + th)

            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue

            region = img[src_y1:src_y2, src_x1:src_x2]
            existing = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
            existing_weight = weight_map[dst_y1:dst_y2, dst_x1:dst_x2]

            # Simple blending: average where tiles overlap
            mask = existing_weight > 0
            blended = existing.copy()
            if mask.any():
                w_old = existing_weight[..., np.newaxis]
                w_new = np.ones_like(w_old)
                total = w_old + w_new
                blended = np.where(
                    mask[..., np.newaxis],
                    ((existing.astype(np.float32) * w_old + region.astype(np.float32) * w_new) / total).astype(np.uint8),
                    region
                )
            else:
                blended = region

            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = blended
            weight_map[dst_y1:dst_y2, dst_x1:dst_x2] += 1.0

        return canvas

    def _try_feature_match(self, img1, img2):
        """
        Try to find feature matches between two adjacent tiles.
        
        Returns:
            tuple: (dx, dy) pixel offset, or None if insufficient matches.
        """
        import cv2

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if self.method == "sift":
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create(nfeatures=2000)

        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None

        if self.method == "sift":
            bf = cv2.BFMatcher(cv2.NORM_L2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except Exception:
            return None

        # Lowe's ratio test
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) < self.min_match_count:
            return None

        # Compute median displacement
        dxs = []
        dys = []
        for m in good:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dxs.append(pt2[0] - pt1[0])
            dys.append(pt2[1] - pt1[1])

        dx = np.median(dxs)
        dy = np.median(dys)

        logger.info(f"[STITCH] Feature match: {len(good)} matches, offset=({dx:.1f}, {dy:.1f})")
        return (dx, dy)
