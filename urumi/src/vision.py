import cv2
import numpy as np
import math

def distance_to_center(contour, center):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return math.hypot(cx - center[0], cy - center[1])
    return float('inf')

def get_closest_point(contour, center):
    min_dist = float('inf')
    closest_pt = None
    closest_idx = -1
    
    # Squeeze contour to 2D array of points
    pts = contour.squeeze()
    if pts.ndim == 1:
        pts = np.array([pts])
        
    for i, pt in enumerate(pts):
        dist = math.hypot(pt[0] - center[0], pt[1] - center[1])
        if dist < min_dist:
            min_dist = dist
            closest_pt = pt
            closest_idx = i
            
    return closest_pt, closest_idx

def get_tangent_vector(contour, index, step=5):
    pts = contour.squeeze()
    if pts.ndim == 1 or len(pts) < 2:
        return np.array([1.0, 0.0]) # default
        
    n = len(pts)
    pt1 = pts[(index - step) % n]
    pt2 = pts[(index + step) % n]
    
    vec = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1.0, 0.0])
    return vec / norm

def process_edges(frame, mm_per_px=0.1, Kp=0.5, step_mm=2.0):
    """
    Returns dx_mm, dy_mm for the next movement, and a visualization frame.
    """
    center = (frame.shape[1]//2, frame.shape[0]//2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    dx_mm, dy_mm = 0.0, 0.0
    vis_frame = frame.copy()
    
    # Draw crosshair
    cv2.drawMarker(vis_frame, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    
    if contours:
        # Filter tiny contours
        contours = [c for c in contours if cv2.contourArea(c) > 50]
        if contours:
            best_contour = min(contours, key=lambda c: distance_to_center(c, center))
            
            cv2.drawContours(vis_frame, [best_contour], -1, (255, 0, 0), 2)
            
            P, index = get_closest_point(best_contour, center)
            if P is not None:
                T = get_tangent_vector(best_contour, index)
                
                E_x = P[0] - center[0]
                E_y = P[1] - center[1]
                
                dx_mm = (Kp * E_x * mm_per_px) + (T[0] * step_mm)
                dy_mm = (Kp * E_y * mm_per_px) + (T[1] * step_mm)
                
                # Draw closest point
                cv2.circle(vis_frame, tuple(P), 5, (0, 0, 255), -1)
                
                # Draw error vector (Yellow)
                cv2.arrowedLine(vis_frame, center, tuple(P), (0, 255, 255), 2)
                
                # Draw tangent vector (Magenta)
                end_pt = (int(P[0] + T[0]*50), int(P[1] + T[1]*50))
                cv2.arrowedLine(vis_frame, tuple(P), end_pt, (255, 0, 255), 2)
                
    return dx_mm, dy_mm, vis_frame
