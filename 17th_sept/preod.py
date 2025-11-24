"""
padayani_three_eyes.py

Prototype:
- Camera feed (Mediapipe face detection + face mesh)
- Eyes window with 3 pairs of circular eyes in a column
- Hybrid matching: centroid + IoU + optional face embeddings
- Hybrid C + D assignment logic with dynamic filling of freed slots

Dependencies:
  pip install opencv-python mediapipe
  optional: pip install face_recognition  (for stronger identity matching)

Press q to quit.
"""

import time
import math
from collections import deque, OrderedDict
import numpy as np
import cv2
import mediapipe as mp

# Try to import face_recognition for embeddings. If not present, we'll skip embeddings.
### This is too resource intensive on some systems, so optional.
try:
    # import face_recognition
    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False

# --- Configuration ---
EYE_WIN_W = 360
EYE_WIN_H = 600
EYE_POSITIONS = [(180, 120), (180, 300), (180, 480)]  # centers for each pair in the eye window
EYE_RADIUS = 50
PUPIL_RADIUS = 16
PUPIL_RANGE = 22  # max pixel offset pupils can move inside each eyeball
SMOOTHING = 0.5  # [0..1], higher = slower smoothing for pupils
LOOK_YAW_THRESHOLD = 0.30  # how strict we are about someone "looking" at the installation
IOU_MATCH_THRESHOLD = 0.35
CENTROID_DIST_THRESHOLD = 120  # pixels, fallback
EMBEDDING_SIMILARITY_THRESHOLD = 0.45  # lower is more similar for face_recognition (use distance)
MAX_TRACK_AGE = 1.5  # seconds before we consider a tracked person gone

# --- Mediapipe ---
mp_face_detect = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# --- Utilities ---
def iou(boxA, boxB):
    # boxes are (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    unionArea = boxAA + boxBB - interArea
    if unionArea == 0:
        return 0.0
    return interArea / float(unionArea)

def box_center(box):
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def box_area(box):
    x1,y1,x2,y2 = box
    return max(0, (x2-x1)) * max(0, (y2-y1))

def clamp(x, a, b):
    return max(a, min(b, x))

# --- Tracker structures ---
class TrackedPerson:
    def __init__(self, person_id, bbox, embedding, timestamp):
        self.id = person_id
        self.bbox = bbox  # absolute pixel coords x1,y1,x2,y2
        self.center = box_center(bbox)
        self.area = box_area(bbox)
        self.embedding = embedding  # None if not available
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.assigned_slot = None  # 1,2,3 or None
        self.looking = False
        # pupil smoothing state for eyes (dx, dy target and smoothed)
        self.pupil_target = (0.0, 0.0)
        self.pupil_smoothed = (0.0, 0.0)

    def update(self, bbox, embedding, timestamp):
        self.bbox = bbox
        self.center = box_center(bbox)
        self.area = box_area(bbox)
        if embedding is not None:
            self.embedding = embedding
        self.last_seen = timestamp

    def age(self, now):
        return now - self.last_seen

# --- Matching algorithm: embeddings -> IoU -> centroid ---
def match_detections_to_tracked(detections, tracked_people, now):
    """
    detections: list of dicts: {'bbox':(x1,y1,x2,y2), 'embedding':arr or None, 'center':(cx,cy), 'area':a, 'looking':bool}
    tracked_people: dict person_id -> TrackedPerson
    returns:
      matches: dict detection_index -> person_id (for assigned)
      unmatched_dets: list of detection indices not matched
      unmatched_tracks: list of person_ids not matched
    """
    matches = {}
    unmatched_dets = list(range(len(detections)))
    unmatched_tracks = list(tracked_people.keys())

    # 1) Embedding-based matching (strongest), if embeddings exist
    if HAS_EMBEDDINGS:
        # compute distance matrix between detections with embeddings and tracked embeddings
        for di, det in enumerate(detections):
            if det['embedding'] is None:
                continue
            best_pid = None
            best_dist = 999
            for pid in list(unmatched_tracks):
                tp = tracked_people[pid]
                if tp.embedding is None:
                    continue
                # face_recognition uses euclidean distance on 128-d vectors
                dist = np.linalg.norm(det['embedding'] - tp.embedding)
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid
            if best_pid is not None and best_dist < EMBEDDING_SIMILARITY_THRESHOLD:
                # match
                matches[di] = best_pid
                if di in unmatched_dets:
                    unmatched_dets.remove(di)
                if best_pid in unmatched_tracks:
                    unmatched_tracks.remove(best_pid)

    # 2) IoU-based matching
    # For remaining dets and tracks, greedily match by IoU
    if unmatched_dets and unmatched_tracks:
        iou_pairs = []
        for di in unmatched_dets:
            for pid in unmatched_tracks:
                val = iou(detections[di]['bbox'], tracked_people[pid].bbox)
                if val > 0.0:
                    iou_pairs.append((val, di, pid))
        iou_pairs.sort(reverse=True, key=lambda x: x[0])  # best IoU first
        used_d = set()
        used_p = set()
        for val, di, pid in iou_pairs:
            if di in used_d or pid in used_p:
                continue
            if val >= IOU_MATCH_THRESHOLD:
                matches[di] = pid
                used_d.add(di)
                used_p.add(pid)
        # remove used ones from unmatched lists
        for di in used_d:
            if di in unmatched_dets:
                unmatched_dets.remove(di)
        for pid in used_p:
            if pid in unmatched_tracks:
                unmatched_tracks.remove(pid)

    # 3) Centroid distance fallback
    if unmatched_dets and unmatched_tracks:
        cent_pairs = []
        for di in unmatched_dets:
            for pid in unmatched_tracks:
                cx,cy = detections[di]['center']
                tcx,tcy = tracked_people[pid].center
                dist = math.hypot(cx - tcx, cy - tcy)
                cent_pairs.append((dist, di, pid))
        cent_pairs.sort(key=lambda x: x[0])  # nearest first
        used_d = set()
        used_p = set()
        for dist, di, pid in cent_pairs:
            if di in used_d or pid in used_p:
                continue
            if dist <= CENTROID_DIST_THRESHOLD:
                matches[di] = pid
                used_d.add(di)
                used_p.add(pid)
        for di in used_d:
            if di in unmatched_dets:
                unmatched_dets.remove(di)
        for pid in used_p:
            if pid in unmatched_tracks:
                unmatched_tracks.remove(pid)

    return matches, unmatched_dets, unmatched_tracks

# --- Assignment logic per your final table and dynamic filling ---
def assign_slots(tracked_people, now):
    """
    Ensure tracked_people have assigned_slot according to:
    - Consider only people who are looking
    - Order is: for newly unassigned slots, pick closest unassigned person
    - For counts, enforce the table:
        1 person -> all pairs track P1
        2 persons -> P1, P2, P2
        3+ -> P1, P2, P3
    tracked_people: dict id -> TrackedPerson
    """
    # filter only currently looking people and not too old
    active = [tp for tp in tracked_people.values() if tp.looking and tp.age(now) <= MAX_TRACK_AGE]
    if not active:
        # clear assignments
        for tp in tracked_people.values():
            tp.assigned_slot = None
        return

    # Sort by first_seen time for FCFS lock, but ensure initial assignment uses closeness for new unassigned people.
    # We'll compute two orderings:
    # 1) locked order: persons currently assigned keep that relative priority (by their first_seen)
    # 2) unassigned candidates sorted by area (closest first)
    assigned = {tp.assigned_slot: tp for tp in active if tp.assigned_slot is not None}
    unassigned = [tp for tp in active if tp.assigned_slot is None]

    # For initial filling: sort unassigned by area desc (closest first)
    unassigned.sort(key=lambda t: t.area, reverse=True)

    # If some slots are missing but we have fewer active than slots, we will result to the canonical mapping later.
    # Ensure assigned slots remain unless their person vanished or stopped looking.

    # Available slot indices 1..3
    slot_indices = [1,2,3]
    free_slots = [s for s in slot_indices if s not in assigned]

    # Fill free slots by closest unassigned people
    for s in free_slots:
        if not unassigned:
            break
        tp = unassigned.pop(0)
        tp.assigned_slot = s
        assigned[s] = tp

    # At this point assigned dict has the persons assigned to some slots (may be < 3).
    # Build list of currently assigned people sorted by slot index
    assigned_list = [assigned[s] for s in slot_indices if s in assigned]

    # Count of distinct people
    n = len(assigned_list)

    # If less than 3 active people exist, enforce canonical mapping
    active_sorted_by_firstseen = sorted(active, key=lambda t: t.first_seen)  # FCFS order for canonical table
    if len(active_sorted_by_firstseen) == 1:
        p1 = active_sorted_by_firstseen[0]
        p1.assigned_slot = 1
        p1.assigned_slot = 1
        # we want all slots to point to this person logically; representation keep p1 assigned to slot1, others None
        # The drawing function will map pairs according to table rules
    elif len(active_sorted_by_firstseen) == 2:
        # ensure first person has slot1 and second person has slot2
        p1, p2 = active_sorted_by_firstseen[0], active_sorted_by_firstseen[1]
        p1.assigned_slot = 1
        p2.assigned_slot = 2
    else:
        # 3 or more active people: ensure first three by FCFS occupy slots 1..3 if not already assigned
        for idx in range(3):
            if idx < len(active_sorted_by_firstseen):
                tp = active_sorted_by_firstseen[idx]
                tp.assigned_slot = idx + 1

    # At the end, if some slots are still empty, they remain empty and the drawing stage will handle fallback mapping.

# --- Drawing logic for eyes based on assigned targets ---
def compute_pupil_offsets_for_display(tp, frame_w, frame_h):
    """
    Convert a tracked person's center into a normalized dx,dy in [-1,1]
    relative to camera center for pupil movement.
    We also perform smoothing on tp.pupil_smoothed for stability.
    """
    if tp is None:
        return 0.0, 0.0
    cx, cy = tp.center
    dx = (cx - frame_w / 2) / (frame_w / 2)
    dy = (cy - frame_h / 2) / (frame_h / 2)
    dx = clamp(dx, -1.0, 1.0)
    dy = clamp(dy, -1.0, 1.0)
    # smoothing (exponential)
    sx, sy = tp.pupil_smoothed
    nx = sx * (1 - SMOOTHING) + dx * SMOOTHING
    ny = sy * (1 - SMOOTHING) + dy * SMOOTHING
    tp.pupil_smoothed = (nx, ny)
    return nx, ny

def draw_eye_pair(win, center_x, center_y, dx, dy):
    # left eye center, right eye center
    gap = 70
    lx, ly = center_x - gap//2, center_y
    rx, ry = center_x + gap//2, center_y
    # white eyeballs
    cv2.circle(win, (lx, ly), EYE_RADIUS, (255,255,255), -1)
    cv2.circle(win, (rx, ry), EYE_RADIUS, (255,255,255), -1)
    # limit pupil movement inside the eyeball radius - pupil_range
    px = int(dx * PUPIL_RANGE)
    py = int(dy * PUPIL_RANGE)
    # pupils
    cv2.circle(win, (lx+px, ly+py), PUPIL_RADIUS, (10,10,10), -1)
    cv2.circle(win, (rx+px, ry+py), PUPIL_RADIUS, (10,10,10), -1)
    # subtle pupil rim
    cv2.circle(win, (lx+px, ly+py), PUPIL_RADIUS, (40,40,40), 1)
    cv2.circle(win, (rx+px, ry+py), PUPIL_RADIUS, (40,40,40), 1)

# --- Main application ---
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Prepare windows
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Padayani Eyes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Padayani Eyes", EYE_WIN_W, EYE_WIN_H)

    # tracker containers
    tracked_people = OrderedDict()  # person_id -> TrackedPerson
    next_person_id = 1

    # For matching we will use face detection + face mesh for landmarks
    face_detector = mp_face_detect.FaceDetection(model_selection=0, min_detection_confidence=0.55)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=6, min_detection_confidence=0.5)

    last_cleanup = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            now = time.time()

            # run mediapipe face detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det_results = face_detector.process(rgb)
            mesh_results = face_mesh.process(rgb)

            detections = []  # collect face detections with bbox, center, area, embedding, looking boolean

            if det_results.detections:
                # build lookup for mesh landmarks keyed by approximate center (since mediapipe detection and mesh may differ)
                mesh_by_center = []
                if mesh_results.multi_face_landmarks:
                    for fl in mesh_results.multi_face_landmarks:
                        # compute approximate center from landmarks
                        pts = [(int(lm.x*frame_w), int(lm.y*frame_h)) for lm in fl.landmark]
                        cx = int(np.mean([p[0] for p in pts]))
                        cy = int(np.mean([p[1] for p in pts]))
                        mesh_by_center.append((cx, cy, fl))

                for det in det_results.detections:
                    # relative bbox
                    r = det.location_data.relative_bounding_box
                    x1 = int(r.xmin * frame_w)
                    y1 = int(r.ymin * frame_h)
                    bw = int(r.width * frame_w)
                    bh = int(r.height * frame_h)
                    x2 = x1 + bw
                    y2 = y1 + bh
                    # clamp
                    x1 = clamp(x1, 0, frame_w-1)
                    y1 = clamp(y1, 0, frame_h-1)
                    x2 = clamp(x2, 0, frame_w-1)
                    y2 = clamp(y2, 0, frame_h-1)
                    bbox = (x1,y1,x2,y2)
                    cx, cy = box_center(bbox)
                    area = box_area(bbox)

                    # embed face if possible (use face_recognition)
                    embedding = None
                    if HAS_EMBEDDINGS:
                        # extract face crop and compute encoding
                        try:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size != 0:
                                # face_recognition expects RGB
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                # encs = face_recognition.face_encodings(crop_rgb)
                                # if encs:
                                    # embedding = encs[0]
                        except Exception:
                            embedding = None

                    # find mesh landmarks that match this detection (closest center)
                    matched_mesh = None
                    if mesh_by_center:
                        dists = [(math.hypot(cx - m[0], cy - m[1]), m[2]) for m in mesh_by_center]
                        dists.sort(key=lambda x: x[0])
                        if dists and dists[0][0] < max(bw, bh)*0.9:
                            matched_mesh = dists[0][1]

                    # compute simple "looking" metric using nose x offset vs bbox center
                    is_looking = False
                    if matched_mesh:
                        # use nose tip (landmark 1) against bbox center to estimate yaw
                        # Mediapipe index 1 is approximate. We'll fallback on averaged nose landmarks.
                        lm = matched_mesh.landmark[1] if len(matched_mesh.landmark) > 1 else None
                        if lm is not None:
                            nose_x = lm.x * frame_w
                            # yaw estimate: normalized distance from box center
                            yaw = (nose_x - cx) / max(1, bw)
                            if abs(yaw) <= LOOK_YAW_THRESHOLD:
                                is_looking = True
                        else:
                            # fallback: consider it looking
                            is_looking = True
                    else:
                        # if no mesh, use bbox center proximity to frame center as a weak proxy
                        yaw = (cx - frame_w/2) / max(1, bw)
                        is_looking = abs(yaw) <= LOOK_YAW_THRESHOLD

                    detections.append({
                        'bbox': bbox,
                        'center': (cx, cy),
                        'area': area,
                        'embedding': embedding,
                        'looking': is_looking
                    })

            # match detections to tracked people
            if detections:
                matches, unmatched_dets, unmatched_tracks = match_detections_to_tracked(detections, tracked_people, now)
            else:
                matches, unmatched_dets, unmatched_tracks = {}, [], list(tracked_people.keys())

            # update matched tracked persons
            for di, pid in matches.items():
                det = detections[di]
                tp = tracked_people.get(pid)
                if tp:
                    tp.update(det['bbox'], det['embedding'], now)
                    tp.looking = det['looking']

            # create new tracked persons for unmatched detections
            for di in unmatched_dets:
                det = detections[di]
                # new person id
                pid = next_person_id = (max(tracked_people.keys()) + 1) if tracked_people else 1
                tp = TrackedPerson(pid, det['bbox'], det['embedding'], now)
                tp.looking = det['looking']
                tracked_people[pid] = tp

            # stale / disappeared tracked cleanup: if age > MAX_TRACK_AGE remove, but we keep brief tolerance
            to_remove = []
            for pid, tp in list(tracked_people.items()):
                if tp.age(now) > MAX_TRACK_AGE:
                    # free slot if assigned
                    if tp.assigned_slot is not None:
                        # free it
                        tp.assigned_slot = None
                    # mark for removal
                    to_remove.append(pid)
            for pid in to_remove:
                tracked_people.pop(pid, None)

            # run assignment logic
            assign_slots(tracked_people, now)

            # Build mapping from slot->person (for display) using your canonical table
            # First gather active looking persons sorted by first_seen (FCFS)
            active_looking = [tp for tp in tracked_people.values() if tp.looking and tp.age(now) <= MAX_TRACK_AGE]
            active_looking.sort(key=lambda t: t.first_seen)

            # Build canonical slots mapping using only currently active people
            canonical = {}  # slot -> tp or None
            if len(active_looking) == 0:
                canonical = {1: None, 2: None, 3: None}
            elif len(active_looking) == 1:
                p1 = active_looking[0]
                canonical = {1: p1, 2: p1, 3: p1}
            elif len(active_looking) == 2:
                p1, p2 = active_looking[0], active_looking[1]
                canonical = {1: p1, 2: p2, 3: p2}
            else:
                p1, p2, p3 = active_looking[0], active_looking[1], active_looking[2]
                canonical = {1: p1, 2: p2, 3: p3}

            # Now ensure that freed slots get filled by the next available person by closest size if needed
            # Find which slots are empty (None) and fill with nearest unassigned looking people by area
            slots_filled = {}
            assigned_person_ids = set()
            for s in (1,2,3):
                tp = canonical.get(s)
                if tp is not None:
                    slots_filled[s] = tp
                    assigned_person_ids.add(tp.id)
                else:
                    slots_filled[s] = None
            # find unassigned candidates
            candidates = [tp for tp in active_looking if tp.id not in assigned_person_ids]
            candidates.sort(key=lambda t: t.area, reverse=True)
            for s in (1,2,3):
                if slots_filled[s] is None and candidates:
                    tp = candidates.pop(0)
                    slots_filled[s] = tp
                    assigned_person_ids.add(tp.id)

            # finally set each tracked person's assigned_slot for bookkeeping
            for tp in tracked_people.values():
                tp.assigned_slot = None
            for s, tp in slots_filled.items():
                if tp:
                    tp.assigned_slot = s

            # draw camera feed overlays
            vis = frame.copy()
            # draw all detection boxes and ids
            for pid, tp in tracked_people.items():
                x1,y1,x2,y2 = tp.bbox
                color = (0,200,0) if tp.looking else (120,120,120)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cx,cy = tp.center
                cv2.circle(vis, (cx,cy), 3, color, -1)
                cv2.putText(vis, f"ID:{tp.id} s:{tp.assigned_slot or 0}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # draw lines from eye slots to assigned persons for debugging
            for s in (1,2,3):
                tp = slots_filled.get(s)
                if tp:
                    # compute offset for debug line
                    # map slot s to eye position on screen (small inset)
                    ex = int(frame_w - 120)
                    ey = int(100 + (s-1)*120)
                    cv2.line(vis, (ex,ey), tp.center, (200,100,20), 1)

            cv2.imshow("Camera Feed", vis)

            # build eyes window
            eye_win = np.zeros((EYE_WIN_H, EYE_WIN_W, 3), dtype=np.uint8)
            eye_win[:] = (50,50,60)

            # draw title
            cv2.putText(eye_win, "Padayani Eyes", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,220), 2)

            # For each pair compute dx,dy target from assigned person
            for idx, (ex,ey) in enumerate(EYE_POSITIONS):
                slot = idx + 1
                tp = slots_filled.get(slot)
                if tp is not None:
                    dx, dy = compute_pupil_offsets_for_display(tp, frame_w, frame_h)
                else:
                    # neutral
                    dx, dy = 0.0, 0.0
                draw_eye_pair(eye_win, ex, ey, dx, dy)
                # label with slot and person id
                label = f"S{slot}: {tp.id}" if tp else f"S{slot}: -"
                cv2.putText(eye_win, label, (ex - 60, ey + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Padayani Eyes", eye_win)

            # housekeeping: remove old tracked entries periodically
            if now - last_cleanup > 1.0:
                to_kill = []
                for pid, tp in tracked_people.items():
                    if tp.age(now) > (MAX_TRACK_AGE * 2):
                        to_kill.append(pid)
                for pid in to_kill:
                    tracked_people.pop(pid, None)
                last_cleanup = now

            # ESC to quit or q
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        face_detector.close()
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
