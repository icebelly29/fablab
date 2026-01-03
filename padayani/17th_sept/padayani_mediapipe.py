import cv2
import mediapipe as mp
import numpy as np

# ---------------- Configuration ----------------

# Perimeter region (relative coordinates 0–1, centered area of the frame)
PERIMETER_X_MIN = 0.0
PERIMETER_X_MAX = 0.99
PERIMETER_Y_MIN = 0.0
PERIMETER_Y_MAX = 0.99

# Minimum face area as fraction of the full frame (filters out people too far away)
MIN_FACE_AREA_FRAC = 0.01

# Distance threshold (pixels) for associating detections with existing tracks
TRACK_MATCH_DIST_PX = 10.0

# Eye drawing settings
EYE_CANVAS_WIDTH = 900
EYE_CANVAS_HEIGHT = 300
EYE_RADIUS = 50
PUPIL_RADIUS = 15
PUPIL_MAX_OFFSET_FACTOR = 0.45  # how far pupils can move inside the eye (horizontal)
SMOOTHING_FACTOR = 0.25         # 0–1, higher is faster following

# Global track ID counter
track_id_counter = 0


# ---------------- Helpers: Eyes ----------------

class Eye:
    def __init__(self, center, radius_eye, radius_pupil, pair_index):
        self.center = np.array(center, dtype=np.float32)
        self.radius_eye = radius_eye
        self.radius_pupil = radius_pupil
        self.pupil_pos = np.array(center, dtype=np.float32)
        self.pair_index = pair_index

    def update(self, norm_target, max_offset, smooth=0.2):
        # norm_target.x in [-1,1], norm_target.y is ignored or small
        # We only use horizontal movement to emphasize left-right tracking
        desired = self.center + np.array(
            [norm_target[0] * max_offset, 0.0],
            dtype=np.float32
        )
        self.pupil_pos = (1.0 - smooth) * self.pupil_pos + smooth * desired

    def draw(self, canvas):
        cx, cy = int(self.center[0]), int(self.center[1])
        px, py = int(self.pupil_pos[0]), int(self.pupil_pos[1])

        # Background of eyeball (white)
        cv2.circle(canvas, (cx, cy), self.radius_eye, (255, 255, 255), -1)

        # Iris: circular and concentric
        iris_radius = int(self.radius_eye * 0.6)
        cv2.circle(canvas, (cx, cy), iris_radius, (180, 220, 255), -1)

        # Outer outline (black circle)
        cv2.circle(canvas, (cx, cy), self.radius_eye, (0, 0, 0), 3)

        # Pupil (moves horizontally toward target)
        cv2.circle(canvas, (px, py), self.radius_pupil, (0, 0, 0), -1)


def create_eyes(canvas_w, canvas_h):
    eyes = []
    # Three pairs across the canvas, nicely spaced
    pair_centers_x = [
        canvas_w * 1 / 6,
        canvas_w * 3 / 6,
        canvas_w * 5 / 6,
    ]
    y = canvas_h / 2
    eye_spacing = 70  # distance between left and right eye in a pair

    for pair_index, cx in enumerate(pair_centers_x):
        left_center = (cx - eye_spacing / 2, y)
        right_center = (cx + eye_spacing / 2, y)
        eyes.append(Eye(left_center, EYE_RADIUS, PUPIL_RADIUS, pair_index))
        eyes.append(Eye(right_center, EYE_RADIUS, PUPIL_RADIUS, pair_index))

    return eyes


def compute_normalized_target(face_center, frame_w, frame_h):
    x, y = face_center
    nx = (x - frame_w / 2) / (frame_w / 2)
    nx = max(-1.0, min(1.0, nx))

    # Disable vertical influence: focus on left-right only
    ny = 0.0

    return nx, ny


# ---------------- Helpers: Tracking ----------------

class TrackState:
    def __init__(self, tid, center, area, bbox, inside_perimeter, frontal, frame_index, keypoints):
        self.id = tid
        self.center = center
        self.area = area
        self.bbox = bbox
        self.inside_perimeter = inside_perimeter
        self.frontal = frontal
        self.last_seen = frame_index
        self.enter_frame = frame_index if (inside_perimeter and frontal) else None
        self.keypoints = keypoints

    def update(self, center, area, bbox, inside_perimeter, frontal, frame_index, keypoints):
        self.center = center
        self.area = area
        self.bbox = bbox
        self.inside_perimeter = inside_perimeter
        self.frontal = frontal
        self.last_seen = frame_index
        self.keypoints = keypoints
        if inside_perimeter and frontal:
            if self.enter_frame is None:
                self.enter_frame = frame_index
        else:
            self.enter_frame = None


def is_frontal_face(face_landmarks, ratio_tol=0.35):
    lm = face_landmarks.landmark
    try:
        left_eye = lm[263]
        right_eye = lm[33]
        nose = lm[1]
    except IndexError:
        return True

    lx, ly = left_eye.x, left_eye.y
    rx, ry = right_eye.x, right_eye.y
    nx, ny = nose.x, nose.y

    d_left = np.hypot(nx - lx, ny - ly)
    d_right = np.hypot(nx - rx, ny - ry)

    if d_left < 1e-6 or d_right < 1e-6:
        return False

    ratio = d_left / d_right
    return (1.0 - ratio_tol) <= ratio <= (1.0 + ratio_tol)


def detect_faces(face_mesh, frame_bgr):
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    observations = []

    if not results.multi_face_landmarks:
        return observations

    kp_indices = [1, 33, 263, 61, 291]  # nose, eyes, mouth corners (approx)

    for face_landmarks in results.multi_face_landmarks:
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        min_x = int(min(xs) * w)
        max_x = int(max(xs) * w)
        min_y = int(min(ys) * h)
        max_y = int(max(ys) * h)
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w - 1, max_x)
        max_y = min(h - 1, max_y)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        if bbox_w <= 0 or bbox_h <= 0:
            continue

        area = float(bbox_w * bbox_h)
        area_frac = area / float(w * h)
        if area_frac < MIN_FACE_AREA_FRAC:
            continue

        cx = min_x + bbox_w / 2.0
        cy = min_y + bbox_h / 2.0

        inside_perimeter = (
            PERIMETER_X_MIN * w <= cx <= PERIMETER_X_MAX * w and
            PERIMETER_Y_MIN * h <= cy <= PERIMETER_Y_MAX * h
        )

        frontal = is_frontal_face(face_landmarks)

        keypoints = []
        for idx in kp_indices:
            lm = face_landmarks.landmark[idx]
            kx = int(lm.x * w)
            ky = int(lm.y * h)
            keypoints.append((kx, ky))

        obs = {
            "center": (cx, cy),
            "bbox": (min_x, min_y, max_x, max_y),
            "area": area,
            "inside_perimeter": inside_perimeter,
            "frontal": frontal,
            "keypoints": keypoints,
        }
        observations.append(obs)

    return observations


def update_tracks(tracks, observations, frame_index):
    global track_id_counter
    new_tracks = {}
    used_track_ids = set()

    for obs in observations:
        cx, cy = obs["center"]
        best_id = None
        best_dist2 = None

        for tid, tr in tracks.items():
            if tid in used_track_ids:
                continue
            tx, ty = tr.center
            dx = cx - tx
            dy = cy - ty
            dist2 = dx * dx + dy * dy
            if dist2 <= TRACK_MATCH_DIST_PX * TRACK_MATCH_DIST_PX:
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_id = tid

        if best_id is None:
            track_id_counter += 1
            tid = track_id_counter
            tr = TrackState(
                tid,
                obs["center"],
                obs["area"],
                obs["bbox"],
                obs["inside_perimeter"],
                obs["frontal"],
                frame_index,
                obs["keypoints"],
            )
        else:
            tr = tracks[best_id]
            tr.update(
                obs["center"],
                obs["area"],
                obs["bbox"],
                obs["inside_perimeter"],
                obs["frontal"],
                frame_index,
                obs["keypoints"],
            )
            used_track_ids.add(best_id)

        new_tracks[tr.id] = tr

    return new_tracks


def select_primary_target(valid_tracks):
    valid_tracks_sorted = sorted(
        valid_tracks,
        key=lambda t: (t.enter_frame, -t.area),
    )
    return valid_tracks_sorted[0]


# ---------------- Drawing Helpers ----------------

def draw_perimeter_box(frame):
    h, w = frame.shape[:2]
    x1 = int(PERIMETER_X_MIN * w)
    x2 = int(PERIMETER_X_MAX * w)
    y1 = int(PERIMETER_Y_MIN * h)
    y2 = int(PERIMETER_Y_MAX * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)


def draw_tracks_on_frame(frame, tracks, valid_ids, primary_id=None, pair_targets=None):
    for tr in tracks.values():
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
        if tr.id in valid_ids:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tr.id}"
        if tr.id == primary_id:
            label += " *"
        if pair_targets and tr.id in pair_targets:
            label += f" P{pair_targets[tr.id] + 1}"

        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        for kx, ky in tr.keypoints:
            cv2.circle(frame, (kx, ky), 2, color, -1)


# ---------------- Main Loop ----------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open default camera.")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # Keep Eyes window autosized so circles stay circular (no stretching)
    cv2.namedWindow("Eyes", cv2.WINDOW_AUTOSIZE)

    eyes = create_eyes(EYE_CANVAS_WIDTH, EYE_CANVAS_HEIGHT)

    mp_face_mesh = mp.solutions.face_mesh

    tracks = {}
    frame_index = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            observations = detect_faces(face_mesh, frame)
            tracks = update_tracks(tracks, observations, frame_index)

            valid_tracks = [t for t in tracks.values() if t.enter_frame is not None]
            valid_ids = set(t.id for t in valid_tracks)

            pair_targets_norm = [(0.0, 0.0)] * 3
            mode_text = "No target"
            primary_id = None
            pair_targets_debug = {}

            if len(valid_tracks) == 1:
                primary = select_primary_target(valid_tracks)
                primary_id = primary.id
                nx, ny = compute_normalized_target(primary.center, frame_w, frame_h)
                pair_targets_norm = [(nx, ny)] * 3
                mode_text = f"All eyes on ID {primary.id}"

            elif len(valid_tracks) >= 2:
                sorted_tracks = sorted(valid_tracks, key=lambda t: t.enter_frame)
                primary = select_primary_target(valid_tracks)
                primary_id = primary.id

                pair_targets_norm = []
                for pair_index in range(3):
                    tr_for_pair = sorted_tracks[min(pair_index, len(sorted_tracks) - 1)]
                    nx, ny = compute_normalized_target(tr_for_pair.center, frame_w, frame_h)
                    pair_targets_norm.append((nx, ny))
                    pair_targets_debug[tr_for_pair.id] = pair_index

                mode_text = "Multi-target (FCFS)"

            draw_perimeter_box(frame)
            draw_tracks_on_frame(frame, tracks, valid_ids, primary_id, pair_targets_debug)
            cv2.putText(
                frame,
                mode_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Camera", frame)

            eye_canvas = np.zeros((EYE_CANVAS_HEIGHT, EYE_CANVAS_WIDTH, 3), dtype=np.uint8)
            eye_canvas[:] = (30, 30, 30)

            max_offset = EYE_RADIUS * PUPIL_MAX_OFFSET_FACTOR

            for eye in eyes:
                pair_index = eye.pair_index
                norm_target = pair_targets_norm[pair_index]
                eye.update(norm_target, max_offset, smooth=SMOOTHING_FACTOR)
                eye.draw(eye_canvas)

            cv2.imshow("Eyes", eye_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
