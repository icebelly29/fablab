Here’s a walkthrough of what your **Padayani Eye Tracker Prototype** does and how each part works. 

---

## 1. High-level overview

The script:

1. Opens a webcam (or video).
2. Preprocesses each frame to reduce glare and normalize brightness.
3. Runs **MoveNet MultiPose** (TFLite) to detect up to 6 people and their body keypoints.
4. Tracks detected people over time using a simple **centroid tracker** with stable IDs.
5. Filters people by:

   * Being inside a central “perimeter zone”
   * Being large enough (approx distance)
   * Having enough confident keypoints
   * Roughly facing the camera/exhibit (via face/shoulder heuristic)
6. In **single** or **multi** mode, chooses one or up to three people as gaze targets.
7. Maps those targets to **horizontal offsets** in a separate **eyes canvas**, where three pairs of circular cartoon eyes look left/right according to person positions.
8. Shows two windows:

   * `"Padayani Camera"`: original camera feed with boxes, keypoints, and labels.
   * `"Padayani Eyes"`: the cartoon eyes with smooth left/right motion.

You can press:

* `m` to toggle between `"single"` and `"multi"` focus.
* `q` or `Esc` to quit.

---

## 2. Configuration section

All of these constants at the top define how the system behaves:

* **Model & selection mode**

  * `MOVENET_MODEL_PATH` – path to the MoveNet MultiPose TFLite model.
  * `SELECTION_MODE` – `"single"` or `"multi"`.

* **Model input**

  * `MOVENET_INPUT_SIZE = 256` – frames are resized to 256×256 before going to MoveNet.

* **Keypoint & instance thresholds**

  * `KEYPOINT_CONF_THRESH` – minimum confidence for keypoints.
  * `INSTANCE_SCORE_THRESH` – minimum detection score for a person.
  * `CORE_KEYPOINT_MIN_COUNT` – how many of (nose, shoulders, hips) must be above the threshold for the person to be considered reliable.

* **Perimeter region**

  * `PERIMETER_X_MARGIN`, `PERIMETER_Y_MARGIN` – define a central rectangle as a fraction of width/height. People must stand inside this zone to be “eligible”.

* **Bounding box size filter**

  * `MIN_BBOX_AREA_RATIO` – bounding box must be at least this fraction of the frame area.

* **Brightness / glare control**

  * `BRIGHTNESS_LOW`, `BRIGHTNESS_HIGH`, `BRIGHTNESS_VERY_HIGH` – thresholds on mean brightness that decide how to brighten/darken/fallback.

* **Tracker**

  * `TRACKER_MAX_DISTANCE` – max distance to match detections to existing tracks (in pixels).
  * `TRACKER_MAX_AGE` – how many frames a track can go unseen before being removed.

* **Eyes canvas**

  * `EYES_CANVAS_WIDTH`, `EYES_CANVAS_HEIGHT` – size of the cartoon eyes window.
  * `EYE_RADIUS`, `PUPIL_RADIUS` – sizes of eye and pupil circles.
  * `PUPIL_TRAVEL_PX` – max left/right travel distance of pupils inside the eye.

* **Smoothing**

  * `SMOOTHING_ALPHA` – exponential smoothing factor (small = smoother but more lag).

* **Face detection**

  * `FACE_MIN_SIZE_PX` – minimum face size for Haar cascade detection.

These are essentially **hyperparameters** for detection, tracking, and rendering.

---

## 3. Loading the MoveNet TFLite model

### `load_movenet_interpreter(model_path)`

* Uses `tf.lite.Interpreter` to load the `.tflite` MoveNet model.
* Gets input details, then explicitly **resizes the input tensor** to `[1, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3]`.
* Calls `allocate_tensors()` to finalize shapes.
* Returns the configured interpreter.

**Why resize?**
The MoveNet MultiPose model supports dynamic image size but expects the shape to be set before allocation. This function standardizes to 256×256 for every frame.

---

## 4. Preprocessing: glare & brightness mitigation

### `apply_gamma(image, gamma)`

* Builds a lookup table for gamma correction:

  * Computes `(i/255)^inv_gamma * 255` for each intensity `i`.
  * Uses `cv2.LUT` to apply the table to the image.

### `preprocess_frame_for_detection(frame_bgr)`

Goal: normalize brightness and contrast so the pose detector sees people more reliably.

Steps:

1. Copy the input frame.
2. Convert to grayscale and compute `mean_brightness`.
3. **Gamma correction**:

   * If mean is below `BRIGHTNESS_LOW`: `gamma=1.4` (brighten).
   * If above `BRIGHTNESS_HIGH`: `gamma=0.7` (darken).
4. If brightness is *extremely* high (`> BRIGHTNESS_VERY_HIGH`):

   * Apply **histogram equalization** on grayscale (`cv2.equalizeHist`) and convert back to BGR.
5. Otherwise:

   * Convert to LAB.
   * Run **CLAHE** on the L (lightness) channel to enhance local contrast.
   * Merge channels and convert back to BGR.

Returns a new BGR image used for both detection and face/pose estimation.

---

## 5. Parsing MoveNet MultiPose output

### `KEYPOINT_NAMES`

A list of 17 named keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.

### `parse_movenet_multipose(output_1x6x56, frame_width, frame_height)`

MoveNet MultiPose output shape: `(1, 6, 56)`:

* For each of up to 6 detections:

  * First 51 values = `17 keypoints × (y, x, score)`.
  * Last 5 values = `[ymin, xmin, ymax, xmax, instance_score]`.

The function:

1. Loops over each detection vector `det`.
2. Splits:

   * `keypoints_flat = det[:51]`
   * `bbox_and_score = det[51:]`
3. Extracts `ymin, xmin, ymax, xmax, instance_score`.

   * If `instance_score < INSTANCE_SCORE_THRESH`, skip this person.
4. Converts bbox from normalized [0,1] to pixel coordinates using `frame_width`, `frame_height` (and clamps them to image bounds).
5. Iterates over 17 keypoints:

   * Each keypoint uses `(y_norm, x_norm, score)` → convert to pixel `(x_px, y_px)` and record `[x, y, score]`.
6. Appends a dict for each valid person:

   ```python
   {
       "keypoints": (17, 3) array [x, y, score],
       "bbox": (x_min, y_min, x_max, y_max),
       "instance_score": float
   }
   ```

This is the **standardized representation** used by the rest of the code.

---

## 6. Simple centroid tracker

### Class: `SimpleCentroidTracker`

Purpose: maintain a stable ID per person across frames with a simple greedy nearest-centroid strategy.

#### Internal state

* `self.tracks`: dict `id -> track_info` where each track has:

  * `"id"`, `"bbox"`, `"keypoints"`, `"centroid"`, `"instance_score"`.
  * `"last_seen"` – frame index when last matched.
  * `"matched_in_frame"` – helper flag per update.
  * `"is_eligible"` – whether they pass the eligibility filters.
  * `"eligible_start_frame"` – when they first became continuously eligible.

* `self.next_id`: the next ID to assign.

#### `_compute_centroid(keypoints, bbox)`

* Uses the **center of the bounding box** as the centroid `(cx, cy)`.
* This is more stable for distant people than averaging keypoints.

#### `update(persons, frame_idx)`

1. Builds a list of `detections` – one per person: bbox, keypoints, instance_score, centroid.
2. Marks all existing tracks as `matched_in_frame = False`.
3. For each detection:

   * Searches all existing tracks for the **nearest centroid** (Euclidean distance).
   * If the best distance ≤ `max_distance`:

     * Update that track’s bbox, keypoints, centroid, instance_score, `last_seen`, `matched_in_frame`.
   * Otherwise, create a **new track** with a new ID.
4. After assigning all detections:

   * Any track whose `frame_idx - last_seen > max_age` is removed (stale).
5. Returns the updated `self.tracks`.

This is a **very lightweight tracker**, suitable for small numbers of people and not too much occlusion.

---

## 7. Facing / gaze heuristic

### `is_person_facing_exhibit(track, gray_frame, face_detector)`

Goal: approximate whether a person is **facing towards the camera/exhibit**.

Inputs:

* `track` – has bbox and keypoints.
* `gray_frame` – grayscale preprocessed frame.
* `face_detector` – Haar cascade.

Outputs:

* `(facing_bool, used_face_detector_bool)`.

Process:

1. Extract some keypoints: nose, eyes, shoulders.

2. **Face-based heuristic (preferred)**:

   * Define a **head region** using bbox top to either:

     * Halfway down the box, or
     * The average Y of both shoulders (if shoulders are confident).
   * Crop that region from `gray_frame`.
   * Run `detectMultiScale()` with `minSize=(FACE_MIN_SIZE_PX, FACE_MIN_SIZE_PX)`.
   * If any faces detected:

     * Take the largest face by area.
     * If nose and eyes are confident:

       * Compute mid-eye x coordinate; compare nose x with that.
       * If `|nose_x – mid_eye_x|` < `0.12 × face_width` → consider **facing**.
     * Else (no good facial keypoints):

       * Use face aspect ratio: if roughly square (0.8–1.4), treat as facing.
     * Return (facing, True).

3. **Fallback: body orientation via shoulders + nose**:

   * Check nose, left shoulder, right shoulder are confident.
   * Compute shoulder midpoint `shoulder_mid_x` and width.
   * Compare nose_x with shoulder_mid_x:

     * If the offset is less than `0.35 × shoulder_width`, treat as **facing**.
   * Return (facing, False).

So, if face detection works, it uses nose vs eyes alignment; otherwise, it uses body orientation.

---

## 8. Eligibility & perimeter logic

### `compute_perimeter(frame_width, frame_height)`

* Calculates a central rectangle using margins:

  ```python
  x_min = frame_width * PERIMETER_X_MARGIN
  x_max = frame_width * (1 - PERIMETER_X_MARGIN)
  y_min = frame_height * PERIMETER_Y_MARGIN
  y_max = frame_height * (1 - PERIMETER_Y_MARGIN)
  ```

* This is the “interaction zone” where people must stand.

### `update_eligibility(tracks, frame_idx, frame_width, frame_height, gray_frame, face_detector)`

For each track:

1. **Perimeter check**:

   * Get track centroid `(cx, cy)`.
   * Check if it lies inside the perimeter rectangle.

2. **Size check**:

   * Compute bbox area.
   * Check `bbox_area >= MIN_BBOX_AREA_RATIO × frame_area`.

3. **Keypoint reliability**:

   * Look at core keypoints indices: nose (0), shoulders (5,6), hips (11,12).
   * Count how many have `score >= KEYPOINT_CONF_THRESH`.
   * Require at least `CORE_KEYPOINT_MIN_COUNT`.

4. **Facing check**:

   * Call `is_person_facing_exhibit()`.

5. Combine all conditions to get `is_eligible`:

   * Inside perimeter
   * Large enough
   * Enough good core keypoints
   * Facing

6. Update track:

   * If `is_eligible` just became `True`:

     * Set `eligible_start_frame = frame_idx`.

   * If `is_eligible` is `False`:

     * Reset `eligible_start_frame = None`.

   * Store `track["is_eligible"] = is_eligible`.

Eligibility is used later to decide which people drive the eyes.

---

## 9. Target selection: single vs multi focus

### `select_single_focus(tracks, frame_idx, locked_target_id)`

Single mode logic:

1. Build a list of **eligible tracks**.
2. If there is a currently **locked target** and it’s still eligible:

   * Keep using it.
3. Otherwise, if no eligible tracks:

   * Return `(None, None)`.
4. Else:

   * Choose the track with:

     * Longest continuous eligibility (`frame_idx - eligible_start_frame`).
     * If tie, larger bbox area.
5. Return the chosen track’s ID and the track dict.

This gives one **primary person** for all eye pairs to follow.

### `select_multi_focus(tracks, frame_idx)`

Multi mode logic:

1. Get all eligible tracks.
2. If none:

   * Return `[None, None, None]` and `None`.
3. Sort eligible tracks by `eligible_start_frame` (oldest first).
4. `primary = earliest eligible track`.
5. For each of the 3 eye pairs:

   * If there is a track at that index in the sorted list, assign that track’s ID.
   * Otherwise, use the `primary` track ID.
6. Returns:

   * `pair_target_ids` – 3-element list of IDs.
   * `primary` track.

So:

* Pair 1 → earliest eligible
* Pair 2 → second earliest (if exists, else primary)
* Pair 3 → third earliest (if exists, else primary)

---

## 10. Eyes controller & rendering

### Class: `EyesController`

Controls pupil positions over time.

#### State

* `self.num_pairs` – number of eye pairs (3).
* `self.smoothed_offsets` – list of normalized horizontal offsets per pair, each in `[-1, 1]`.

#### `update_offsets(raw_offsets)`

* `raw_offsets`: list of offsets (one per pair) in `[-1,1]` or `None`.
* For each offset:

  * If `None`, treat as `0.0`.

  * Clamp to `[-1, 1]`.

  * Compute **exponential moving average**:

    ```python
    new_val = alpha * x + (1 - alpha) * prev
    ```

  * Store in `self.smoothed_offsets`.

This gives smooth, non-jerky eye motion.

#### `render()`

1. Create a white canvas (`EYES_CANVAS_HEIGHT × EYES_CANVAS_WIDTH`).
2. Define three rows (`rows_y`) where each pair will be drawn (top, middle, bottom).
3. Set horizontal centers:

   * `pair_center_x` at middle of canvas.
   * `eye_spacing` to separate left & right eyes.
4. For each pair:

   * Compute `pupil_dx = offset_norm * PUPIL_TRAVEL_PX`.
   * Compute centers for left and right eyes.
   * Draw:

     * Eye (white filled circle + black outline).
     * Pupil (solid black circle), offset horizontally by `pupil_dx`.

Returns the final **eyes canvas** image.

---

## 11. Main loop & integration

### `main()`

1. **Load MoveNet**:

   * `interpreter = load_movenet_interpreter(MOVENET_MODEL_PATH)`
   * Store `input_index`, `output_index`.

2. **Load Haar cascade**:

   * Uses `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`.
   * Warns if loading fails.

3. **Open video capture**:

   * `cap = cv2.VideoCapture(1)` (you can change to `0` or a video file path).
   * Set resolution to 1280×720.

4. Initialize supporting objects:

   * `tracker = SimpleCentroidTracker()`
   * `eyes = EyesController(num_pairs=3)`
   * `locked_target_id = None` (single mode lock)
   * FPS counters.

5. **Main loop**:

   ```python
   while True:
       ret, frame = cap.read()
       if not ret: break
       frame_idx += 1
   ```

   Per frame:

   * Mirror image: `frame = cv2.flip(frame, 1)`.

   * Get `h, w`.

   * Preprocess: `proc_frame = preprocess_frame_for_detection(frame)`.

   * Convert to gray and RGB.

   * Resize RGB to 256×256, cast to `uint8`, expand dims for TFLite.

   * **Run MoveNet**:

     ```python
     interpreter.set_tensor(input_index, input_img)
     interpreter.invoke()
     output = interpreter.get_tensor(output_index)  # (1, 6, 56)
     ```

   * **Parse persons**: `persons = parse_movenet_multipose(output, w, h)`.

   * **Update tracker**: `tracks = tracker.update(persons, frame_idx)`.

   * **Update eligibility**: `update_eligibility(tracks, frame_idx, w, h, gray, face_detector)`.

   * **Selection & offsets**:

     * If `SELECTION_MODE == "single"`:

       * `locked_target_id, primary_track = select_single_focus(...)`.

       * Compute `norm_x` from primary track’s centroid:

         ```python
         norm_x = (cx / w - 0.5) * 2.0  # maps [0,w] → [-1,1]
         ```

       * `pair_offsets = [norm_x, norm_x, norm_x]`.

     * Else (`"multi"`):

       * `pair_target_ids, primary_track = select_multi_focus(...)`.
       * For each ID:

         * If track exists, compute `norm_x` from that track’s centroid.
         * Otherwise, use `0.0`.

   * Update eyes:

     ```python
     eyes.update_offsets(pair_offsets)
     eyes_canvas = eyes.render()
     ```

   * **Draw camera overlay**:

     * Draw perimeter rectangle.
     * For each track:

       * Draw bbox (green if eligible, red otherwise).
       * Draw centroid.
       * Draw keypoints (blue circles).
       * Build label: `"ID {tid}"`, plus `[ELIGIBLE]` and `[PRIMARY]` if applicable.
       * Put the text near the bbox.

   * **FPS**:

     * Update counters, every second recompute fps and reset counters.

   * Add text overlay:

     * `"Mode: SINGLE/MULTI  FPS: X.Y"`.
     * `"Press 'm' to toggle mode, 'q' to quit."`.

   * **Show windows**:

     * `cv2.imshow("Padayani Camera", frame)`.
     * `cv2.imshow("Padayani Eyes", eyes_canvas)`.

   * Handle keypress:

     * `q` or `Esc`: break loop.
     * `m`: toggle `SELECTION_MODE` between `"single"` and `"multi"`, and reset `locked_target_id`.

6. At the end:

   * `cap.release()`
   * `cv2.destroyAllWindows()`.

### `if __name__ == "__main__": main()`

Standard Python entry-point.

---

## 12. How everything fits together (data flow summary)

For each frame:

1. **Frame in** → brightness/glare normalization.
2. Normalized frame → **MoveNet MultiPose** → persons (keypoints + bbox + scores).
3. Persons → **centroid tracker** → stable IDs & tracks.
4. Tracks + frame → **eligibility** (perimeter, size, keypoints, facing).
5. Eligible tracks → **target selection** (single or multi).
6. Selected tracks → **normalized x offsets** → smoothed by `EyesController`.
7. Smoothed offsets → **eyes canvas** (three rows of circular eyes with horizontal pupil motion).
8. Camera frame → **annotations** (bboxes, IDs, statuses).
9. Both camera & eyes windows are displayed simultaneously.

