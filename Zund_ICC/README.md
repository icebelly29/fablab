# Zund ICC Prototype

This project is a Python-based prototype demonstrating a computer vision pipeline for autonomously detecting the boundary of a material placed on a Zund cutting machine bed. It simulates the process of finding an object, tracing its perimeter, and generating a digital cut path.

## How it Works

The process is orchestrated by the `Zund_ICC_Prototype.py` script and is broken down into several key stages:

### 1. Synthetic Image Generation
The script first creates a synthetic image to simulate a piece of material (currently an irregular hexagon) on a contrasting background, mimicking the Zund's vacuum bed.

### 2. Edge Detection
Using OpenCV, the script performs a series of image processing steps to clearly identify the material's edges:
- The image is converted to grayscale.
- A Gaussian blur is applied to reduce noise.
- The Canny edge detection algorithm is used to extract the sharp edges.

### 3. Autonomous Edge Search
To begin tracing the boundary, the system needs a starting point. This is achieved by:
- Defining a manual **seed point** in a location where the material is expected to be.
- Performing an autonomous **spiral search** outwards from this seed point.
- The search stops as soon as it intersects with an edge pixel detected in the previous stage. This becomes the starting point for tracing.

### 4. Perimeter Tracing
Once the initial edge point is found, the script follows the connected edge pixels around the entire shape, recording the coordinates of each point along the perimeter.

### 5. Output Generation
Upon successfully tracing the entire boundary, the script generates the following outputs:

- **`cut_boundary.csv`**: A CSV file containing the pixel and millimeter coordinates of every point on the traced path.
- **`cut_boundary.json`**: A JSON file containing the same boundary points along with metadata like scale and units.
- **`output_images/`**: A directory containing a series of PNG images that visualize each major step of the process, from edge detection to the final boundary map.
- **`frames/`**: A directory containing individual frames captured during the search and trace operations, which can be compiled into a video to animate the process.

## How to Run

Simply execute the main Python script from your terminal:
```bash
python Zund_ICC_Prototype.py
```
The script will run the entire pipeline and generate the output files in the project directory.

## Customization

You can modify the script to change the material shape or the starting conditions:

- **Change the Shape**: In `Zund_ICC_Prototype.py`, locate the `material` variable. You can change the list of coordinates to define any polygon.
- **Adjust the Seed Point**: If you change the shape or its location, you may need to update the `seed_point` variable to ensure the spiral search can find the new edge. The seed point should be a location inside the new shape, close to an edge.

---

## Detailed Breakdown

This section provides two perspectives on the computer vision pipeline: a high-level overview for those new to the field, and a technical analysis for experienced practitioners.

### A. Conceptual Overview (For the Curious)

Imagine you're teaching a robot to see a piece of fabric on a table and trace its outline so it can be cut precisely.

1.  **Practice Sheet**: First, you can't use the real, expensive fabric for practice. So, you draw a shape on a piece of paper (`Synthetic Image Generation`). This gives the robot a perfect, predictable image to learn from.

2.  **Finding the Outline**: The robot's camera sees a picture, but it doesn't understand where the shape begins and the table ends. To help it, you apply a filter that makes the edges pop, like tracing the drawing with a thick, dark marker (`Edge Detection`). The robot now sees a clear, bright line defining the shape's boundary.

3.  **Finding a Starting Point**: The robot needs to know where to begin tracing. You tell it to put its "finger" down in the middle of the paper and spiral outwards until it feels the marker line (`Autonomous Edge Search`). The moment it touches the line, it knows it has found the edge.

4.  **Tracing the Shape**: Keeping its "finger" on the marker line, the robot now carefully follows it all the way around the shape, memorizing the path it takes (`Perimeter Tracing`).

5.  **Creating the Blueprint**: Finally, the robot takes the path it memorized and converts it into a set of coordinates—a digital blueprint (`Output Generation`). This blueprint can be sent to a cutting machine, which will follow the exact same path to cut the real fabric.

### B. Technical Deep Dive (For the Expert)

This section evaluates the algorithms, parameters, and assumptions in the current pipeline.

1.  **Dataset & Environment**:
    -   The use of `np.zeros` and `cv2.fillPoly` creates a synthetic, noise-free dataset with ideal contrast. This is excellent for validating geometric algorithms but does not account for real-world challenges like material texture, shadows, lighting variations, or a non-uniform background.
    -   **Future Work**: Robustness could be tested by introducing synthetic noise (Gaussian, salt-and-pepper) or by using a dataset of real-world images of materials on a Zund bed.

2.  **Edge Detection**:
    -   The pipeline uses `cv2.Canny`, a multi-stage algorithm involving Gaussian smoothing, Sobel gradient calculation, non-maximum suppression, and hysteresis thresholding.
    -   **Parameters**: The current implementation uses hardcoded Canny thresholds. These values are highly sensitive and may not be optimal for different materials or lighting conditions.
    -   **Alternatives**: For more complex scenarios, other detectors could be considered. The Laplacian of Gaussian (LoG) can be effective, though more sensitive to noise. For textured materials, statistical methods or even machine learning-based segmentation models (e.g., U-Net) could provide a more robust boundary definition than classical edge detection.

3.  **Search Strategy**:
    -   The **spiral search** is a deterministic method to find the first edge point from a given `seed_point`. It is simple and guarantees finding a connected edge if one exists within its path.
    -   **Limitations**: Its efficiency is dependent on the seed point's proximity to an edge. Furthermore, it is computationally inefficient for large search areas or complex, non-convex shapes where the seed point might be far from any edge.
    -   **Alternatives**: A more robust method would be to automate seed point selection. Calculating the centroid of the largest contour after a preliminary thresholding pass could provide a reliable starting point. For materials with holes, a raster scan or random sampling approach might be more effective at identifying all boundaries.

4.  **Perimeter Tracing**:
    -   The `EdgeFollower` class implements a local neighborhood search, likely an 8-connectivity Moore-Neighbor tracing algorithm. It moves from one "on" pixel to the next by checking its immediate neighbors.
    -   **Assumptions**: This method assumes the edge detected by Canny is a single-pixel-wide, continuous, and non-branching path. Gaps in the edge (due to poor detection) will prematurely terminate the trace.
    -   **Improvements**: A more resilient algorithm could incorporate a "jump" or "search" behavior to bridge small gaps. For production systems, using `cv2.findContours` with `cv2.CHAIN_APPROX_NONE` would be a more standard, highly optimized, and robust approach to extract all boundary points at once.

5.  **Coordinate Transformation & Calibration**:
    -   The script uses a single `px_per_mm` scalar for converting pixel coordinates to physical dimensions.
    -   **Real-World Considerations**: This assumes a perfect orthographic projection, which is not realistic. A production system would require camera calibration to find intrinsic (focal length, optical center) and extrinsic (camera position, rotation) parameters. These parameters are essential for undistorting the image and performing an accurate perspective transformation, ensuring metric accuracy across the entire cutting bed.
