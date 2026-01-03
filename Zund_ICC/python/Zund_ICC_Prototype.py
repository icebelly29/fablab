import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import os

# --- Create output directory ---
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# --- Cell 1: Setup and Image Generation ---

def create_synthetic_image(width, height, material_shape):
    """Creates a synthetic image with a contrasting background and a material shape."""
    # Background (lighter gray, simulating the vacuum bed)
    image = np.full((height, width, 3), (180, 180, 180), dtype=np.uint8)

    # Material (blue irregular hexagon)
    pts = np.array(material_shape, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (255, 0, 0)) # Blue in BGR

    return image

# --- Parameters ---
IMG_WIDTH = 800
IMG_HEIGHT = 600
PIXEL_TO_MM = 0.5 # Example scale: 1 pixel = 0.5 mm

# Define the material's position and size as an irregular hexagon
material = [[250, 300], [325, 430], [475, 440], [550, 300], [475, 170], [325, 160]]

# Create the image
original_image = create_synthetic_image(IMG_WIDTH, IMG_HEIGHT, material)

plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Synthetic Input Image (Material on Bed)')
plt.xticks([]), plt.yticks([])
plt.savefig(os.path.join(output_dir, '01_synthetic_input.png'))
print(f"Saved plot to {os.path.join(output_dir, '01_synthetic_input.png')}")
plt.close()

# --- Cell 2: Edge Detection ---

def detect_edges(image):
    """Performs the edge detection pipeline."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges, blurred, gray

edges_detected, blurred_image, gray_image = detect_edges(original_image)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('1. Grayscale Image')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title('2. Blurred Image')
axes[1].set_xticks([]), axes[1].set_yticks([])
axes[2].imshow(edges_detected, cmap='gray')
axes[2].set_title('3. Canny Edges')
axes[2].set_xticks([]), axes[2].set_yticks([])
plt.savefig(os.path.join(output_dir, '02_edge_detection_steps.png'))
print(f"Saved plot to {os.path.join(output_dir, '02_edge_detection_steps.png')}")
plt.close()

# --- Cell 3: Manual Seeding and Autonomous Edge Search ---

def spiral_search(edges, start_point, vis_image, frame_counter, max_radius=100):
    """Performs a spiral search for the first edge pixel and saves frames."""
    x, y = start_point
    path = [(x, y)]
    
    # Save initial frame
    frame_filename = os.path.join(frames_dir, f"frame_{frame_counter:04d}.png")
    cv2.imwrite(frame_filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    frame_counter += 1

    if edges[y, x] != 0:
        return (x, y), path, frame_counter
        
    dx, dy = 1, 0
    steps = 1
    turns = 0
    for _ in range(max_radius * 2):
        for _ in range(steps):
            x, y = x + dx, y + dy
            path.append((x, y))
            
            # Update visualization
            if 0 <= y < vis_image.shape[0] and 0 <= x < vis_image.shape[1]:
                vis_image[y, x] = (0, 0, 255) # Blue trail
                frame_filename = os.path.join(frames_dir, f"frame_{frame_counter:04d}.png")
                cv2.imwrite(frame_filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                frame_counter += 1

            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                if edges[y, x] != 0:
                    return (x, y), path, frame_counter
        dx, dy = -dy, dx
        turns += 1
        if turns % 2 == 0:
            steps += 1
    return None, path, frame_counter

# --- Cell 3: Manual Seeding and Autonomous Edge Search ---

frame_counter = 0
vis_image_search = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

seed_point = (250, 300) # Adjusted for the circle
search_window_size = 70

# Draw search window for context
sw_x, sw_y = seed_point[0] - search_window_size, seed_point[1] - search_window_size
sw_w, sw_h = search_window_size * 2, search_window_size * 2
cv2.rectangle(vis_image_search, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), (255, 0, 0), 2)
cv2.circle(vis_image_search, seed_point, 5, (255, 165, 0), -1)

found_edge_point, search_path, frame_counter = spiral_search(
    edges_detected, 
    seed_point, 
    vis_image_search,
    frame_counter,
    max_radius=search_window_size
)

# Final visualization of the search
vis_image_final_search = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
cv2.rectangle(vis_image_final_search, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), (255, 0, 0), 2)
cv2.circle(vis_image_final_search, seed_point, 5, (255, 165, 0), -1)
for point in search_path:
    vis_image_final_search[point[1], point[0]] = (0, 0, 255)
if found_edge_point:
    cv2.circle(vis_image_final_search, found_edge_point, 7, (0, 255, 0), 2)
    print(f"Edge found at: {found_edge_point}")
else:
    print("Edge not found within the search radius.")

plt.figure(figsize=(12, 9))
plt.imshow(vis_image_final_search)
plt.title('Autonomous Edge Search')
plt.savefig(os.path.join(output_dir, '03_autonomous_search.png'))
print(f"Saved plot to {os.path.join(output_dir, '03_autonomous_search.png')}")
plt.close()

# Save the final search state as a frame
frame_filename = os.path.join(frames_dir, f"frame_{frame_counter:04d}.png")
cv2.imwrite(frame_filename, cv2.cvtColor(vis_image_final_search, cv2.COLOR_RGB2BGR))
frame_counter += 1


# --- Cell 4: Edge Following / Tracing ---

def trace_perimeter(edges, start_point, vis_image, frame_counter, original_image_rgb, max_steps=10000):
    """Traces the perimeter of a shape and saves frames of the process."""
    if start_point is None:
        return [], frame_counter
        
    perimeter_path = []
    edge_map = edges.copy()
    current_point = start_point
    
    # Initial frame with the starting point
    path_array = np.array(perimeter_path + [current_point]).reshape((-1, 1, 2))
    cv2.polylines(vis_image, [path_array], isClosed=False, color=(0, 255, 0), thickness=2)
    cv2.circle(vis_image, start_point, 7, (0, 0, 255), -1) # Start point in blue
    frame_filename = os.path.join(frames_dir, f"frame_{frame_counter:04d}.png")
    cv2.imwrite(frame_filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    frame_counter += 1

    for step in range(max_steps):
        x, y = current_point
        perimeter_path.append(current_point)
        edge_map[y, x] = 0 
        
        # Find the next neighbor
        neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
        found_next = False
        for next_x, next_y in neighbors:
            if 0 <= next_y < edge_map.shape[0] and 0 <= next_x < edge_map.shape[1]:
                if edge_map[next_y, next_x] != 0:
                    current_point = (next_x, next_y)
                    found_next = True
                    break
        
        # Visualize current state
        vis_image = original_image_rgb.copy()
        path_array = np.array(perimeter_path).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [path_array], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.circle(vis_image, start_point, 7, (0, 0, 255), -1)
        cv2.circle(vis_image, (x, y), 5, (255, 0, 0), -1) # Current point in red
        
        frame_filename = os.path.join(frames_dir, f"frame_{frame_counter:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        frame_counter += 1

        if not found_next:
            break
            
    return perimeter_path, frame_counter

# --- Cell 4: Edge Following / Tracing ---

vis_trace_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_rgb_for_trace = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

traced_perimeter, frame_counter = trace_perimeter(
    edges_detected, 
    found_edge_point, 
    vis_trace_image, 
    frame_counter,
    original_image_rgb_for_trace
)

if traced_perimeter:
    print(f"Traced {len(traced_perimeter)} points to form the perimeter.")
    
    # Final plot of the tracing result
    vis_trace_image_final = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    path_array = np.array(traced_perimeter).reshape((-1, 1, 2))
    cv2.polylines(vis_trace_image_final, [path_array], isClosed=False, color=(0, 255, 0), thickness=2)
    cv2.circle(vis_trace_image_final, traced_perimeter[0], 7, (0, 0, 255), -1)
    cv2.circle(vis_trace_image_final, traced_perimeter[-1], 7, (255, 0, 0), -1)
    
    plt.figure(figsize=(12, 9))
    plt.imshow(vis_trace_image_final)
    plt.title('Edge Tracing Result')
    plt.savefig(os.path.join(output_dir, '04_edge_tracing.png'))
    print(f"Saved plot to {os.path.join(output_dir, '04_edge_tracing.png')}")
    plt.close()
else:
    print("Could not trace perimeter.")

# --- Cell 5: Output and Finalization ---

final_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
if traced_perimeter:
    path_array = np.array(traced_perimeter).reshape((-1, 1, 2))
    cv2.polylines(final_image, [path_array], isClosed=True, color=(255, 140, 0), thickness=3)
    
    csv_filename = 'cut_boundary.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x_pixel', 'y_pixel', 'x_mm', 'y_mm'])
        for x, y in traced_perimeter:
            writer.writerow([x, y, x * PIXEL_TO_MM, y * PIXEL_TO_MM])
    print(f"Successfully exported {len(traced_perimeter)} points to {csv_filename}")
    
    json_filename = 'cut_boundary.json'
    json_output = {
        'boundary_type': 'Cut Boundary',
        'unit': 'mm',
        'pixel_to_mm_scale': PIXEL_TO_MM,
        'point_count': len(traced_perimeter),
        'perimeter_px': traced_perimeter,
        'perimeter_mm': [[p[0] * PIXEL_TO_MM, p[1] * PIXEL_TO_MM] for p in traced_perimeter]
    }
    with open(json_filename, 'w') as jsonfile:
        json.dump(json_output, jsonfile, indent=4)
    print(f"Successfully exported data to {json_filename}")
    
    plt.figure(figsize=(12, 9))
    plt.imshow(final_image)
    plt.title('Final Digital Boundary Map')
    plt.savefig(os.path.join(output_dir, '05_final_boundary_map.png'))
    print(f"Saved plot to {os.path.join(output_dir, '05_final_boundary_map.png')}")
    plt.close()
else:
    print("No perimeter was traced, cannot generate output.")

print("\nScript finished. Check the output_images/ directory for plots and the root directory for data files.")
