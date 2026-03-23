import cv2
import numpy as np
import sys

def image_to_svg_sketch(image_path, output_svg_path):
    """
    Reads an image, performs edge detection, and saves the result as an SVG file.
    """
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image file '{image_path}'. Please check the path.")
        return

    # 2. Convert to grayscale. Plotters typically work with single-color lines.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Blur the image slightly. This removes fine details like individual hairs,
    # which can make the plotter sketch look messy.
    # You can increase the (5, 5) to (7, 7) or (9, 9) for a simpler sketch.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Perform Canny edge detection. This finds the strong lines in the image.
    # The two numbers (50, 150) are thresholds.
    # - Lowering the first number (e.g., to 30) will detect more faint edges.
    # - Raising the second number (e.g., to 200) will keep only the strongest edges.
    edges = cv2.Canny(blurred, 50, 150)

    # 5. Find contours. This turns the pixel edges into vector lines.
    # RETR_EXTERNAL retrieves only the main outer outlines. Use RETR_LIST for all details.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Create the SVG file
    height, width = edges.shape
    with open(output_svg_path, 'w') as f:
        # SVG header specifying the size of the drawing canvas
        f.write(f'<svg width="{width}px" height="{height}px" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n')
        
        # A group (<g>) to set the pen style: black ink, 1-pixel width, no fill.
        f.write('  <g fill="none" stroke="black" stroke-width="1">\n')

        # Loop through each detected contour and write it as a polyline
        for contour in contours:
            # Skip very short lines that are likely noise
            if len(contour) < 5:
                continue
            
            # Flatten the array of points and format them for the SVG "points" attribute
            points = " ".join([f"{point[0][0]},{point[0][1]}" for point in contour])
            f.write(f'    <polyline points="{points}" />\n')

        # Close the group and the SVG file
        f.write('  </g>\n')
        f.write('</svg>\n')

    print(f"Success! Your plotter-ready SVG has been saved to: {output_svg_path}")

if __name__ == "__main__":
    # This block allows you to run the script from the command line.
    # Usage: python image_to_svg.py input_image.jpg output_sketch.svg
    if len(sys.argv) != 3:
        print("Usage: python image_to_svg.py <input_image_path> <output_svg_path>")
        print("Example: python image_to_svg.py my_cat.jpg cat_sketch.svg")
    else:
        input_image = sys.argv[1]
        output_svg = sys.argv[2]
        image_to_svg_sketch(input_image, output_svg)