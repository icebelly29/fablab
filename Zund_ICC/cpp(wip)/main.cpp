#include <iostream>
#include "EdgeDetector.hpp"
#include "SearchStrategy.hpp"
#include "EdgeFollower.hpp"
#include "Exporter.hpp"

// These comments simulate the mapping between the software and the physical machine.
// In a real system, these values would be calibrated precisely.
// Known pixel-to-millimeter scale. For this simulation, we assume 1 pixel = 1 mm.
const double PIXEL_TO_MM = 1.0;
// Fixed camera-to-tool offset. We assume the camera is perfectly centered on the tool.
const cv::Point CAMERA_OFFSET(0, 0);

int main() {
    // Create a dummy image for testing
    cv::Mat testImage = cv::Mat::ones(500, 500, CV_8UC3) * 255;
    cv::rectangle(testImage, cv::Rect(100, 100, 300, 300), cv::Scalar(0, 0, 0), -1);
    cv::imwrite("test_image.png", testImage);


    cv::Mat image = cv::imread("test_image.png");
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    EdgeDetector edgeDetector;
    cv::Mat edgeImage = edgeDetector.detectEdges(image);
    cv::imwrite("edges.png", edgeImage);


    SearchStrategy searchStrategy;
    // Manually seeded point, analogous to a laser pointer
    cv::Point seedPoint(50, 50); 
    std::optional<cv::Point> firstEdgePoint = searchStrategy.findFirstEdge(edgeImage, seedPoint, 100);

    if (!firstEdgePoint) {
        std::cerr << "Error: Could not find first edge point." << std::endl;
        return -1;
    }

    std::cout << "Found first edge point at: " << firstEdgePoint->x << ", " << firstEdgePoint->y << std::endl;

    EdgeFollower edgeFollower;
    std::vector<cv::Point> perimeter = edgeFollower.followEdge(edgeImage, *firstEdgePoint);

    // Visualize the perimeter
    cv::Mat resultImage = image.clone();
    for (const auto& point : perimeter) {
        resultImage.at<cv::Vec3b>(point) = cv::Vec3b(0, 0, 255); // Draw perimeter in red
    }
    cv::circle(resultImage, *firstEdgePoint, 5, cv::Scalar(0, 255, 0), -1); // Draw start point in green
    cv::imwrite("perimeter.png", resultImage);


    Exporter exporter;
    exporter.exportPerimeter(perimeter, "perimeter.csv", "csv");
    exporter.exportPerimeter(perimeter, "perimeter.json", "json");

    return 0;
}
