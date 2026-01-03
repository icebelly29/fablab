#include "EdgeFollower.hpp"
#include <iostream>

EdgeFollower::EdgeFollower() : step_size_(5), close_enough_threshold_(10.0) {}

std::vector<cv::Point> EdgeFollower::followEdge(const cv::Mat& edgeImage, const cv::Point& startPoint) {
    std::vector<cv::Point> perimeter;
    cv::Point currentPoint = startPoint;
    perimeter.push_back(currentPoint);

    // This loop simulates the real-time feedback loop of an industrial ICC system.
    // The camera continuously captures images, the system recalculates the edge position,
    // and adjusts the tool path accordingly. This is different from a one-time
    // contour extraction, as it allows for real-time corrections.
    while (true) {
        // Find the next edge point in a small window around the current point
        bool foundNext = false;
        for (int i = -step_size_; i <= step_size_; ++i) {
            for (int j = -step_size_; j <= step_size_; ++j) {
                if (i == 0 && j == 0) continue;

                cv::Point nextPoint = currentPoint + cv::Point(i, j);
                if (nextPoint.x >= 0 && nextPoint.x < edgeImage.cols && nextPoint.y >= 0 && nextPoint.y < edgeImage.rows) {
                    // Check if the point is an edge and not already in the perimeter
                    if (edgeImage.at<uchar>(nextPoint) > 0) {
                        bool alreadyVisited = false;
                        for (const auto& p : perimeter) {
                            if (cv::norm(p - nextPoint) < 2.0) {
                                alreadyVisited = true;
                                break;
                            }
                        }

                        if (!alreadyVisited) {
                            currentPoint = nextPoint;
                            perimeter.push_back(currentPoint);
                            foundNext = true;
                            goto next_step;
                        }
                    }
                }
            }
        }

    next_step:
        if (!foundNext) {
            // No more unvisited edge points found, break the loop
            break;
        }

        // Check for loop closure
        if (perimeter.size() > 10 && cv::norm(currentPoint - startPoint) < close_enough_threshold_) {
            std::cout << "Perimeter closed." << std::endl;
            break;
        }
    }

    return perimeter;
}
