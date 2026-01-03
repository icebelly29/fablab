#include "SearchStrategy.hpp"
#include <iostream>

SearchStrategy::SearchStrategy() {}

std::optional<cv::Point> SearchStrategy::findFirstEdge(const cv::Mat& edgeImage, const cv::Point& seedPoint, int searchRadius) {
    // This seeding process is analogous to the laser pointer on an industrial cutter.
    // The operator directs the laser to a point on the material, and the machine
    // begins its search from there. This manual intervention provides a reliable
    // starting point for the autonomous tracing process.
    return spiralSearch(edgeImage, seedPoint, searchRadius);
}

// Spiral search is robust to placement errors because it searches in an ever-expanding
// pattern. This means that even if the seed point is not exactly on the edge, the
// search will eventually find the edge as long as it is within the search radius.
std::optional<cv::Point> SearchStrategy::spiralSearch(const cv::Mat& edgeImage, const cv::Point& seedPoint, int searchRadius) {
    int x = 0, y = 0;
    int dx = 0, dy = -1;
    for (int i = 0; i < searchRadius * searchRadius; ++i) {
        if ((-searchRadius / 2 < x) && (x <= searchRadius / 2) && (-searchRadius / 2 < y) && (y <= searchRadius / 2)) {
            cv::Point currentPoint = seedPoint + cv::Point(x, y);
            if (currentPoint.x >= 0 && currentPoint.x < edgeImage.cols && currentPoint.y >= 0 && currentPoint.y < edgeImage.rows) {
                if (edgeImage.at<uchar>(currentPoint) > 0) {
                    return currentPoint;
                }
            }
        }
        if (x == y || (x < 0 && x == -y) || (x > 0 && x == 1 - y)) {
            std::swap(dx, dy);
            dx = -dx;
        }
        x += dx;
        y += dy;
    }
    return std::nullopt;
}
