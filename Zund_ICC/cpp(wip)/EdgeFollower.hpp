#ifndef EDGE_FOLLOWER_HPP
#define EDGE_FOLLOWER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class EdgeFollower {
public:
    EdgeFollower();
    std::vector<cv::Point> followEdge(const cv::Mat& edgeImage, const cv::Point& startPoint);

private:
    // Parameters for edge following
    int step_size_;
    double close_enough_threshold_;
};

#endif // EDGE_FOLLOWER_HPP
