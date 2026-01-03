#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <opencv2/opencv.hpp>

class EdgeDetector {
public:
    EdgeDetector();
    cv::Mat detectEdges(const cv::Mat& inputImage);

private:
    // Parameters for Canny edge detection
    double canny_threshold1_;
    double canny_threshold2_;
    int aperture_size_;
    bool l2_gradient_;
};

#endif // EDGE_DETECTOR_HPP
