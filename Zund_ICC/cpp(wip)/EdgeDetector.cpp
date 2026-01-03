#include "EdgeDetector.hpp"

EdgeDetector::EdgeDetector() : 
    canny_threshold1_(100), 
    canny_threshold2_(200), 
    aperture_size_(3), 
    l2_gradient_(false) 
{
    // Canny edge detection is suitable for this use case because it is a step-edge detector.
    // A step-edge is a sharp discontinuity in brightness, which is what we expect to see
    // between the material and the contrasting background. Canny is robust to noise
    // and provides a single-pixel-wide edge, which is ideal for tracing.
}

cv::Mat EdgeDetector::detectEdges(const cv::Mat& inputImage) {
    cv::Mat grayImage, blurredImage, edgeImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);
    cv::Canny(blurredImage, edgeImage, canny_threshold1_, canny_threshold2_, aperture_size_, l2_gradient_);
    return edgeImage;
}
