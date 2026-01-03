#ifndef SEARCH_STRATEGY_HPP
#define SEARCH_STRATEGY_HPP

#include <opencv2/opencv.hpp>
#include <optional>

class SearchStrategy {
public:
    SearchStrategy();
    std::optional<cv::Point> findFirstEdge(const cv::Mat& edgeImage, const cv::Point& seedPoint, int searchRadius);

private:
    // Using spiral search for robustness
    std::optional<cv::Point> spiralSearch(const cv::Mat& edgeImage, const cv::Point& seedPoint, int searchRadius);
};

#endif // SEARCH_STRATEGY_HPP
