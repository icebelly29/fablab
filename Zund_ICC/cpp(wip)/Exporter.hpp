#ifndef EXPORTER_HPP
#define EXPORTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Exporter {
public:
    Exporter();
    void exportPerimeter(const std::vector<cv::Point>& perimeter, const std::string& filename, const std::string& format);

private:
    void exportToCSV(const std::vector<cv::Point>& perimeter, const std::string& filename);
    void exportToJSON(const std::vector<cv::Point>& perimeter, const std::string& filename);
};

#endif // EXPORTER_HPP
