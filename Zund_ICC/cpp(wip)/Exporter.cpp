#include "Exporter.hpp"
#include <fstream>
#include <iostream>

Exporter::Exporter() {}

void Exporter::exportPerimeter(const std::vector<cv::Point>& perimeter, const std::string& filename, const std::string& format) {
    if (format == "csv") {
        exportToCSV(perimeter, filename);
    } else if (format == "json") {
        exportToJSON(perimeter, filename);
    } else {
        std::cerr << "Unsupported export format: " << format << std::endl;
    }
}

void Exporter::exportToCSV(const std::vector<cv::Point>& perimeter, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile << "x,y" << std::endl;
    for (const auto& point : perimeter) {
        outFile << point.x << "," << point.y << std::endl;
    }
    outFile.close();
    std::cout << "Perimeter exported to " << filename << std::endl;
}

void Exporter::exportToJSON(const std::vector<cv::Point>& perimeter, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    outFile << "{" << std::endl;
    outFile << "  \"label\": \"SAFE_ZONE/CUT_BOUNDARY\"," << std::endl;
    outFile << "  \"perimeter\": [" << std::endl;
    for (size_t i = 0; i < perimeter.size(); ++i) {
        outFile << "    {\"x\": " << perimeter[i].x << ", \"y\": " << perimeter[i].y << "}";
        if (i < perimeter.size() - 1) {
            outFile << ",";
        }
        outFile << std::endl;
    }
    outFile << "  ]" << std::endl;
    outFile << "}" << std::endl;
    outFile.close();
    std::cout << "Perimeter exported to " << filename << std::endl;
}
