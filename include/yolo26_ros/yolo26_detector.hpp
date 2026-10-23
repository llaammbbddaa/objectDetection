#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
namespace yolo26 {
struct Detection { float x1,y1,x2,y2; float confidence; int classId; };
struct TileLayout { int columns; int rows; };
struct Resolution { int width; int height; };
Resolution parseResolution(const std::string& value);
TileLayout autoLayout(const cv::Size& size);
TileLayout layoutFromResolution(int width, int height);
class Detector {
public:
 explicit Detector(const std::string& modelPath);
 std::vector<Detection> detect(const cv::Mat&, float, float, TileLayout);
 // Retains the CLI's historical merge pass even for explicit 1x1 tiling.
 std::vector<Detection> detectLegacy(const cv::Mat&, float, float, TileLayout);
private:
 cv::dnn::Net net_;
 std::vector<Detection> run(const cv::Mat&, float, float, TileLayout, bool);
};
cv::Mat annotate(const cv::Mat&, const std::vector<Detection>&);
bool writeCsv(const std::string&, const std::vector<Detection>&, const cv::Size&, double);
}
