#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "yolo26_ros/yolo26_detector.hpp"
namespace yolo26 {
static const int INPUT_SIZE = 640;

static const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

// Resize with aspect ratio preserved and pad to a square (what Ultralytics does).
// Returns the 640x640 image and the scale/padding needed to map boxes back.
static cv::Mat letterbox(const cv::Mat &src, int size, float &scale, int &padX, int &padY)
{
    scale = std::min(static_cast<float>(size) / src.cols,
                     static_cast<float>(size) / src.rows);

    const int newW = static_cast<int>(std::round(src.cols * scale));
    const int newH = static_cast<int>(std::round(src.rows * scale));
    padX = (size - newW) / 2;
    padY = (size - newH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));
    return out;
}

Resolution parseResolution(const std::string &value)
{
    const std::size_t separator = value.find('x');
    if (separator == std::string::npos || separator == 0 || separator == value.size() - 1)
        throw std::invalid_argument("resolution must have the form WIDTHxHEIGHT");

    std::size_t widthCharacters = 0;
    std::size_t heightCharacters = 0;
    int width;
    int height;
    try
    {
        width = std::stoi(value.substr(0, separator), &widthCharacters);
        height = std::stoi(value.substr(separator + 1), &heightCharacters);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("resolution must have the form WIDTHxHEIGHT");
    }

    if (widthCharacters != separator || heightCharacters != value.size() - separator - 1 ||
        width <= 0 || height <= 0)
    {
        throw std::invalid_argument("resolution must have the form WIDTHxHEIGHT");
    }
    return {width, height};
}

static std::vector<cv::Rect> makeTiles(const cv::Size &imageSize, int columns, int rows)
{
    std::vector<cv::Rect> tiles;
    tiles.reserve(static_cast<std::size_t>(columns) * rows);
    for (int row = 0; row < rows; ++row)
    {
        const int y0 = static_cast<int>((static_cast<long long>(row) * imageSize.height) / rows);
        const int y1 =
            static_cast<int>((static_cast<long long>(row + 1) * imageSize.height) / rows);
        for (int column = 0; column < columns; ++column)
        {
            const int x0 =
                static_cast<int>((static_cast<long long>(column) * imageSize.width) / columns);
            const int x1 = static_cast<int>(
                (static_cast<long long>(column + 1) * imageSize.width) / columns);
            if (x1 > x0 && y1 > y0)
                tiles.emplace_back(x0, y0, x1 - x0, y1 - y0);
        }
    }
    return tiles;
}

static std::vector<Detection> detectTile(const cv::Mat &image, cv::dnn::Net &net,
                                         float confThresh, float nmsThresh)
{
    float scale;
    int padX, padY;
    cv::Mat letterboxed = letterbox(image, INPUT_SIZE, scale, padX, padY);

    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                           cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    if (outputs.empty())
        throw std::runtime_error("Network produced no output.");

    const cv::Mat &raw = outputs[0];
    if (raw.dims != 3 || raw.type() != CV_32F)
        throw std::runtime_error("Unexpected output tensor layout.");

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // Maps a box from letterboxed-input pixels back to the original image,
    // clamps it to the image and records it as a candidate.
    auto addCandidate = [&](float x1, float y1, float x2, float y2, float score, int classId)
    {
        const float maxX = static_cast<float>(image.cols - 1);
        const float maxY = static_cast<float>(image.rows - 1);
        x1 = std::max(0.f, std::min((x1 - padX) / scale, maxX));
        y1 = std::max(0.f, std::min((y1 - padY) / scale, maxY));
        x2 = std::max(0.f, std::min((x2 - padX) / scale, maxX));
        y2 = std::max(0.f, std::min((y2 - padY) / scale, maxY));
        boxes.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        confidences.push_back(score);
        classIds.push_back(classId);
    };

    // YOLO26 exports two possible layouts:
    //   end-to-end (default): [1, 300, 6], each row = x1, y1, x2, y2, score, class_id
    //                         (already de-duplicated, so no NMS is needed)
    //   end2end=False:        [1, 4 + classes, 8400], columns = cx, cy, w, h, class scores
    //                         (needs NMS, same as YOLO11)
    const bool endToEnd = raw.size[2] == 6;

    if (endToEnd)
    {
        const int numDetections = raw.size[1];
        const float *data = raw.ptr<float>();
        for (int i = 0; i < numDetections; ++i)
        {
            const float *row = data + static_cast<std::size_t>(i) * 6;
            if (row[4] < confThresh)
                continue;
            addCandidate(row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5]));
        }
    }
    else
    {
        const int numChannels = raw.size[1];
        const int numAnchors = raw.size[2];
        const int numClasses = numChannels - 4;
        cv::Mat flat(numChannels, numAnchors, CV_32F, const_cast<float *>(raw.ptr<float>()));
        cv::Mat preds = flat.t();

        for (int i = 0; i < preds.rows; ++i)
        {
            const float *row = preds.ptr<float>(i);
            cv::Mat scores(1, numClasses, CV_32F, const_cast<float *>(row + 4));
            double maxScore;
            cv::Point classPoint;
            cv::minMaxLoc(scores, nullptr, &maxScore, nullptr, &classPoint);
            if (maxScore < confThresh)
                continue;

            const float cx = row[0], cy = row[1], w = row[2], h = row[3];
            addCandidate(cx - w / 2.f, cy - h / 2.f, cx + w / 2.f, cy + h / 2.f,
                         static_cast<float>(maxScore), classPoint.x);
        }
    }

    std::vector<int> keep;
    if (endToEnd)
    {
        for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
            keep.push_back(i);
    }
    else
    {
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, keep);
    }

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int idx : keep)
    {
        const cv::Rect &b = boxes[idx];
        detections.push_back({static_cast<float>(b.x), static_cast<float>(b.y),
                              static_cast<float>(b.x + b.width),
                              static_cast<float>(b.y + b.height),
                              confidences[idx], classIds[idx]});
    }
    return detections;
}


TileLayout layoutFromResolution(int w,int h) {
 if(w<=0 || h<=0) throw std::invalid_argument("resolution must be positive");
 return {1+(w-1)/640,1+(h-1)/640};
}
TileLayout autoLayout(const cv::Size& s) { return layoutFromResolution(s.width,s.height); }
Detector::Detector(const std::string& path) : net_(cv::dnn::readNetFromONNX(path)) {
 net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
 net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
 detectTile(cv::Mat(640,640,CV_8UC3,cv::Scalar(0,0,0)),net_,0.25f,0.45f);
}
std::vector<Detection> Detector::detect(const cv::Mat& image,float c,float n,TileLayout l) { return run(image,c,n,l,false); }
std::vector<Detection> Detector::detectLegacy(const cv::Mat& image,float c,float n,TileLayout l) { return run(image,c,n,l,true); }
std::vector<Detection> Detector::run(const cv::Mat& image,float confThresh,float nmsThresh,TileLayout l,bool legacy) {
 if(image.empty()) throw std::invalid_argument("empty image");
 auto& net=net_;
 int tileColumns=l.columns, tileRows=l.rows;
    std::vector<Detection> detections;
    if (tileColumns > 0 && tileRows > 0 && (legacy || tileColumns > 1 || tileRows > 1))
    {
        const std::vector<cv::Rect> tiles =
            makeTiles(image.size(), tileColumns, tileRows);

        for (const cv::Rect &tile : tiles)
        {
            for (Detection detection : detectTile(image(tile), net, confThresh, nmsThresh))
            {
                detection.x1 += tile.x;
                detection.x2 += tile.x;
                detection.y1 += tile.y;
                detection.y2 += tile.y;
                detections.push_back(detection);
            }
        }

        // Suppress duplicate boxes from detections near tile boundaries.
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        for (const Detection &detection : detections)
        {
            boxes.emplace_back(
                cv::Point(static_cast<int>(detection.x1), static_cast<int>(detection.y1)),
                cv::Point(static_cast<int>(detection.x2), static_cast<int>(detection.y2)));
            confidences.push_back(detection.confidence);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, keep);
        std::vector<Detection> filtered;
        filtered.reserve(keep.size());
        for (int index : keep)
            filtered.push_back(detections[index]);
        detections.swap(filtered);
    }
    else
    {
        detections = detectTile(image, net, confThresh, nmsThresh);
    }

    return detections;
}
bool writeCsv(const std::string& csvPath,const std::vector<Detection>& detections,const cv::Size& imageSize,double detectionTimeMs) {
    std::ofstream csv(csvPath);
    if (!csv)
    {
        std::cerr << "Could not open CSV output: " << csvPath << "\n";
        return false;
    }
    csv << "x1,y1,x2,y2,confidence,class_id,class_name,resolutionX,resolutionY,detection_time\n";



    for (const Detection &d : detections)
    {
        const std::string &name = (d.classId >= 0 && d.classId < static_cast<int>(CLASS_NAMES.size()))
                                      ? CLASS_NAMES[d.classId]
                                      : "unknown";

        csv << std::fixed << std::setprecision(2)
            << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ","
            << std::setprecision(4) << d.confidence << ","
            << d.classId << "," << name << ","
            << imageSize.width << "," << imageSize.height << ","
            << std::setprecision(3) << detectionTimeMs << "\n";
    }

    csv.close();
    return bool(csv);
}
cv::Mat annotate(const cv::Mat& image,const std::vector<Detection>& detections) {
    cv::Mat annotated = image.clone();
    for (const Detection &d : detections)
    {
        const cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1));
        const cv::Point p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
        const cv::Scalar color(0, 255, 0);

        cv::rectangle(annotated, p1, p2, color, 1); // adjusted line width so you can see object better

        const std::string name = (d.classId >= 0 && d.classId < static_cast<int>(CLASS_NAMES.size()))
                                     ? CLASS_NAMES[d.classId]
                                     : "unknown";
        std::ostringstream label;
        label << name << " " << std::fixed << std::setprecision(2) << d.confidence;

        int baseline = 0;
        const cv::Size textSize =
            cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        const int labelTop = std::max(p1.y - textSize.height - baseline, 0);

        cv::rectangle(annotated, cv::Point(p1.x, labelTop),
                      cv::Point(p1.x + textSize.width - 1,
                                labelTop + textSize.height + baseline - 1),
                      color, cv::FILLED);
        cv::putText(annotated, label.str(), cv::Point(p1.x, labelTop + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    return annotated;
}
}
