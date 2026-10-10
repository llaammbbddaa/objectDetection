// YOLO26n ONNX inference with OpenCV 4.11 (DNN module)
//
// Model export used (end-to-end, NMS-free; this is YOLO26's default):
//   yolo export model=yolo26n.pt format=onnx opset=12 simplify=True
// The classic YOLO-style export is also supported (adds NMS on our side):
//   yolo export model=yolo26n.pt format=onnx opset=12 simplify=True end2end=False
//
// Usage:
//   ./yolo26_detect <input_image> [conf_thresh=0.25] [nms_thresh=0.45]
//   ./yolo26_detect <input_directory> <csv_directory> <annotated_directory>
//                   [conf_thresh=0.25] [nms_thresh=0.45]
//
// Outputs:
//   annotated.jpg   - original image with boxes + labels drawn
//   detections.csv  - one row per detection: x1,y1,x2,y2,confidence,class_id,class_name
//   For directory input, one CSV and one annotated image are written per image.
//   (the same tensor is also printed to the console)

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

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

struct Detection
{
    float x1, y1, x2, y2;
    float confidence;
    int classId;
};

static bool isDirectory(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

static bool makeDirectory(const std::string &path)
{
    if (path.empty() || isDirectory(path))
        return true;

    const std::size_t separator = path.find_last_of('/');
    if (separator != std::string::npos && separator > 0 &&
        !makeDirectory(path.substr(0, separator)))
        return false;

    if (mkdir(path.c_str(), 0755) == 0)
        return true;
    return errno == EEXIST && isDirectory(path);
}

static std::string joinPath(const std::string &directory, const std::string &name)
{
    if (directory.empty() || directory == ".")
        return name;
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

static std::string fileStem(const std::string &path)
{
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot = path.find_last_of('.');
    const std::size_t nameStart = slash == std::string::npos ? 0 : slash + 1;
    if (dot == std::string::npos || dot < nameStart)
        return path.substr(nameStart);
    return path.substr(nameStart, dot - nameStart);
}

static std::string fileExtension(const std::string &path)
{
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return ".jpg";
    return path.substr(dot);
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

static bool processImage(const std::string &imagePath, const std::string &csvPath,
                         const std::string &annotatedPath, cv::dnn::Net &net,
                         float confThresh, float nmsThresh, bool splitIntoTiles)
{
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Could not read image: " << imagePath << "\n";
        return false;
    }

    std::vector<Detection> detections;
    if (splitIntoTiles)
    {
        const int halfWidth = (image.cols + 1) / 2;
        const int halfHeight = (image.rows + 1) / 2;
        const std::vector<cv::Rect> tiles = {
            cv::Rect(0, 0, halfWidth, halfHeight),
            cv::Rect(halfWidth, 0, image.cols - halfWidth, halfHeight),
            cv::Rect(0, halfHeight, halfWidth, image.rows - halfHeight),
            cv::Rect(halfWidth, halfHeight, image.cols - halfWidth,
                     image.rows - halfHeight)};

        for (const cv::Rect &tile : tiles)
        {
            if (tile.width == 0 || tile.height == 0)
                continue;
            for (Detection detection : detectTile(image(tile), net, confThresh, nmsThresh))
            {
                detection.x1 += tile.x;
                detection.x2 += tile.x;
                detection.y1 += tile.y;
                detection.y2 += tile.y;
                detections.push_back(detection);
            }
        }

        // Suppress duplicate boxes from detections near quadrant boundaries.
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

    // ------------------------------------------------- output tensor (N x 6)
    // Columns: x1, y1, x2, y2, confidence, class_id
    cv::Mat tensor(static_cast<int>(detections.size()), 6, CV_32F);
    for (int i = 0; i < tensor.rows; ++i)
    {
        const Detection &d = detections[i];
        float *r = tensor.ptr<float>(i);
        r[0] = d.x1;
        r[1] = d.y1;
        r[2] = d.x2;
        r[3] = d.y2;
        r[4] = d.confidence;
        r[5] = static_cast<float>(d.classId);
    }

    std::ofstream csv(csvPath);
    if (!csv)
    {
        std::cerr << "Could not open CSV output: " << csvPath << "\n";
        return false;
    }
    csv << "x1,y1,x2,y2,confidence,class_id,class_name\n";

    std::cout << imagePath << ": " << detections.size() << " detections\n";

    for (const Detection &d : detections)
    {
        const std::string &name = (d.classId >= 0 && d.classId < static_cast<int>(CLASS_NAMES.size()))
                                      ? CLASS_NAMES[d.classId]
                                      : "unknown";

        csv << std::fixed << std::setprecision(2)
            << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ","
            << std::setprecision(4) << d.confidence << ","
            << d.classId << "," << name << "\n";

    }

    // ------------------------------------------------------------ annotate
    cv::Mat annotated = image.clone();
    for (const Detection &d : detections)
    {
        const cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1));
        const cv::Point p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
        const cv::Scalar color(0, 255, 0);

        cv::rectangle(annotated, p1, p2, color, 2);

        const std::string name = (d.classId >= 0 && d.classId < static_cast<int>(CLASS_NAMES.size()))
                                     ? CLASS_NAMES[d.classId]
                                     : "unknown";
        std::ostringstream label;
        label << name << " " << std::fixed << std::setprecision(2) << d.confidence;

        int baseline = 0;
        const cv::Size textSize =
            cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int labelTop = std::max(p1.y - textSize.height - baseline, 0);

        cv::rectangle(annotated, cv::Point(p1.x, labelTop),
                      cv::Point(p1.x + textSize.width, labelTop + textSize.height + baseline),
                      color, cv::FILLED);
        cv::putText(annotated, label.str(), cv::Point(p1.x, labelTop + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    if (!cv::imwrite(annotatedPath, annotated))
    {
        std::cerr << "Could not write annotated image: " << annotatedPath << "\n";
        return false;
    }
    std::cout << "  CSV: " << csvPath << "\n"
              << "  Image: " << annotatedPath << "\n";
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> [conf_thresh=0.25] [nms_thresh=0.45]\n"
                  << "       " << argv[0]
                  << " <input_directory> <csv_directory> <annotated_directory>"
                  << " [conf_thresh=0.25] [nms_thresh=0.45]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const bool directoryMode = isDirectory(inputPath);
    const std::string csvDirectory = directoryMode
                                         ? (argc > 2 ? argv[2] : "csv_output")
                                         : ".";
    const std::string annotatedDirectory = directoryMode
                                               ? (argc > 3 ? argv[3] : "annotated_output")
                                               : ".";
    const int thresholdIndex = directoryMode ? 4 : 2;

    try
    {
        const float confThresh = argc > thresholdIndex
                                     ? std::stof(argv[thresholdIndex])
                                     : 0.25f;
        const float nmsThresh = argc > thresholdIndex + 1
                                    ? std::stof(argv[thresholdIndex + 1])
                                    : 0.45f;

        if (confThresh < 0.0f || confThresh > 1.0f || nmsThresh < 0.0f ||
            nmsThresh > 1.0f)
        {
            throw std::invalid_argument("thresholds must be between 0 and 1");
        }

        cv::dnn::Net net = cv::dnn::readNetFromONNX("yolo26.onnx");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        if (!directoryMode)
        {
            return processImage(inputPath, "detections.csv", "annotated.jpg", net,
                                confThresh, nmsThresh, false)
                       ? 0
                       : 1;
        }

        if (!makeDirectory(csvDirectory) || !makeDirectory(annotatedDirectory))
        {
            std::cerr << "Could not create output directories.\n";
            return 1;
        }

        std::vector<cv::String> paths;
        cv::glob(joinPath(inputPath, "*"), paths, false);
        std::sort(paths.begin(), paths.end());

        int processed = 0;
        int failed = 0;
        for (const cv::String &path : paths)
        {
            if (isDirectory(path))
                continue;

            const std::string stem = fileStem(path);
            const std::string csvPath = joinPath(csvDirectory, stem + ".csv");
            const std::string annotatedPath =
                joinPath(annotatedDirectory, stem + "_annotated" + fileExtension(path));
            if (processImage(path, csvPath, annotatedPath, net, confThresh, nmsThresh, true))
                ++processed;
            else
                ++failed;
        }

        if (processed == 0)
        {
            std::cerr << "No readable images found in: " << inputPath << "\n";
            return 1;
        }
        std::cout << "Processed " << processed << " image(s)";
        if (failed != 0)
            std::cout << "; " << failed << " failed";
        std::cout << ".\n";
        return failed == 0 ? 0 : 1;
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid argument: " << e.what() << "\n";
        return 1;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 1;
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Failed to load or run yolo26.onnx: " << e.what() << "\n";
        return 1;
    }
}