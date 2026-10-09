// YOLO11n ONNX inference with OpenCV 4.5.4 (DNN module)
//
// Model export used:
//   yolo export model=yolo11n.pt format=onnx opset=12 simplify=True
//
// Usage:
//   ./yolo11_detect <input_image> [conf_thresh=0.25] [nms_thresh=0.45]
//
// Outputs:
//   annotated.jpg   - original image with boxes + labels drawn
//   detections.csv  - one row per detection: x1,y1,x2,y2,confidence,class_id,class_name
//   (the same tensor is also printed to the console)

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> [conf_thresh=0.25] [nms_thresh=0.45]\n";
        return 1;
    }

    const std::string imagePath = argv[1];
    const float confThresh = (argc > 2) ? std::stof(argv[2]) : 0.25f;
    const float nmsThresh = (argc > 3) ? std::stof(argv[3]) : 0.45f;

    // ---------------------------------------------------------------- load
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Could not read image: " << imagePath << "\n";
        return 1;
    }

    cv::dnn::Net net;
    try
    {
        net = cv::dnn::readNetFromONNX("yolo11.onnx");
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Failed to load yolo11.onnx: " << e.what() << "\n";
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // ---------------------------------------------------------- preprocess
    float scale;
    int padX, padY;
    cv::Mat letterboxed = letterbox(image, INPUT_SIZE, scale, padX, padY);

    // 1/255 scaling, BGR -> RGB, no crop  =>  NCHW float blob [1,3,640,640]
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                           cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false);

    // ----------------------------------------------------------- inference
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty())
    {
        std::cerr << "Network produced no output.\n";
        return 1;
    }

    // YOLO11 detect output: [1, 4 + numClasses, numAnchors] = [1, 84, 8400]
    // Each column is: cx, cy, w, h, class0 ... class79 (no objectness score).
    const cv::Mat &raw = outputs[0];
    const int numChannels = raw.size[1];
    const int numAnchors = raw.size[2];
    const int numClasses = numChannels - 4;

    // Flatten to 84 x 8400, then transpose to 8400 x 84 so each row is one anchor.
    cv::Mat flat(numChannels, numAnchors, CV_32F, const_cast<float *>(raw.ptr<float>()));
    cv::Mat preds = flat.t();

    // -------------------------------------------------------- decode boxes
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

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

        // Undo letterbox: remove padding, divide by scale, clamp to image.
        float x1 = (cx - w / 2.f - padX) / scale;
        float y1 = (cy - h / 2.f - padY) / scale;
        float x2 = (cx + w / 2.f - padX) / scale;
        float y2 = (cy + h / 2.f - padY) / scale;

        x1 = std::max(0.f, std::min(x1, static_cast<float>(image.cols - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(image.rows - 1)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(image.cols - 1)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(image.rows - 1)));

        boxes.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        confidences.push_back(static_cast<float>(maxScore));
        classIds.push_back(classPoint.x);
    }

    // Non-max suppression (class-aware via per-class NMS)
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, keep);

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

    std::ofstream csv("detections.csv");
    csv << "x1,y1,x2,y2,confidence,class_id,class_name\n";

    std::cout << "Detections: " << detections.size() << "\n";
    std::cout << std::left << std::setw(8) << "x1" << std::setw(8) << "y1"
              << std::setw(8) << "x2" << std::setw(8) << "y2"
              << std::setw(12) << "confidence" << std::setw(10) << "class_id"
              << "class_name\n";

    for (const Detection &d : detections)
    {
        const std::string &name = (d.classId >= 0 && d.classId < static_cast<int>(CLASS_NAMES.size()))
                                      ? CLASS_NAMES[d.classId]
                                      : "unknown";

        csv << std::fixed << std::setprecision(2)
            << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ","
            << std::setprecision(4) << d.confidence << ","
            << d.classId << "," << name << "\n";

        std::cout << std::fixed << std::setprecision(1) << std::left
                  << std::setw(8) << d.x1 << std::setw(8) << d.y1
                  << std::setw(8) << d.x2 << std::setw(8) << d.y2
                  << std::setprecision(4) << std::setw(12) << d.confidence
                  << std::setw(10) << d.classId << name << "\n";
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

    cv::imwrite("annotated.jpg", annotated);
    std::cout << "\nSaved annotated.jpg and detections.csv\n";
    return 0;
}