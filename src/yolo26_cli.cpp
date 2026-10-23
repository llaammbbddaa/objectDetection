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
using namespace yolo26;
static constexpr int INPUT_SIZE=640;
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

static bool processImage(const std::string& imagePath,const std::string& csvPath,const std::string& annotatedPath,Detector& net,float c,float n,int columns,int rows) {
 auto image=cv::imread(imagePath);
 if(image.empty()) { std::cerr<<"Could not read image: "<<imagePath<<"\n"; return false; }
 auto start=std::chrono::steady_clock::now();
 auto detections=net.detectLegacy(image,c,n,{columns,rows});
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 if(!writeCsv(csvPath,detections,image.size(),ms)) return false;
 std::cout<<imagePath<<": "<<detections.size()<<" detections\n";
 if(!cv::imwrite(annotatedPath,annotate(image,detections))) return false;
 std::cout<<"  CSV: "<<csvPath<<"\n  Image: "<<annotatedPath<<"\n";
 return true;
}
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> [conf_thresh=0.25] [nms_thresh=0.45]"
                  << " [--resolution WIDTHxHEIGHT]\n"
                  << "       " << argv[0]
                  << " <input_directory> <csv_directory> <annotated_directory>"
                  << " [conf_thresh=0.25] [nms_thresh=0.45] [--resolution WIDTHxHEIGHT]\n";
        return 1;
    }

    try
    {
        std::vector<std::string> positional;
        Resolution resolution{0, 0};
        bool hasResolution = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--resolution")
            {
                if (hasResolution || i + 1 >= argc)
                    throw std::invalid_argument("--resolution requires WIDTHxHEIGHT");
                resolution = parseResolution(argv[++i]);
                hasResolution = true;
            }
            else if (argument.rfind("--resolution=", 0) == 0)
            {
                if (hasResolution)
                    throw std::invalid_argument("resolution specified more than once");
                resolution = parseResolution(argument.substr(13));
                hasResolution = true;
            }
            else
            {
                positional.push_back(argument);
            }
        }

        if (positional.empty())
            throw std::invalid_argument("an input image or directory is required");

        const std::string inputPath = positional[0];
        const bool directoryMode = isDirectory(inputPath);
        positional.erase(positional.begin());
        const std::string csvDirectory =
            directoryMode ? (positional.size() > 0 ? positional[0] : "csv_output") : ".";
        const std::string annotatedDirectory =
            directoryMode ? (positional.size() > 1 ? positional[1] : "annotated_output") : ".";
        const std::size_t thresholdOffset = directoryMode ? 2 : 0;
        const float confThresh = positional.size() > thresholdOffset
                                     ? std::stof(positional[thresholdOffset])
                                     : 0.25f;
        const float nmsThresh = positional.size() > thresholdOffset + 1
                                    ? std::stof(positional[thresholdOffset + 1])
                                    : 0.45f;

        if (positional.size() > thresholdOffset + 2)
            throw std::invalid_argument("too many positional arguments");

        if (confThresh < 0.0f || confThresh > 1.0f || nmsThresh < 0.0f ||
            nmsThresh > 1.0f)
        {
            throw std::invalid_argument("thresholds must be between 0 and 1");
        }

        Detector net("yolo26.onnx");
        if (!directoryMode)
        {
            const int tileColumns =
                hasResolution ? static_cast<int>(std::ceil(
                                    static_cast<double>(resolution.width) / INPUT_SIZE))
                              : 0;
            const int tileRows =
                hasResolution ? static_cast<int>(std::ceil(
                                    static_cast<double>(resolution.height) / INPUT_SIZE))
                              : 0;
            return processImage(inputPath, "detections.csv", "annotated.jpg", net,
                                confThresh, nmsThresh, tileColumns, tileRows)
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

        const int tileColumns =
            hasResolution
                ? static_cast<int>(
                      std::ceil(static_cast<double>(resolution.width) / INPUT_SIZE))
                : 2;
        const int tileRows =
            hasResolution
                ? static_cast<int>(
                      std::ceil(static_cast<double>(resolution.height) / INPUT_SIZE))
                : 2;

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
            if (processImage(path, csvPath, annotatedPath, net, confThresh, nmsThresh,
                             tileColumns, tileRows))
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
