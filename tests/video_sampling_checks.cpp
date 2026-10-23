// Compare every emitted sample with the expected sequentially decoded source frame.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
 assert(argc==3);
 cv::VideoCapture video(argv[1]); assert(video.isOpened());
 double fps=video.get(cv::CAP_PROP_FPS); assert(fps>0);
 std::vector<cv::Mat> source;
 cv::Mat frame;
 while(video.read(frame)) source.push_back(frame.clone());
 int samples=static_cast<int>(std::ceil(source.size()/fps*10-1e-7));
 for(int i=0;i<samples;++i) {
  auto expected=static_cast<size_t>(std::floor(i*0.1*fps+1e-7));
  auto path=std::filesystem::path(argv[2])/("img"+std::to_string(i+1)+".png");
  auto actual=cv::imread(path.string());
  assert(!actual.empty());
  assert(actual.size()==source.at(expected).size());
  assert(cv::norm(actual,source.at(expected),cv::NORM_INF)==0);
 }
 assert(!std::filesystem::exists(std::filesystem::path(argv[2])/("img"+std::to_string(samples+1)+".png")));
 std::cout<<"PASS "<<samples<<" sample pixels match source timestamps at "<<fps<<" fps\n";
}
