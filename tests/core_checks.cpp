// Standalone checks: no ROS or ONNX model required.
#include "yolo26_ros/yolo26_detector.hpp"
#include "yolo26_ros/file_watcher.hpp"
#include "yolo26_ros/fs_utils.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cassert>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>
#include <iostream>
#include <unistd.h>
using namespace yolo26;
int main() {
 auto root=fs::temp_directory_path()/("yolo26-check-"+std::to_string(getpid()));
 fs::create_directories(root);
 struct Cleanup { fs::path p; ~Cleanup(){ fs::remove_all(p); } } cleanup{root};
 auto l=autoLayout({1920,1080}); assert(l.columns==3&&l.rows==2);
 l=layoutFromResolution(1280,720); assert(l.columns==2&&l.rows==2);
 l=autoLayout({640,480}); assert(l.columns==1&&l.rows==1);
 for(auto bad:{"auto","0x720","1280x-1","1280x720junk"}) {
  bool rejected=false; try { parseResolution(bad); } catch(const std::invalid_argument&) { rejected=true; } assert(rejected);
 }
 assert(inside(root/"nested",root)); assert(!inside(root.string()+"-other",root));
 cv::Mat image(40,60,CV_8UC3,cv::Scalar(0,0,0));
 auto annotated=annotate(image,{{1,1,30,30,0.75f,0}});
 assert(cv::countNonZero(image.reshape(1))==0);
 assert(cv::countNonZero(annotated.reshape(1))>0);
 assert(writeCsv((root/"empty.csv").string(),{},image.size(),12.5));
 std::ifstream csv(root/"empty.csv"); std::string line;
 std::getline(csv,line); assert(line=="x1,y1,x2,y2,confidence,class_id,class_name,resolutionX,resolutionY,detection_time"); assert(!std::getline(csv,line));
 // Existing images must not be emitted, even though the watcher is started later.
 cv::imwrite((root/"existing.png").string(),image);
 FileWatcher watcher(root.string());
 std::mutex mutex; std::condition_variable condition; std::vector<std::string> seen;
 std::thread thread([&] { watcher.run([&](std::string name,FileWatcher::Arrival) {
  std::lock_guard<std::mutex> lock(mutex); seen.push_back(name); condition.notify_all();
 },[](const std::string& warning){ throw std::runtime_error(warning); }); });
 auto await=[&](size_t count) { std::unique_lock<std::mutex> lock(mutex); return condition.wait_for(lock,std::chrono::seconds(2),[&]{return seen.size()>=count;}); };
 { std::ofstream ignored(root/"ignored.txt"); }
 { std::ofstream ignored(root/".hidden.png"); }
 fs::create_directory(root/"directory.png");
 std::ofstream partial(root/"slow.png"); partial<<"incomplete"; partial.flush();
 assert(!await(1));
 partial.close(); assert(await(1));
 atomicWrite(root/"moved.PNG",[&](const fs::path& temp){ assert(cv::imwrite(temp.string(),image)); });
 assert(await(2));
 watcher.stop(); thread.join();
 assert((seen==std::vector<std::string>{"slow.png","moved.PNG"}));
 std::cout<<"Core layout, CSV, annotation, atomic-write and inotify checks passed\n";
}
