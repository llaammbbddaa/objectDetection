#include "yolo26_ros/yolo26_detector.hpp"
#include "yolo26_ros/file_watcher.hpp"
#include "yolo26_ros/fs_utils.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <memory>
#include <future>
#include <array>
using namespace yolo26;
using Clock=std::chrono::steady_clock;
static double milliseconds(Clock::duration d) { return std::chrono::duration<double,std::milli>(d).count(); }
class DetectorNode : public rclcpp::Node {
 struct Frame { std::string name; Clock::time_point arrival; };
 fs::path watch_,output_;
 std::string model_,tiling_;
 float conf_,nms_;
 TileLayout fixed_{0,0};
 std::unique_ptr<FileWatcher> watcher_;
 std::thread watchThread_,worker_;
 std::atomic<bool> stopped_{false},watchFailed_{false};
 rclcpp::TimerBase::SharedPtr health_;
 std::mutex mutex_,stopMutex_;
 std::condition_variable cv_;
 std::deque<Frame> queue_;
 std::set<std::string> waiting_;
 std::set<std::pair<int,int>> sizes_;
 std::vector<double> detectTimes_,latencies_;
 size_t failed_=0,maxDepth_=0,warnAt_=10;
public:
 DetectorNode():Node("detector_node") {
  watch_=resolved(declare_parameter<std::string>("watch_dir","camera_feed_images"));
  output_=resolved(declare_parameter<std::string>("output_dir","annotated_output"));
  model_=resolved(declare_parameter<std::string>("model_path","yolo26.onnx")).string();
  auto c=declare_parameter<double>("conf_threshold",0.25),n=declare_parameter<double>("nms_threshold",0.45);
  if(!std::isfinite(c)||!std::isfinite(n)||c<0||c>1||n<0||n>1) throw std::invalid_argument("thresholds must be finite and in [0,1]");
  conf_=c; nms_=n;
  tiling_=declare_parameter<std::string>("tiling","auto");
  if(tiling_!="auto" && tiling_!="none") { auto r=parseResolution(tiling_); fixed_=layoutFromResolution(r.width,r.height); }
  if(inside(output_,watch_)) throw std::invalid_argument("output_dir must not be inside watch_dir");
  fs::create_directories(watch_); fs::create_directories(output_/"annotated_images"); fs::create_directories(output_/"csv");
 }
 void start() {
  // Model construction and every subsequent Net call belong to this worker alone.
  std::promise<void> ready;
  auto future=ready.get_future();
  worker_=std::thread([this,&ready] {
   std::unique_ptr<Detector> detector;
   try { detector=std::make_unique<Detector>(model_); ready.set_value(); }
   catch(...) { ready.set_exception(std::current_exception()); return; }
   work(*detector);
  });
  try {
   future.get();
   watcher_=std::make_unique<FileWatcher>(watch_.string());
   watchThread_=std::thread([this] {
    try { watcher_->run([this](std::string name,Clock::time_point time) {
     std::lock_guard<std::mutex> lock(mutex_);
     if(stopped_ || !waiting_.insert(name).second) return;
     queue_.push_back({std::move(name),time});
     maxDepth_=std::max(maxDepth_,queue_.size());
     if(queue_.size()>warnAt_) { RCLCPP_WARN(get_logger(),"Detector falling behind feed: queue_depth=%zu",queue_.size()); warnAt_*=2; }
     cv_.notify_one();
    },[this](const std::string& message) { RCLCPP_WARN(get_logger(),"%s",message.c_str()); }); }
    catch(const std::exception& e) { RCLCPP_ERROR(get_logger(),"Watcher: %s",e.what()); watchFailed_=true; }
   });
   health_=create_wall_timer(std::chrono::milliseconds(100),[this]{if(watchFailed_) rclcpp::shutdown();});
   RCLCPP_INFO(get_logger(),"Ready: watching %s",watch_.c_str());
  } catch(...) { stop(); throw; }
 }
 void stop() {
  std::lock_guard<std::mutex> stopping(stopMutex_);
  if(stopped_.exchange(true)) return;
  if(watcher_) watcher_->stop();
  cv_.notify_all();
  if(watchThread_.joinable()) watchThread_.join();
  if(worker_.joinable()) worker_.join();
  size_t left; { std::lock_guard<std::mutex> lock(mutex_); left=queue_.size(); }
  auto stats=[](std::vector<double> values) {
   if(values.empty()) return std::array<double,3>{0,0,0};
   double sum=0; for(auto v:values) sum+=v;
   std::sort(values.begin(),values.end());
   return std::array<double,3>{sum/values.size(),values[static_cast<size_t>(std::ceil(values.size()*0.95))-1],values.back()};
  };
  auto d=stats(detectTimes_),l=stats(latencies_);
  RCLCPP_INFO(get_logger(),"Summary: processed=%zu failed=%zu unprocessed=%zu detect mean/p95/max=%.3f/%.3f/%.3fms latency mean/p95/max=%.3f/%.3f/%.3fms max_queue_depth=%zu",detectTimes_.size(),failed_,left,d[0],d[1],d[2],l[0],l[1],l[2],maxDepth_);
 }
 ~DetectorNode() override { stop(); }
private:
 void work(Detector& detector) {
  while(true) {
   Frame frame; size_t depth; Clock::time_point dequeued;
   { std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[this]{return stopped_||!queue_.empty();});
    if(stopped_) return;
    frame=queue_.front(); queue_.pop_front(); waiting_.erase(frame.name); depth=queue_.size(); dequeued=Clock::now(); }
   try {
    cv::Mat image;
    for(int attempt=0;attempt<4;++attempt) {
     try { image=cv::imread((watch_/frame.name).string()); } catch(const cv::Exception&) { if(attempt==3) throw; }
     if(!image.empty()) break;
     if(attempt<3) { std::unique_lock<std::mutex> lock(mutex_); if(cv_.wait_for(lock,std::chrono::milliseconds(50),[this]{return stopped_.load();})) return; }
    }
    if(image.empty()) throw std::runtime_error("could not decode image after 3 retries");
    auto layout=tiling_=="auto"?autoLayout(image.size()):fixed_;
    if(sizes_.insert({image.cols,image.rows}).second)
     RCLCPP_INFO(get_logger(),"Image size %dx%d -> %dx%d tiles",image.cols,image.rows,std::max(1,layout.columns),std::max(1,layout.rows));
    auto begin=Clock::now(); auto detections=detector.detect(image,conf_,nms_,layout); auto end=Clock::now();
    double detection=milliseconds(end-begin);
    fs::path name(frame.name);
    atomicWrite(output_/"annotated_images"/("annotated_"+frame.name),[&](const fs::path& temp) {
     if(!cv::imwrite(temp.string(),annotate(image,detections))) throw std::runtime_error("image write failed"); });
    atomicWrite(output_/"csv"/(name.stem().string()+".csv"),[&](const fs::path& temp) {
     if(!writeCsv(temp.string(),detections,image.size(),detection)) throw std::runtime_error("CSV write failed"); });
    auto done=Clock::now(); double latency=milliseconds(done-frame.arrival);
    detectTimes_.push_back(detection); latencies_.push_back(latency);
    RCLCPP_INFO(get_logger(),"[%s] detections=%zu detect=%.3fms queue_wait=%.3fms total=%.3fms latency=%.3fms queue_depth=%zu",frame.name.c_str(),detections.size(),detection,milliseconds(dequeued-frame.arrival),milliseconds(done-dequeued),latency,depth);
   } catch(const std::exception& e) { ++failed_; RCLCPP_ERROR(get_logger(),"[%s] %s",frame.name.c_str(),e.what()); }
  }
 }
};
int main(int argc,char** argv) {
 rclcpp::init(argc,argv);
 try {
  auto node=std::make_shared<DetectorNode>();
  std::weak_ptr<DetectorNode> weak=node;
  rclcpp::on_shutdown([weak]{if(auto n=weak.lock()) n->stop();});
  node->start(); rclcpp::spin(node); node->stop();
 } catch(const std::exception& e) { RCLCPP_ERROR(rclcpp::get_logger("detector_node"),"Startup failed: %s",e.what()); rclcpp::shutdown(); return 1; }
 rclcpp::shutdown(); return 0;
}
