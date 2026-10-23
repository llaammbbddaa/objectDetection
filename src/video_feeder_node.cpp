#include "yolo26_ros/fs_utils.hpp"
#include <rclcpp/rclcpp.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <limits>
using namespace yolo26;
class VideoFeeder : public rclcpp::Node {
 cv::VideoCapture video_;
 cv::Mat current_,lookahead_;
 fs::path output_;
 std::string prefix_,format_;
 double fps_=0,currentTime_=0,nextTime_=0,lastTimestamp_=-1;
 int64_t interval_=100,index_=1,decoded_=0,sample_=0;
 size_t written_=0;
 int compression_=1;
 bool loop_=false,dump_=false,eof_=false;
 rclcpp::TimerBase::SharedPtr timer_;
 std::chrono::steady_clock::time_point due_;
public:
 VideoFeeder():Node("video_feeder_node") {
  auto path=declare_parameter<std::string>("video_path","");
  if(path.empty()) throw std::invalid_argument("video_path is required");
  output_=resolved(declare_parameter<std::string>("output_dir","camera_feed_images"));
  interval_=declare_parameter<int64_t>("frame_interval_ms",100);
  index_=declare_parameter<int64_t>("start_index",1);
  prefix_=declare_parameter<std::string>("file_prefix","img");
  format_=declare_parameter<std::string>("image_format","png");
  auto compression=declare_parameter<int64_t>("png_compression",1);
  loop_=declare_parameter<bool>("loop",false); dump_=declare_parameter<bool>("dump_all",false);
  if(interval_<=0 || interval_>86400000 || index_<1 || compression<0 || compression>9 || (format_!="png"&&format_!="jpg"))
   throw std::invalid_argument("invalid interval (1..86400000 ms), start_index (>=1), compression (0..9), or image_format (png/jpg)");
  if(prefix_.empty() || prefix_[0]=='.' || prefix_.find('/')!=std::string::npos || prefix_.find('\\')!=std::string::npos)
   throw std::invalid_argument("file_prefix must be a non-hidden filename prefix without separators");
  compression_=static_cast<int>(compression);
  if(!video_.open(resolved(path).string())) throw std::runtime_error("cannot open video: "+path);
  fps_=video_.get(cv::CAP_PROP_FPS);
  if(!std::isfinite(fps_)||fps_<=0) fps_=0;
  fs::create_directories(output_);
  RCLCPP_INFO(get_logger(),"Video %s resolution=%.0fx%.0f fps=%.3f frames=%.0f interval=%ldms",path.c_str(),video_.get(cv::CAP_PROP_FRAME_WIDTH),video_.get(cv::CAP_PROP_FRAME_HEIGHT),fps_,video_.get(cv::CAP_PROP_FRAME_COUNT),static_cast<long>(interval_));
  if(fps_>0 && fps_<1000.0/interval_) RCLCPP_WARN(get_logger(),"Source fps below sampling rate; source frames will repeat");
  due_=std::chrono::steady_clock::now()+std::chrono::milliseconds(interval_);
  timer_=create_wall_timer(dump_?std::chrono::nanoseconds(1):std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(interval_)),[this]{tick();});
 }
private:
 bool read(cv::Mat& frame,double& timestamp) {
  if(!video_.read(frame) || frame.empty()) return false;
  timestamp=fps_>0?decoded_*1000.0/fps_:video_.get(cv::CAP_PROP_POS_MSEC);
  if(!std::isfinite(timestamp)||timestamp<0 || (decoded_>0 && timestamp<=lastTimestamp_))
   throw std::runtime_error("video has no usable monotonically increasing timestamps");
  lastTimestamp_=timestamp; ++decoded_; return true;
 }
 bool sample(cv::Mat& frame) {
  if(current_.empty()) {
   if(!read(current_,currentTime_)) return false;
   // T=0 is required; a source with no timestamp at or before zero is unusable.
   if(currentTime_>0) throw std::runtime_error("video has no frame at timestamp zero");
  }
  double target=static_cast<double>(sample_)*interval_;
  while(true) {
   if(lookahead_.empty() && !eof_) eof_=!read(lookahead_,nextTime_);
   if(!lookahead_.empty() && nextTime_<=target) { current_=lookahead_; lookahead_.release(); currentTime_=nextTime_; continue; }
   break;
  }
  // A CFR frame occupies [timestamp,timestamp+1/fps). For timestamp fallback,
  // the final observed frame interval estimates the last frame's duration.
  double end=currentTime_+(fps_>0?1000.0/fps_:lastStep_);
  if(eof_ && target>=end-1e-7) return false;
  if(!lookahead_.empty()) lastStep_=nextTime_-currentTime_;
  frame=current_; ++sample_; return true;
 }
 double lastStep_=0;
 void tick() {
  auto begin=std::chrono::steady_clock::now();
  if(!dump_ && begin<due_) return;
  try {
   cv::Mat frame;
   if(!sample(frame)) {
    if(!loop_) { RCLCPP_INFO(get_logger(),"Finished: %zu frames written",written_); timer_->cancel(); rclcpp::shutdown(); return; }
    // Reopen for looping; sampling within each pass always decodes sequentially.
    auto path=resolved(get_parameter("video_path").as_string());
    video_.release(); if(!video_.open(path.string())) throw std::runtime_error("cannot reopen video");
    current_.release(); lookahead_.release(); decoded_=0; sample_=0; lastTimestamp_=-1; eof_=false; lastStep_=0;
    if(!sample(frame)) throw std::runtime_error("video contains no sampleable frames");
   }
   auto target=output_/(prefix_+std::to_string(index_)+"."+format_);
   atomicWrite(target,[&](const fs::path& temp) {
    std::vector<int> params;
    if(format_=="png") params={cv::IMWRITE_PNG_COMPRESSION,compression_};
    if(!cv::imwrite(temp.string(),frame,params)) throw std::runtime_error("frame write failed"); });
   ++written_;
   if(index_==std::numeric_limits<int64_t>::max()) throw std::runtime_error("filename index exhausted");
   ++index_;
   if(written_%50==0) RCLCPP_INFO(get_logger(),"Written %zu frames",written_);
   auto end=std::chrono::steady_clock::now();
   if(end-begin>std::chrono::milliseconds(interval_)) RCLCPP_WARN(get_logger(),"Tick exceeded %ldms interval",static_cast<long>(interval_));
   // After a slow tick, resume a full interval from this tick; never burst to catch up.
   due_=std::max(due_+std::chrono::milliseconds(interval_),end);
  } catch(const std::exception& e) { RCLCPP_ERROR(get_logger(),"Feeder failed: %s",e.what()); timer_->cancel(); failed=true; rclcpp::shutdown(); }
 }
public:
 bool failed=false;
};
int main(int argc,char** argv) {
 rclcpp::init(argc,argv);
 try { auto node=std::make_shared<VideoFeeder>(); rclcpp::spin(node); bool failed=node->failed; rclcpp::shutdown(); return failed?1:0; }
 catch(const std::exception& e) { RCLCPP_ERROR(rclcpp::get_logger("video_feeder_node"),"Startup failed: %s",e.what()); rclcpp::shutdown(); return 1; }
}
