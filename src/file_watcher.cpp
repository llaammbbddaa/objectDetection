#include "yolo26_ros/file_watcher.hpp"
#include <sys/inotify.h>
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <set>
#include <stdexcept>
namespace yolo26 {
FileWatcher::FileWatcher(const std::string& directory) {
 fd_=inotify_init1(IN_NONBLOCK|IN_CLOEXEC);
 if(fd_<0) throw std::runtime_error("inotify_init: "+std::string(strerror(errno)));
 wake_=eventfd(0,EFD_NONBLOCK|EFD_CLOEXEC);
 if(wake_<0 || inotify_add_watch(fd_,directory.c_str(),IN_CLOSE_WRITE|IN_MOVED_TO|IN_DELETE_SELF|IN_MOVE_SELF)<0) {
  auto message=std::string(strerror(errno));
  if(wake_>=0) close(wake_);
  close(fd_); throw std::runtime_error("watch setup: "+message);
 }
}
FileWatcher::~FileWatcher() { close(fd_); close(wake_); }
void FileWatcher::stop() noexcept { uint64_t one=1; auto ignored=write(wake_,&one,sizeof(one)); (void)ignored; }
void FileWatcher::run(const std::function<void(std::string,Arrival)>& arrival,const std::function<void(const std::string&)>& warning) {
 pollfd descriptors[2]={{fd_,POLLIN,0},{wake_,POLLIN,0}};
 alignas(inotify_event) char buffer[65536];
 const std::set<std::string> extensions={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".ppm"};
 while(true) {
  int result=poll(descriptors,2,-1);
  if(result<0) { if(errno==EINTR) continue; throw std::runtime_error("inotify poll failed"); }
  if(descriptors[1].revents) return;
  if(descriptors[0].revents&(POLLERR|POLLHUP|POLLNVAL)) throw std::runtime_error("inotify unavailable");
  while(true) {
   auto count=read(fd_,buffer,sizeof(buffer));
   if(count<0) { if(errno==EINTR) continue; if(errno==EAGAIN) break; throw std::runtime_error("inotify read failed"); }
   if(count==0) break;
   for(size_t offset=0;offset<static_cast<size_t>(count);) {
    auto* event=reinterpret_cast<const inotify_event*>(buffer+offset);
    offset+=sizeof(inotify_event)+event->len;
    if(event->mask&IN_Q_OVERFLOW) { warning("inotify queue overflow: arrivals may have been lost"); continue; }
    if(event->mask&(IN_DELETE_SELF|IN_MOVE_SELF|IN_IGNORED)) throw std::runtime_error("watched directory removed or moved");
    if(!event->len || event->mask&IN_ISDIR || !(event->mask&(IN_CLOSE_WRITE|IN_MOVED_TO))) continue;
    std::string name(event->name);
    if(name.empty() || name.front()=='.') continue;
    auto ext=std::filesystem::path(name).extension().string();
    for(char& c:ext) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(extensions.count(ext)) arrival(name,std::chrono::steady_clock::now());
   }
  }
 }
}
}
