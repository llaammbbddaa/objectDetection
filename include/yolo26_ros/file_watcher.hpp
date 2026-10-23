#pragma once
#include <chrono>
#include <functional>
#include <string>
namespace yolo26 {
class FileWatcher {
public:
 using Arrival=std::chrono::steady_clock::time_point;
 explicit FileWatcher(const std::string& directory);
 ~FileWatcher();
 FileWatcher(const FileWatcher&)=delete;
 FileWatcher& operator=(const FileWatcher&)=delete;
 void run(const std::function<void(std::string,Arrival)>& arrival,
          const std::function<void(const std::string&)>& warning);
 void stop() noexcept;
private:
 int fd_=-1, wake_=-1;
};
}
