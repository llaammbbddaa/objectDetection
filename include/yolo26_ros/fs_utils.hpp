#pragma once
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
namespace yolo26 {
namespace fs = std::filesystem;
inline fs::path resolved(const std::string& p) {
 if(p.empty()) throw std::invalid_argument("path must not be empty");
 return fs::weakly_canonical(fs::absolute(p));
}
inline bool inside(const fs::path& child,const fs::path& parent) {
 auto c=child.begin();
 for(auto p=parent.begin();p!=parent.end();++p,++c)
  if(c==child.end() || *c!=*p) return false;
 return true;
}
inline void atomicWrite(const fs::path& target,const std::function<void(const fs::path&)>& write) {
 auto temp=target.parent_path()/(target.extension()==".csv"
  ? "."+target.filename().string()+".tmp"
  : "."+target.stem().string()+".tmp"+target.extension().string());
 try { write(temp); fs::rename(temp,target); }
 catch(...) { std::error_code ec; fs::remove(temp,ec); throw; }
}
}
