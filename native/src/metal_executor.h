#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace zipdepth_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const std::string& model_path);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;
    void infer(const float* rgb, std::uint32_t width, std::uint32_t height,
               float* depth, std::uint64_t elements);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zipdepth_native
