#pragma once

#include "model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace zipdepth_native {

struct CpuTensor {
    std::uint32_t channels = 0, height = 0, width = 0;
    std::vector<float> data;
};

class CpuExecutor {
public:
    explicit CpuExecutor(const std::string& model_path) : model_(model_path) {}
    zipdepth_model_kind kind() const { return model_.kind(); }
    CpuTensor infer(const float* rgb_chw, std::uint32_t width,
                    std::uint32_t height) const;

private:
    CpuTensor conv(const CpuTensor& input, const std::string& weight,
                   const char* bias, std::uint32_t stride = 1,
                   std::uint32_t padding = 0, std::uint32_t dilation = 1,
                   std::uint32_t groups = 1) const;
    CpuTensor bn(const CpuTensor& input, const std::string& prefix,
                 bool relu) const;
    CpuTensor conv_bn(const CpuTensor& input, const std::string& prefix,
                      std::uint32_t stride = 1, bool relu = true) const;
    CpuTensor rep(const CpuTensor& input, const std::string& prefix,
                  std::uint32_t stride = 1) const;
    CpuTensor fusion(const CpuTensor& high, const CpuTensor& low,
                     const std::string& prefix) const;
    ModelFile model_;
};

CpuTensor resize_bilinear(const CpuTensor& input, std::uint32_t width,
                          std::uint32_t height, bool align_corners);

}  // namespace zipdepth_native
