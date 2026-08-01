#pragma once

#include "zipdepth_native.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace zipdepth_native {

struct TensorView {
    const float* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

struct Derivation {
    std::array<std::uint8_t, 32> canonical_sha256{};
    std::string converter;
    std::uint32_t format_version = 0;
    zipdepth_model_kind kind = ZIPDEPTH_MODEL_BASE_GPU;
};

class ModelFile {
public:
    explicit ModelFile(const std::string& path_utf8);
    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    const std::vector<std::string>& tensor_names() const { return names_; }
    const Derivation& derivation() const { return derivation_; }
    zipdepth_model_kind kind() const { return derivation_.kind; }

private:
    std::vector<std::byte> bytes_;
    std::unordered_map<std::string, TensorView> tensors_;
    std::vector<std::string> names_;
    Derivation derivation_;
};

}  // namespace zipdepth_native
