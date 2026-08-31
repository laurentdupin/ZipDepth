#pragma once

#include "external_gpu.h"
#include "zipdepth_native.h"

#include <memory>

namespace zipdepth_native {

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    zipdepth_context* context);

}  // namespace zipdepth_native
