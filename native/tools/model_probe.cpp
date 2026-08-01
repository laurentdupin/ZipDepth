#include "model.h"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        zipdepth_native::ModelFile model(argv[1]);
        std::printf("kind=%u tensors=%zu\n",
            static_cast<unsigned>(model.kind()), model.tensor_names().size());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
