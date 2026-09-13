#include "model.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() / fs::u8path(
        u8"zipdepth-\u4e2d\u6587-\u00e9-\u65e5\u672c\u8a9e-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    bool passed = true;
    for (const auto* name : {u8"ascii.zipd", u8"\u6a21\u578b-\U0001f4c1.zipd"}) {
        const auto path = directory / fs::u8path(name);
        {
            std::ofstream fixture(path, std::ios::binary);
            fixture << "truncated fixture";
            if (!fixture) return 1;
        }
        try {
            zipdepth_native::ModelFile model(path.u8string());
            passed = false;
        } catch (const std::exception& error) {
            // Reaching validation proves the existing file was opened correctly.
            if (std::string(error.what()) != "truncated ZipDepth model") {
                std::cerr << error.what() << '\n';
                passed = false;
            }
        }
        fs::remove(path);
    }
    fs::remove(directory);
    return passed ? 0 : 1;
}
