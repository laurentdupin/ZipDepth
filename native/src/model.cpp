#include "model.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace zipdepth_native {
namespace {

constexpr char kMagic[8] = {'Z','I','P','D','M','O','D','1'};
constexpr char kMetadataMagic[8] = {'Z','I','P','M','E','T','A','1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndian = 0x01020304;
constexpr std::uint32_t kFloat32 = 1;

#pragma pack(push, 1)
struct Header {
    char magic[8];
    std::uint32_t version, endian, kind, tensor_count;
    std::uint64_t directory_offset, directory_bytes, data_offset, file_bytes,
        metadata_offset;
};
struct Record {
    char name[112];
    std::uint32_t dtype, rank;
    std::uint64_t dimensions[4], data_offset, data_bytes, element_count;
    std::uint32_t crc32, flags;
    std::uint64_t reserved;
};
struct Metadata {
    char magic[8];
    std::uint32_t version, bytes, format_version, kind, flags, reserved;
    std::uint8_t canonical_sha256[32];
    char converter[64];
};
#pragma pack(pop)
static_assert(sizeof(Header) == 64);
static_assert(sizeof(Record) == 192);
static_assert(sizeof(Metadata) == 128);

bool range(std::uint64_t offset, std::uint64_t bytes, std::uint64_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

}  // namespace

ModelFile::ModelFile(const std::string& path_utf8) {
    if (path_utf8.empty()) throw std::invalid_argument("model path is empty");
    std::ifstream source(path_utf8, std::ios::binary | std::ios::ate);
    if (!source) throw std::runtime_error("failed to open ZipDepth model");
    const auto end = source.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) >
            std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("invalid ZipDepth model size");
    bytes_.resize(static_cast<std::size_t>(end));
    source.seekg(0);
    source.read(reinterpret_cast<char*>(bytes_.data()),
                static_cast<std::streamsize>(bytes_.size()));
    if (!source || bytes_.size() < sizeof(Header))
        throw std::runtime_error("truncated ZipDepth model");
    const auto& header = *reinterpret_cast<const Header*>(bytes_.data());
    if (std::memcmp(header.magic, kMagic, 8) != 0 ||
        header.version != kVersion || header.endian != kEndian ||
        header.kind > ZIPDEPTH_MODEL_BASE_MOBILE ||
        header.tensor_count == 0 || header.tensor_count > 2048 ||
        header.file_bytes != bytes_.size() ||
        header.directory_bytes !=
            std::uint64_t(header.tensor_count) * sizeof(Record) ||
        !range(header.directory_offset, header.directory_bytes, bytes_.size()) ||
        !range(header.metadata_offset, sizeof(Metadata), bytes_.size()))
        throw std::runtime_error("invalid ZipDepth model header");
    const auto& metadata = *reinterpret_cast<const Metadata*>(
        bytes_.data() + header.metadata_offset);
    if (std::memcmp(metadata.magic, kMetadataMagic, 8) != 0 ||
        metadata.version != 1 || metadata.bytes != sizeof(Metadata) ||
        metadata.format_version != kVersion || metadata.kind != header.kind ||
        metadata.flags != 0 || metadata.reserved != 0 ||
        std::memchr(metadata.converter, '\0', sizeof(metadata.converter)) == nullptr)
        throw std::runtime_error("invalid ZipDepth derivation metadata");
    derivation_.kind = static_cast<zipdepth_model_kind>(header.kind);
    derivation_.format_version = metadata.format_version;
    derivation_.converter = metadata.converter;
    std::memcpy(derivation_.canonical_sha256.data(), metadata.canonical_sha256, 32);
    const auto* records = reinterpret_cast<const Record*>(
        bytes_.data() + header.directory_offset);
    tensors_.reserve(header.tensor_count);
    names_.reserve(header.tensor_count);
    for (std::uint32_t i = 0; i < header.tensor_count; ++i) {
        const Record& record = records[i];
        const char* terminator = static_cast<const char*>(
            std::memchr(record.name, '\0', sizeof(record.name)));
        if (!terminator || record.name[0] == '\0' ||
            record.dtype != kFloat32 || record.rank == 0 || record.rank > 4 ||
            record.flags != 0 || record.reserved != 0 ||
            !range(record.data_offset, record.data_bytes, bytes_.size()))
            throw std::runtime_error("invalid ZipDepth tensor record");
        std::uint64_t count = 1;
        for (std::uint32_t d = 0; d < 4; ++d) {
            if (d < record.rank) {
                if (record.dimensions[d] == 0 ||
                    count > std::numeric_limits<std::uint64_t>::max() /
                        record.dimensions[d])
                    throw std::runtime_error("invalid ZipDepth tensor shape");
                count *= record.dimensions[d];
            } else if (record.dimensions[d] != 0) {
                throw std::runtime_error("invalid unused tensor dimension");
            }
        }
        if (count != record.element_count || record.data_bytes != count * 4)
            throw std::runtime_error("invalid ZipDepth tensor byte count");
        std::string name(record.name, terminator);
        TensorView view{
            reinterpret_cast<const float*>(bytes_.data() + record.data_offset),
            {record.dimensions[0], record.dimensions[1],
             record.dimensions[2], record.dimensions[3]},
            record.rank, count};
        if (!tensors_.emplace(name, view).second)
            throw std::runtime_error("duplicate ZipDepth tensor");
        names_.push_back(std::move(name));
    }
}

const TensorView& ModelFile::tensor(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end())
        throw std::runtime_error("missing ZipDepth tensor: " + std::string(name));
    return found->second;
}

bool ModelFile::contains(std::string_view name) const {
    return tensors_.find(std::string(name)) != tensors_.end();
}

}  // namespace zipdepth_native
