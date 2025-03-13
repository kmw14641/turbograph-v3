#include "storage/extent/extent_manager.hpp"
#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/list.hpp"
#include "storage/cache/chunk_cache_manager.h"
#include "storage/extent/compression/compression_function.hpp"

#include "common/directory_helper.hpp"
#include "common/types/data_chunk.hpp"
#include "main/client_context.hpp"
#include "main/database.hpp"
#include "parser/parsed_data/create_chunkdefinition_info.hpp"
#include "parser/parsed_data/create_extent_info.hpp"

namespace duckdb {

ExtentManager::ExtentManager() {}

ExtentID ExtentManager::CreateExtent(ClientContext &context, DataChunk &input,
                                     PartitionCatalogEntry &part_cat,
                                     PropertySchemaCatalogEntry &ps_cat)
{
    // Get New ExtentID & Create ExtentCatalogEntry
    PartitionID pid = part_cat.GetPartitionID();
    PropertySchemaID psid = ps_cat.GetOid();
    ExtentID new_eid = part_cat.GetNewExtentID();
    Catalog &cat_instance = context.db->GetCatalog();
    string extent_name = DEFAULT_EXTENT_PREFIX + std::to_string(new_eid);
    CreateExtentInfo extent_info(DEFAULT_SCHEMA, extent_name.c_str(),
                                 ExtentType::EXTENT, new_eid, pid, psid,
                                 input.size());
    ExtentCatalogEntry *extent_cat_entry =
        (ExtentCatalogEntry *)cat_instance.CreateExtent(context, &extent_info);

    // MkDir for the extent
    std::string extent_dir_path = DiskAioParameters::WORKSPACE + "/part_" +
                                  std::to_string(pid) + "/ext_" +
                                  std::to_string(new_eid);
    MkDir(extent_dir_path, true);

    // Append Chunk
    _AppendChunkToExtentWithCompression(context, input, cat_instance,
                                        *extent_cat_entry, pid, new_eid);
    _UpdatePartitionMinMaxArray(context, cat_instance, part_cat, ps_cat,
                                *extent_cat_entry);
    return new_eid;
}

void ExtentManager::CreateExtent(ClientContext &context, DataChunk &input,
                                 PartitionCatalogEntry &part_cat,
                                 PropertySchemaCatalogEntry &ps_cat,
                                 ExtentID new_eid)
{
    // Create ExtentCatalogEntry
    PartitionID pid = part_cat.GetPartitionID();
    PropertySchemaID psid = ps_cat.GetOid();
    Catalog &cat_instance = context.db->GetCatalog();
    string extent_name = DEFAULT_EXTENT_PREFIX + std::to_string(new_eid);
    CreateExtentInfo extent_info(DEFAULT_SCHEMA, extent_name.c_str(),
                                 ExtentType::EXTENT, new_eid, pid, psid,
                                 input.size());
    ExtentCatalogEntry *extent_cat_entry =
        (ExtentCatalogEntry *)cat_instance.CreateExtent(context, &extent_info);

    // MkDir for the extent
    std::string extent_dir_path = DiskAioParameters::WORKSPACE + "/part_" +
                                  std::to_string(pid) + "/ext_" +
                                  std::to_string(new_eid);
    MkDir(extent_dir_path, true);

    // Append Chunk
    _AppendChunkToExtentWithCompression(context, input, cat_instance,
                                        *extent_cat_entry, pid, new_eid);
    _UpdatePartitionMinMaxArray(context, cat_instance, part_cat, ps_cat,
                                *extent_cat_entry);
}

void ExtentManager::AppendChunkToExistingExtent(ClientContext &context,
                                                DataChunk &input, ExtentID eid)
{
    Catalog &cat_instance = context.db->GetCatalog();
    ExtentCatalogEntry *extent_cat_entry =
        (ExtentCatalogEntry *)cat_instance.GetEntry(
            context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
            DEFAULT_EXTENT_PREFIX + std::to_string(eid));
    PartitionID pid = static_cast<PartitionID>(eid >> 16);
    _AppendChunkToExtentWithCompression(context, input, cat_instance,
                                        *extent_cat_entry, pid, eid);
}

void ExtentManager::AppendTuplesToExistingExtent(ClientContext &context,
                                                 DataChunk &input, ExtentID eid)
{
    Catalog &cat_instance = context.db->GetCatalog();
    ExtentCatalogEntry *extent_cat_entry =
        (ExtentCatalogEntry *)cat_instance.GetEntry(
            context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
            DEFAULT_EXTENT_PREFIX + std::to_string(eid));
    PartitionID pid = static_cast<PartitionID>(eid >> 16);
    _AppendTuplesToExtentWithCompression(context, input, cat_instance,
                                         *extent_cat_entry, pid, eid);
}

// Helper class for type-specific operations
class ChunkWriter {
   protected:
    uint8_t *buf_ptr;
    size_t comp_header_size;
    CompressionHeader &comp_header;
    DataChunk &input;
    idx_t input_chunk_idx;
    size_t alloc_buf_size;

   public:
    ChunkWriter(uint8_t *buf_ptr, CompressionHeader &comp_header,
                DataChunk &input, idx_t input_chunk_idx, size_t alloc_buf_size)
        : buf_ptr(buf_ptr),
          comp_header_size(comp_header.GetSizeWoBitSet()),
          comp_header(comp_header),
          input(input),
          input_chunk_idx(input_chunk_idx),
          alloc_buf_size(alloc_buf_size)
    {}

    virtual void WriteData() = 0;
    virtual void WriteExistingAndNewData(
        uint8_t *existing_buf_ptr, CompressionHeader &existing_comp_header,
        idx_t existing_count) = 0;

    virtual ~ChunkWriter() = default;
};

class VarCharChunkWriter : public ChunkWriter {
   public:
    VarCharChunkWriter(uint8_t *buf_ptr, CompressionHeader &comp_header,
                       DataChunk &input, idx_t input_chunk_idx,
                       size_t alloc_buf_size)
        : ChunkWriter(buf_ptr, comp_header, input, input_chunk_idx,
                      alloc_buf_size)
    {}

    void WriteData() override
    {
        size_t input_size = input.size();
        size_t string_t_offset = comp_header_size;
        size_t string_data_offset =
            comp_header_size + input_size * sizeof(string_t);

        comp_header.SetSwizzlingType(SwizzlingType::SWIZZLE_VARCHAR);
        memcpy(buf_ptr, &comp_header, comp_header_size);

        string_t *string_buffer =
            (string_t *)input.data[input_chunk_idx].GetData();
        uint64_t accumulated_string_len = 0;

        for (size_t i = 0; i < input_size; i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_t &str = string_buffer[i];
                if (str.IsInlined()) {
                    memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
                }
                else {
                    // Copy actual string and update pointer
                    memcpy(
                        buf_ptr + string_data_offset + accumulated_string_len,
                        str.GetDataUnsafe(), str.GetSize());
                    string_t swizzled_str(
                        reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                                 accumulated_string_len),
                        str.GetSize());
                    memcpy(buf_ptr + string_t_offset, &swizzled_str,
                           sizeof(string_t));
                    accumulated_string_len += str.GetSize();
                }
            }
            string_t_offset += sizeof(string_t);
        }
    }

    void WriteExistingAndNewData(uint8_t *existing_buf_ptr,
                                 CompressionHeader &existing_comp_header,
                                 idx_t existing_count) override
    {
        size_t comp_header_size = comp_header.GetSizeWoBitSet();
        memcpy(buf_ptr, &comp_header, comp_header_size);

        size_t string_t_offset = comp_header_size;
        size_t string_data_offset =
            comp_header_size +
            (existing_count + input.size()) * sizeof(string_t);
        size_t accumulated_string_len = 0;

        // Copy existing strings
        string_t *existing_strings =
            (string_t *)(existing_buf_ptr +
                         existing_comp_header.GetSizeWoBitSet());
        for (idx_t i = 0; i < existing_count; i++) {
            string_t &str = existing_strings[i];
            if (str.IsInlined()) {
                memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
            }
            else {
                memcpy(buf_ptr + string_data_offset + accumulated_string_len,
                       str.GetDataUnsafe(), str.GetSize());
                string_t new_str(
                    reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                             accumulated_string_len),
                    str.GetSize());
                memcpy(buf_ptr + string_t_offset, &new_str, sizeof(string_t));
                accumulated_string_len += str.GetSize();
            }
            string_t_offset += sizeof(string_t);
        }

        // Append new strings
        string_t *new_strings =
            (string_t *)input.data[input_chunk_idx].GetData();
        for (idx_t i = 0; i < input.size(); i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_t &str = new_strings[i];
                if (str.IsInlined()) {
                    memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
                }
                else {
                    memcpy(
                        buf_ptr + string_data_offset + accumulated_string_len,
                        str.GetDataUnsafe(), str.GetSize());
                    string_t new_str(
                        reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                                 accumulated_string_len),
                        str.GetSize());
                    memcpy(buf_ptr + string_t_offset, &new_str,
                           sizeof(string_t));
                    accumulated_string_len += str.GetSize();
                }
            }
            string_t_offset += sizeof(string_t);
        }
    }
};

class ListChunkWriter : public ChunkWriter {
   public:
    ListChunkWriter(uint8_t *buf_ptr, CompressionHeader &comp_header,
                    DataChunk &input, idx_t input_chunk_idx,
                    size_t alloc_buf_size)
        : ChunkWriter(buf_ptr, comp_header, input, input_chunk_idx,
                      alloc_buf_size)
    {}

    void WriteData() override
    {
        size_t input_size = input.size();
        Vector &child_vec = ListVector::GetEntry(input.data[input_chunk_idx]);

        memcpy(buf_ptr, &comp_header, comp_header_size);
        memcpy(buf_ptr + comp_header_size,
               input.data[input_chunk_idx].GetData(),
               input_size * sizeof(list_entry_t));
        memcpy(buf_ptr + comp_header_size + input_size * sizeof(list_entry_t),
               child_vec.GetData(),
               alloc_buf_size - comp_header_size -
                   input_size * sizeof(list_entry_t));
    }

    void WriteExistingAndNewData(uint8_t *existing_buf_ptr,
                                 CompressionHeader &existing_comp_header,
                                 idx_t existing_count) override
    {
        size_t comp_header_size = comp_header.GetSizeWoBitSet();
        memcpy(buf_ptr, &comp_header, comp_header_size);

        size_t string_t_offset = comp_header_size;
        size_t string_data_offset =
            comp_header_size +
            (existing_count + input.size()) * sizeof(string_t);
        size_t accumulated_string_len = 0;

        // Copy existing strings
        string_t *existing_strings =
            (string_t *)(existing_buf_ptr +
                         existing_comp_header.GetSizeWoBitSet());
        for (idx_t i = 0; i < existing_count; i++) {
            string_t &str = existing_strings[i];
            if (str.IsInlined()) {
                memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
            }
            else {
                memcpy(buf_ptr + string_data_offset + accumulated_string_len,
                       str.GetDataUnsafe(), str.GetSize());
                string_t new_str(
                    reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                             accumulated_string_len),
                    str.GetSize());
                memcpy(buf_ptr + string_t_offset, &new_str, sizeof(string_t));
                accumulated_string_len += str.GetSize();
            }
            string_t_offset += sizeof(string_t);
        }

        // Append new strings
        string_t *new_strings =
            (string_t *)input.data[input_chunk_idx].GetData();
        for (idx_t i = 0; i < input.size(); i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_t &str = new_strings[i];
                if (str.IsInlined()) {
                    memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
                }
                else {
                    memcpy(
                        buf_ptr + string_data_offset + accumulated_string_len,
                        str.GetDataUnsafe(), str.GetSize());
                    string_t new_str(
                        reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                                 accumulated_string_len),
                        str.GetSize());
                    memcpy(buf_ptr + string_t_offset, &new_str,
                           sizeof(string_t));
                    accumulated_string_len += str.GetSize();
                }
            }
            string_t_offset += sizeof(string_t);
        }
    }
};

class AdjListChunkWriter : public ChunkWriter {
   public:
    using ChunkWriter::ChunkWriter;

    void WriteData() override
    {
        memcpy(buf_ptr, &comp_header, comp_header_size);
        memcpy(buf_ptr + comp_header_size,
               input.data[input_chunk_idx].GetData(),
               alloc_buf_size - comp_header_size);
    }

    void WriteExistingAndNewData(uint8_t *existing_buf_ptr,
                                 CompressionHeader &existing_comp_header,
                                 idx_t existing_count) override
    {
        size_t comp_header_size = comp_header.GetSizeWoBitSet();
        memcpy(buf_ptr, &comp_header, comp_header_size);

        size_t string_t_offset = comp_header_size;
        size_t string_data_offset =
            comp_header_size +
            (existing_count + input.size()) * sizeof(string_t);
        size_t accumulated_string_len = 0;

        // Copy existing strings
        string_t *existing_strings =
            (string_t *)(existing_buf_ptr +
                         existing_comp_header.GetSizeWoBitSet());
        for (idx_t i = 0; i < existing_count; i++) {
            string_t &str = existing_strings[i];
            if (str.IsInlined()) {
                memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
            }
            else {
                memcpy(buf_ptr + string_data_offset + accumulated_string_len,
                       str.GetDataUnsafe(), str.GetSize());
                string_t new_str(
                    reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                             accumulated_string_len),
                    str.GetSize());
                memcpy(buf_ptr + string_t_offset, &new_str, sizeof(string_t));
                accumulated_string_len += str.GetSize();
            }
            string_t_offset += sizeof(string_t);
        }

        // Append new strings
        string_t *new_strings =
            (string_t *)input.data[input_chunk_idx].GetData();
        for (idx_t i = 0; i < input.size(); i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_t &str = new_strings[i];
                if (str.IsInlined()) {
                    memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
                }
                else {
                    memcpy(
                        buf_ptr + string_data_offset + accumulated_string_len,
                        str.GetDataUnsafe(), str.GetSize());
                    string_t new_str(
                        reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                                 accumulated_string_len),
                        str.GetSize());
                    memcpy(buf_ptr + string_t_offset, &new_str,
                           sizeof(string_t));
                    accumulated_string_len += str.GetSize();
                }
            }
            string_t_offset += sizeof(string_t);
        }
    }
};

class PrimitiveChunkWriter : public ChunkWriter {
   public:
    using ChunkWriter::ChunkWriter;

    void WriteData() override
    {
        size_t input_size = input.size();
        PhysicalType p_type = input.GetTypes()[input_chunk_idx].InternalType();

        // TODO: create minmaxarray

        memcpy(buf_ptr, &comp_header, comp_header_size);
        memcpy(buf_ptr + comp_header_size,
               input.data[input_chunk_idx].GetData(),
               input_size * GetTypeIdSize(p_type));
    }

    void WriteExistingAndNewData(uint8_t *existing_buf_ptr,
                                 CompressionHeader &existing_comp_header,
                                 idx_t existing_count) override
    {
        size_t comp_header_size = comp_header.GetSizeWoBitSet();
        memcpy(buf_ptr, &comp_header, comp_header_size);

        size_t string_t_offset = comp_header_size;
        size_t string_data_offset =
            comp_header_size +
            (existing_count + input.size()) * sizeof(string_t);
        size_t accumulated_string_len = 0;

        // Copy existing strings
        string_t *existing_strings =
            (string_t *)(existing_buf_ptr +
                         existing_comp_header.GetSizeWoBitSet());
        for (idx_t i = 0; i < existing_count; i++) {
            string_t &str = existing_strings[i];
            if (str.IsInlined()) {
                memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
            }
            else {
                memcpy(buf_ptr + string_data_offset + accumulated_string_len,
                       str.GetDataUnsafe(), str.GetSize());
                string_t new_str(
                    reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                             accumulated_string_len),
                    str.GetSize());
                memcpy(buf_ptr + string_t_offset, &new_str, sizeof(string_t));
                accumulated_string_len += str.GetSize();
            }
            string_t_offset += sizeof(string_t);
        }

        // Append new strings
        string_t *new_strings =
            (string_t *)input.data[input_chunk_idx].GetData();
        for (idx_t i = 0; i < input.size(); i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_t &str = new_strings[i];
                if (str.IsInlined()) {
                    memcpy(buf_ptr + string_t_offset, &str, sizeof(string_t));
                }
                else {
                    memcpy(
                        buf_ptr + string_data_offset + accumulated_string_len,
                        str.GetDataUnsafe(), str.GetSize());
                    string_t new_str(
                        reinterpret_cast<char *>(buf_ptr + string_data_offset +
                                                 accumulated_string_len),
                        str.GetSize());
                    memcpy(buf_ptr + string_t_offset, &new_str,
                           sizeof(string_t));
                    accumulated_string_len += str.GetSize();
                }
            }
            string_t_offset += sizeof(string_t);
        }
    }
};

void ExtentManager::_AppendChunkToExtentWithCompression(
    ClientContext &context, DataChunk &input, Catalog &cat_instance,
    ExtentCatalogEntry &extent_cat_entry, PartitionID pid, ExtentID new_eid)
{
    // Reaquire partition, property schema catalog entry
    auto ps_oid = extent_cat_entry.ps_oid;
    auto &prop_schema_cat_entry =
        *((PropertySchemaCatalogEntry *)cat_instance.GetEntry(
            context, DEFAULT_SCHEMA, ps_oid));
    auto partition_oid = prop_schema_cat_entry.partition_oid;
    auto &part_cat_entry = *((PartitionCatalogEntry *)cat_instance.GetEntry(
        context, DEFAULT_SCHEMA, partition_oid));
    auto &property_keys = *prop_schema_cat_entry.GetPropKeyIDs();

    // Actual run
    idx_t input_chunk_idx = 0;
    ChunkDefinitionID cdf_id_base = new_eid;
    cdf_id_base = cdf_id_base << 32;
    for (auto &l_type : input.GetTypes()) {
        auto prop_key_id = property_keys[input_chunk_idx];
        // Get Physical Type
        PhysicalType p_type = l_type.InternalType();
        // For each Vector in DataChunk create new chunk definition
        LocalChunkDefinitionID chunk_definition_idx;
        if (l_type == LogicalType::FORWARD_ADJLIST ||
            l_type == LogicalType::BACKWARD_ADJLIST) {
            chunk_definition_idx =
                extent_cat_entry.GetNextAdjListChunkDefinitionID();
        }
        else {
            chunk_definition_idx = extent_cat_entry.GetNextChunkDefinitionID();
        }
        ChunkDefinitionID cdf_id = cdf_id_base + chunk_definition_idx;
        string chunkdefinition_name =
            DEFAULT_CHUNKDEFINITION_PREFIX + std::to_string(cdf_id);
        CreateChunkDefinitionInfo chunkdefinition_info(
            DEFAULT_SCHEMA, chunkdefinition_name, l_type);
        ChunkDefinitionCatalogEntry *chunkdefinition_cat =
            (ChunkDefinitionCatalogEntry *)cat_instance.CreateChunkDefinition(
                context, &chunkdefinition_info);
        if (l_type == LogicalType::FORWARD_ADJLIST ||
            l_type == LogicalType::BACKWARD_ADJLIST) {
            extent_cat_entry.AddAdjListChunkDefinitionID(cdf_id);
        }
        else {
            extent_cat_entry.AddChunkDefinitionID(cdf_id);
        }
        chunkdefinition_cat->SetNumEntriesInColumn(input.size());

        // Analyze compression to find best compression method
        CompressionFunctionType best_compression_function =
            UNCOMPRESSED;  // TODO: Implement compression

        // Create Compressionheader, based on nullity
        CompressionHeader comp_header(best_compression_function, input.size(),
                                      SwizzlingType::SWIZZLE_NONE);
        SetupCompressionHeader(comp_header, input, input_chunk_idx);

        // Calculate buffer size and allocate
        size_t alloc_buf_size = 0;
        size_t bitmap_size = 0;
        uint8_t *buf_ptr = nullptr;
        size_t buf_size = 0;
        CalculateBufferSize(l_type, input, input_chunk_idx, comp_header,
                            alloc_buf_size, bitmap_size);

        // Get Buffer from Cache Manager. Cache Object ID: 64bit = ChunkDefinitionID
        string file_path_prefix =
            DiskAioParameters::WORKSPACE + "/part_" + std::to_string(pid) +
            "/ext_" + std::to_string(new_eid) + std::string("/chunk_");
        ChunkCacheManager::ccm->CreateSegment(
            cdf_id, file_path_prefix, alloc_buf_size + bitmap_size, false);
        ChunkCacheManager::ccm->PinSegment(cdf_id, file_path_prefix, &buf_ptr,
                                           &buf_size, false, true);
        std::memset(buf_ptr, 0, buf_size);

        // Write data using appropriate writer
        unique_ptr<ChunkWriter> writer;
        if (l_type.id() == LogicalTypeId::VARCHAR) {
            writer = make_unique<VarCharChunkWriter>(
                buf_ptr, comp_header, input, input_chunk_idx, alloc_buf_size);
        }
        else if (l_type.id() == LogicalTypeId::LIST) {
            writer = make_unique<ListChunkWriter>(
                buf_ptr, comp_header, input, input_chunk_idx, alloc_buf_size);
        }
        else if (l_type.id() == LogicalTypeId::FORWARD_ADJLIST ||
                 l_type.id() == LogicalTypeId::BACKWARD_ADJLIST) {
            writer = make_unique<AdjListChunkWriter>(
                buf_ptr, comp_header, input, input_chunk_idx, alloc_buf_size);
        }
        else {
            writer = make_unique<PrimitiveChunkWriter>(
                buf_ptr, comp_header, input, input_chunk_idx, alloc_buf_size);
            if (input.GetTypes()[input_chunk_idx] == LogicalType::UBIGINT ||
                input.GetTypes()[input_chunk_idx] == LogicalType::ID ||
                input.GetTypes()[input_chunk_idx] == LogicalType::BIGINT) {
                chunkdefinition_cat->CreateMinMaxArray(
                    input.data[input_chunk_idx], input.size());
            }
        }

        writer->WriteData();

        // Handle null mask if needed
        WriteNullMask(comp_header, input, input_chunk_idx, buf_ptr,
                      alloc_buf_size);

        // Set Dirty & Unpin Segment & Flush
        ChunkCacheManager::ccm->SetDirty(cdf_id);
        ChunkCacheManager::ccm->UnPinSegment(cdf_id);
        input_chunk_idx++;
    }
}

void ExtentManager::_AppendTuplesToExtentWithCompression(
    ClientContext &context, DataChunk &input, Catalog &cat_instance,
    ExtentCatalogEntry &extent_cat_entry, PartitionID pid, ExtentID eid)
{
    // Reaquire partition, property schema catalog entry
    auto ps_oid = extent_cat_entry.ps_oid;
    auto &prop_schema_cat_entry =
        *((PropertySchemaCatalogEntry *)cat_instance.GetEntry(
            context, DEFAULT_SCHEMA, ps_oid));
    auto partition_oid = prop_schema_cat_entry.partition_oid;
    auto &part_cat_entry = *((PartitionCatalogEntry *)cat_instance.GetEntry(
        context, DEFAULT_SCHEMA, partition_oid));
    auto &property_keys = *prop_schema_cat_entry.GetPropKeyIDs();
    idx_t input_chunk_idx = 0;
    auto &chunk_definition_ids = extent_cat_entry.chunks;

    D_ASSERT(input.ColumnCount() == property_keys.size());

    // Calculate how many rows can fit in current extent
    ChunkDefinitionID first_cdf_id = chunk_definition_ids[0];
    ChunkDefinitionCatalogEntry *first_chunkdef_cat =
        (ChunkDefinitionCatalogEntry *)cat_instance.GetEntry(
            context, CatalogType::CHUNKDEFINITION_ENTRY, DEFAULT_SCHEMA,
            DEFAULT_CHUNKDEFINITION_PREFIX + std::to_string(first_cdf_id));

    uint8_t *existing_buf_ptr = nullptr;
    size_t existing_buf_size = 0;
    string file_path_prefix = DiskAioParameters::WORKSPACE + "/part_" +
                              std::to_string(pid) + "/ext_" +
                              std::to_string(eid) + std::string("/chunk_");
    ChunkCacheManager::ccm->PinSegment(first_cdf_id, file_path_prefix,
                                       &existing_buf_ptr, &existing_buf_size,
                                       false, true);

    CompressionHeader existing_comp_header;
    memcpy(&existing_comp_header, existing_buf_ptr,
           existing_comp_header.GetSizeWoBitSet());

    idx_t existing_count = existing_comp_header.data_len;
    idx_t rows_that_fit = STORAGE_STANDARD_VECTOR_SIZE - existing_count;
    idx_t rows_to_append = std::min(input.size(), rows_that_fit);

    ChunkCacheManager::ccm->UnPinSegment(first_cdf_id);

    if (rows_to_append == 0) {
        // Current extent is full, create new extent for remaining data
        DataChunk remaining_chunk;
        remaining_chunk.Initialize(input.GetTypes());
        remaining_chunk.Reference(input);
        CreateExtent(context, remaining_chunk, part_cat_entry,
                     prop_schema_cat_entry);
        return;
    }

    // If we need to split the input, create a temporary chunk with just the rows that fit
    unique_ptr<DataChunk> chunk_to_append;
    if (rows_to_append < input.size()) {
        chunk_to_append = make_unique<DataChunk>();
        chunk_to_append->Initialize(input.GetTypes());
        for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
            chunk_to_append->data[col_idx].Slice(input.data[col_idx], 0);
        }
        chunk_to_append->SetCardinality(rows_to_append);
    }
    else {
        chunk_to_append = make_unique<DataChunk>();
        chunk_to_append->Initialize(input.GetTypes());
        chunk_to_append->Reference(input);
    }

    // Create a temporary buffer to store existing data
    vector<unique_ptr<uint8_t[]>> temp_buffers;
    vector<size_t> temp_buffer_sizes;
    
    // First pass: Pin all segments and copy to temp buffers
    for (auto &cdf_id : chunk_definition_ids) {
        uint8_t *existing_buf_ptr = nullptr;
        size_t existing_buf_size = 0;
        
        ChunkCacheManager::ccm->PinSegment(cdf_id, file_path_prefix,
                                          &existing_buf_ptr, &existing_buf_size,
                                          false, true);
                                          
        // Create temp buffer and copy data
        auto temp_buffer = make_unique<uint8_t[]>(existing_buf_size);
        memcpy(temp_buffer.get(), existing_buf_ptr, existing_buf_size);
        temp_buffers.push_back(move(temp_buffer));
        temp_buffer_sizes.push_back(existing_buf_size);
        
        ChunkCacheManager::ccm->UnPinSegment(cdf_id);
    }

    // Second pass: Process each column
    input_chunk_idx = 0;
    for (auto &l_type : chunk_to_append->GetTypes()) {
        auto prop_key_id = property_keys[input_chunk_idx];
        ChunkDefinitionID cdf_id = chunk_definition_ids[input_chunk_idx];
        
        // Get existing chunk definition
        ChunkDefinitionCatalogEntry *chunkdefinition_cat =
            (ChunkDefinitionCatalogEntry *)cat_instance.GetEntry(
                context, CatalogType::CHUNKDEFINITION_ENTRY, DEFAULT_SCHEMA,
                DEFAULT_CHUNKDEFINITION_PREFIX + std::to_string(cdf_id));

        // Create new compression header and calculate buffer sizes
        CompressionHeader new_comp_header(UNCOMPRESSED, rows_to_append,
                                        existing_comp_header.swizzle_type);
        if (existing_comp_header.HasNullMask() ||
            FlatVector::HasNull(input.data[input_chunk_idx])) {
            new_comp_header.SetNullMask();
        }

        // Create temporary DataChunk that combines existing and new data
        DataChunk combined_chunk;
        combined_chunk.Initialize(chunk_to_append->GetTypes(),
                                  STORAGE_STANDARD_VECTOR_SIZE);
        combined_chunk.SetCardinality(rows_to_append);

        // Calculate buffer size considering both existing and new data
        size_t alloc_buf_size = 0;
        size_t bitmap_size = 0;
        CalculateBufferSize(l_type, combined_chunk, input_chunk_idx,
                            new_comp_header, alloc_buf_size, bitmap_size);

        // Now create new segment (this will delete the existing file)
        ChunkCacheManager::ccm->CreateSegment(
            cdf_id, file_path_prefix, alloc_buf_size + bitmap_size, true);
            
        uint8_t *new_buf_ptr = nullptr;
        size_t buf_size = 0;
        ChunkCacheManager::ccm->PinSegment(
            cdf_id, file_path_prefix, &new_buf_ptr, &buf_size, false, true);
        std::memset(new_buf_ptr, 0, buf_size);

        // Use the temp buffer instead of the original buffer for writing
        unique_ptr<ChunkWriter> writer;
        if (l_type.id() == LogicalTypeId::VARCHAR) {
            writer = make_unique<VarCharChunkWriter>(
                new_buf_ptr, new_comp_header, *chunk_to_append, input_chunk_idx,
                alloc_buf_size);
        }
        else if (l_type.id() == LogicalTypeId::LIST) {
            writer = make_unique<ListChunkWriter>(
                new_buf_ptr, new_comp_header, *chunk_to_append, input_chunk_idx,
                alloc_buf_size);
        }
        else if (l_type.id() == LogicalTypeId::FORWARD_ADJLIST ||
                 l_type.id() == LogicalTypeId::BACKWARD_ADJLIST) {
            writer = make_unique<AdjListChunkWriter>(
                new_buf_ptr, new_comp_header, *chunk_to_append, input_chunk_idx,
                alloc_buf_size);
        }
        else {
            writer = make_unique<PrimitiveChunkWriter>(
                new_buf_ptr, new_comp_header, *chunk_to_append, input_chunk_idx,
                alloc_buf_size);
        }

        // Write data using temp buffer
        writer->WriteExistingAndNewData(temp_buffers[input_chunk_idx].get(),
                                      existing_comp_header,
                                      existing_count);

        // Handle null mask if needed
        if (new_comp_header.HasNullMask()) {
            // Copy existing null mask
            if (existing_comp_header.HasNullMask()) {
                memcpy(new_buf_ptr + new_comp_header.GetNullBitmapOffset(),
                       temp_buffers[input_chunk_idx].get() +
                           existing_comp_header.GetNullBitmapOffset(),
                       (existing_count + 7) / 8);
            }

            // Append new null mask
            if (FlatVector::HasNull(input.data[input_chunk_idx])) {
                memcpy(
                    new_buf_ptr + new_comp_header.GetNullBitmapOffset() +
                        (existing_count + 7) / 8,
                    FlatVector::Validity(input.data[input_chunk_idx]).GetData(),
                    (rows_to_append + 7) / 8);
            }
        }

        // Update chunk definition entry and cleanup
        chunkdefinition_cat->SetNumEntriesInColumn(rows_to_append);
        ChunkCacheManager::ccm->UnPinSegment(cdf_id);
        input_chunk_idx++;
    }

    // If there are remaining rows, create new extent for them
    if (rows_to_append < input.size()) {
        DataChunk remaining_chunk;
        remaining_chunk.Initialize(input.GetTypes());
        remaining_chunk.SetCardinality(input.size() - rows_to_append);
        CreateExtent(context, remaining_chunk, part_cat_entry,
                     prop_schema_cat_entry);
    }
}

void ExtentManager::_UpdatePartitionMinMaxArray(
    ClientContext &context, Catalog &cat_instance,
    PartitionCatalogEntry &part_cat, PropertySchemaCatalogEntry &ps_cat,
    ExtentCatalogEntry &extent_cat_entry)
{
    auto &property_keys = *ps_cat.GetPropKeyIDs();
    auto &chunkdef_ids = extent_cat_entry.chunks;
    for (int i = 0; i < property_keys.size(); i++) {
        auto property_key_id = property_keys[i];
        auto chunkdef_id = chunkdef_ids[i];
        auto chunkdef_cat_entry =
            (ChunkDefinitionCatalogEntry *)cat_instance.GetEntry(
                context, CatalogType::CHUNKDEFINITION_ENTRY, DEFAULT_SCHEMA,
                DEFAULT_CHUNKDEFINITION_PREFIX + std::to_string(chunkdef_id));

        if (chunkdef_cat_entry->IsMinMaxArrayExist()) {
            vector<minmax_t> minmax =
                move(chunkdef_cat_entry->GetMinMaxArray());
            for (auto &minmax_pair : minmax) {
                part_cat.UpdateMinMaxArray(property_key_id, minmax_pair.min,
                                           minmax_pair.max);
            }
        }
    }
}

void ExtentManager::_UpdatePartitionMinMaxArray(
    PartitionCatalogEntry &part_cat, PropertyKeyID prop_key_id,
    ChunkDefinitionCatalogEntry &chunkdef_cat_entry)
{
    if (chunkdef_cat_entry.IsMinMaxArrayExist()) {
        vector<minmax_t> minmax = move(chunkdef_cat_entry.GetMinMaxArray());
        for (auto &minmax_pair : minmax) {
            part_cat.UpdateMinMaxArray(prop_key_id, minmax_pair.min,
                                       minmax_pair.max);
        }
    }
}

// Helper functions
void ExtentManager::SetupCompressionHeader(CompressionHeader &comp_header,
                                           DataChunk &input,
                                           idx_t input_chunk_idx)
{
    if (FlatVector::HasNull(input.data[input_chunk_idx])) {
        if (input.size() != FlatVector::Validity(input.data[input_chunk_idx])
                                .CountValid(input.size())) {
            comp_header.SetNullMask();
        }
    }
}

void ExtentManager::WriteNullMask(CompressionHeader &comp_header,
                                  DataChunk &input, idx_t input_chunk_idx,
                                  uint8_t *buf_ptr, size_t alloc_buf_size)
{
    if (comp_header.HasNullMask()) {
        size_t bitmap_size = (input.size() + 7) / 8;
        auto *validity_data =
            (char *)(FlatVector::Validity(input.data[input_chunk_idx])
                         .GetData());
        memcpy(buf_ptr + alloc_buf_size, validity_data, bitmap_size);
    }
}

void ExtentManager::CalculateBufferSize(LogicalType &l_type, DataChunk &input,
                                        idx_t input_chunk_idx,
                                        CompressionHeader &comp_header,
                                        size_t &alloc_buf_size,
                                        size_t &bitmap_size)
{
    alloc_buf_size = comp_header.GetSizeWoBitSet();
    bitmap_size = 0;
    PhysicalType p_type = l_type.InternalType();
    const size_t slot_for_num_adj = 1;

    if (l_type.id() == LogicalTypeId::VARCHAR) {
        // For VARCHAR: header + array of string_t + actual string data
        size_t string_len_total = 0;
        string_t *string_buffer =
            (string_t *)input.data[input_chunk_idx].GetData();

        for (idx_t i = 0; i < input.size(); i++) {
            if (!FlatVector::IsNull(input.data[input_chunk_idx], i)) {
                string_len_total += string_buffer[i].IsInlined()
                                        ? 0
                                        : string_buffer[i].GetSize();
            }
        }

        // string_t array + string data + null mask (?)
        alloc_buf_size +=
            (input.size() * sizeof(string_t)) + string_len_total + 512;
    }
    else if (l_type.id() == LogicalTypeId::LIST) {
        // For LIST: header + list entries + child vector data
        // TODO: support nested LIST
        size_t list_len_total = 0;
        size_t child_type_size =
            GetTypeIdSize(ListType::GetChildType(l_type).InternalType());
        list_entry_t *list_buffer =
            (list_entry_t *)input.data[input_chunk_idx].GetData();
        for (size_t i = 0; i < input.size();
             i++) {  // Accumulate the length of all child datas
            list_len_total += list_buffer[i].length;
        }
        list_len_total *= child_type_size;
        list_len_total += input.size() * sizeof(list_entry_t);
        alloc_buf_size += list_len_total;
    }
    else if (l_type.id() == LogicalTypeId::FORWARD_ADJLIST ||
             l_type.id() == LogicalTypeId::BACKWARD_ADJLIST) {
        // For adjacency lists: header + actual data
        idx_t *adj_list_buffer = (idx_t *)input.data[input_chunk_idx].GetData();
        size_t num_adj_lists = adj_list_buffer[0];
        size_t adj_list_size =
            num_adj_lists == 0 ? 0 : adj_list_buffer[num_adj_lists];
        alloc_buf_size += sizeof(idx_t) * (slot_for_num_adj + adj_list_size);
    }
    else if (TypeIsConstantSize(p_type)) {
        // For primitive types: header + data
        alloc_buf_size += input.size() * GetTypeIdSize(p_type);
    }
    else {
        throw NotImplementedException(
            "Unsupported type for buffer size calculation");
    }

    // Add space for null bitmap if needed
    if (comp_header.HasNullMask()) {
        bitmap_size = (input.size() + 7) / 8;
        comp_header.SetNullBitmapOffset(alloc_buf_size);
    }
}

}  // namespace duckdb