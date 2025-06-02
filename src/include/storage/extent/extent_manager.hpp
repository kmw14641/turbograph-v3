#ifndef EXTENT_MANAGER_H
#define EXTENT_MANAGER_H

#include "common/common.hpp"
#include "common/vector.hpp"

namespace duckdb {

class Catalog;
class DataChunk;
class ClientContext;
class CompressionHeader;
class ExtentCatalogEntry;
class PartitionCatalogEntry;
class PropertySchemaCatalogEntry;
class ChunkDefinitionCatalogEntry;

class ExtentManager {

public:
    ExtentManager();
    ~ExtentManager() {}

    // for bulk loading
    ExtentID CreateExtent(ClientContext &context, DataChunk &input,
                          PartitionCatalogEntry &part_cat,
                          PropertySchemaCatalogEntry &ps_cat);
    void CreateExtent(ClientContext &context, DataChunk &input,
                      PartitionCatalogEntry &part_cat,
                      PropertySchemaCatalogEntry &ps_cat, ExtentID new_eid);
    void AppendChunkToExistingExtent(ClientContext &context, DataChunk &input,
                                     ExtentID eid);

    // for bulk update
    void AppendTuplesToExistingExtent(ClientContext &context, DataChunk &input,
                                      ExtentID eid);

    // Add Index
    void AddIndex(ClientContext &context, DataChunk &input) {}

   private:
    void _AppendChunkToExtentWithCompression(
        ClientContext &context, DataChunk &input, Catalog &cat_instance,
        ExtentCatalogEntry &extent_cat_entry, PartitionID pid, ExtentID eid);
    void _AppendTuplesToExtentWithCompression(
        ClientContext &context, DataChunk &input, Catalog &cat_instance,
        ExtentCatalogEntry &extent_cat_entry, PartitionID pid, ExtentID eid);
    void _UpdatePartitionMinMaxArray(ClientContext &context,
                                     Catalog &cat_instance,
                                     PartitionCatalogEntry &part_cat,
                                     PropertySchemaCatalogEntry &ps_cat,
                                     ExtentCatalogEntry &extent_cat_entry);
    void _UpdatePartitionMinMaxArray(
        PartitionCatalogEntry &part_cat, PropertyKeyID prop_key_id,
        ChunkDefinitionCatalogEntry &chunkdef_cat_entry);

    void SetupCompressionHeader(CompressionHeader &comp_header,
                                DataChunk &input, idx_t input_chunk_idx);

    void CalculateBufferSize(LogicalType &l_type, DataChunk &input,
                             idx_t input_chunk_idx,
                             CompressionHeader &comp_header,
                             size_t &alloc_buf_size, size_t &bitmap_size);

    void CalculateBufferSizeForAppend(
        LogicalType &l_type, DataChunk &input, idx_t input_chunk_idx,
        CompressionHeader &comp_header, idx_t existing_count,
        uint8_t *existing_buf, size_t existing_buf_size, size_t &alloc_buf_size,
        size_t &bitmap_size);

    void WriteNullMask(CompressionHeader &comp_header,
                       DataChunk &input, idx_t input_chunk_idx,
                       uint8_t *buf_ptr, size_t alloc_buf_size);
};

} // namespace duckdb

#endif // EXTENT_MANAGER_H
