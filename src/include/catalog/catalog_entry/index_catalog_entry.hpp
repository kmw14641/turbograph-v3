//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/catalog/catalog_entry/index_catalog_entry.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "catalog/standard_entry.hpp"
#include "parser/parsed_data/create_index_info.hpp"
#include "common/boost_typedefs.hpp"

namespace duckdb {

// struct DataTableInfo;
class Index;

//! An index catalog entry
class IndexCatalogEntry : public StandardEntry {
public:
	//! Create a real TableCatalogEntry and initialize storage for it
	IndexCatalogEntry(Catalog *catalog, SchemaCatalogEntry *schema, CreateIndexInfo *info, const void_allocator &void_alloc);
	~IndexCatalogEntry() override;

	//! The index type
	IndexType index_type;
	
	//! True for (1-1) or (n-1) relationship, False for (1-n) or (n-n) relationship
	//! Used to determine the join type in adjidx join
	uint8_t is_target_unique;

	//! OID of the partition to which this index belongs
	idx_t pid;

	//! OID of the segment to which this index belongs (temporary)
	idx_t psid;
	
	//! Index of src/tgt column in the extent (e.g., _sid, _tid)
	int64_t_vector index_key_columns;

	//! Index of this adjacency column in the extent
	idx_t adj_col_idx;

public:
	string ToSQL() override;
    idx_t GetPartitionID();
    idx_t GetPropertySchemaID();
    int64_t_vector *GetIndexKeyColumns();
    IndexType GetIndexType();
    idx_t GetAdjColIdx();

    void SetIsTargetUnique(bool is_target_unique_)
    {
        is_target_unique = is_target_unique_;
    }
};

} // namespace duckdb
