//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/perfect_hash_join_executor.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common/row_operations/row_operations.hpp"
#include "execution/execution_context.hpp"
#include "execution/join_hashtable.hpp"
#include "execution/physical_operator/physical_operator.hpp"

namespace duckdb {

class PhysicalHashJoinState;
class HashJoinGlobalState;
class PhysicalHashJoin;

struct PerfectHashJoinStats {
	// The max size our build must have to run the perfect HJ
	static const int64_t MAX_BUILD_SIZE = 1000000;

	int64_t build_min;
	int64_t build_max;
	int64_t build_range = 0;
	// if is_build_small is false, min/max/range may be invalid
	bool is_build_small = false;

	bool is_physical_id = false;
	// if is_physical_id is true, following is used
	uint16_t partition_min = std::numeric_limits<uint16_t>::max();
	uint16_t partition_max = std::numeric_limits<uint16_t>::min();
	uint16_t partition_range;
	uint16_t extent_min = std::numeric_limits<uint16_t>::max();
	uint16_t extent_max = std::numeric_limits<uint16_t>::min();
	uint16_t extent_range;
	uint32_t tuple_min = std::numeric_limits<uint32_t>::max();
	uint32_t tuple_max = std::numeric_limits<uint32_t>::min();
	uint32_t tuple_range;
};

//! PhysicalHashJoin represents a hash loop join between two tables
class PerfectHashJoinExecutor {
	using PerfectHashTable = std::vector<Vector>;

public:
	explicit PerfectHashJoinExecutor(const PhysicalHashJoin &join, PerfectHashJoinStats pjoin_stats);

public:
	bool CanDoPerfectHashJoin();

	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context);
	OperatorResultType ProbePerfectHashTable(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                         OperatorState &state, unique_ptr<JoinHashTable> &ht);
	bool BuildPerfectHashTable(LogicalType &type, unique_ptr<JoinHashTable> &ht);

private:
	std::tuple<uint16_t, uint16_t, uint32_t> DestructPhysicalId(uint64_t physical_id);
	uint64_t GetPhysicalIdHash(uint16_t partition_id, uint16_t extent_id, uint32_t tuple_id);
	uint64_t GetPhysicalIdHash(uint64_t physical_id);

	void FillJoinStatForPhysicalId(Vector &source, idx_t count);

	void FillSelectionVectorSwitchProbe(Vector &source, SelectionVector &build_sel_vec, SelectionVector &probe_sel_vec,
	                                    idx_t count, idx_t &probe_sel_count);
	template <typename T>
	void TemplatedFillSelectionVectorProbe(Vector &source, SelectionVector &build_sel_vec,
	                                       SelectionVector &probe_sel_vec, idx_t count, idx_t &prob_sel_count);

	bool FillSelectionVectorSwitchBuild(Vector &source, SelectionVector &sel_vec, SelectionVector &seq_sel_vec,
	                                    idx_t count);
	template <typename T>
	bool TemplatedFillSelectionVectorBuild(Vector &source, SelectionVector &sel_vec, SelectionVector &seq_sel_vec,
	                                       idx_t count);

private:
	const PhysicalHashJoin &join;
	//! Columnar perfect hash table
	PerfectHashTable perfect_hash_table;
	//! Build and probe statistics
	PerfectHashJoinStats perfect_join_statistics;
	//! Stores the occurences of each value in the build side
	unique_ptr<bool[]> bitmap_build_idx;
	//! Stores the number of unique keys in the build side
	idx_t unique_keys = 0;
};

} // namespace duckdb
