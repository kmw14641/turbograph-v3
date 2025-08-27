#include "execution/physical_operator/perfect_hash_join_executor.hpp"

#include "common/types/row_layout.hpp"
#include "execution/physical_operator/physical_hash_join.hpp"

namespace duckdb {

PerfectHashJoinExecutor::PerfectHashJoinExecutor(const PhysicalHashJoin &join_p,
                                                 PerfectHashJoinStats perfect_join_stats)
    : join(join_p), perfect_join_statistics(move(perfect_join_stats)) {
}

bool PerfectHashJoinExecutor::CanDoPerfectHashJoin() {
	return perfect_join_statistics.is_build_small;
}

//===--------------------------------------------------------------------===//
// Build
//===--------------------------------------------------------------------===//
bool PerfectHashJoinExecutor::BuildPerfectHashTable(LogicalType &key_type, unique_ptr<JoinHashTable> &ht) {
	// Fill columns with build data
	Vector tuples_addresses(LogicalType::POINTER, ht->Count());              // allocate space for all the tuples
	auto key_locations = FlatVector::GetData<data_ptr_t>(tuples_addresses); // get a pointer to vector data
	// TODO: In a parallel finalize: One should exclusively lock and each thread should do one part of the code below.
	// Go through all the blocks and fill the keys addresses
	JoinHTScanState state;
	auto keys_count = ht->FillWithHTOffsets(key_locations, state);
	// Scan the build keys in the hash table
	Vector build_vector(key_type, keys_count);
	RowOperations::FullScanColumn(ht->layout, tuples_addresses, build_vector, keys_count, 0);

	if (perfect_join_statistics.is_physical_id) {
		FillJoinStatForPhysicalId(build_vector, keys_count);
		if (!perfect_join_statistics.is_build_small) {
			return false;
		}
	}

	// Allocate memory for each build column
	auto build_size = perfect_join_statistics.build_range + 1;
	for (const auto &type : ht->build_types) {
		perfect_hash_table.emplace_back(type, build_size);
	}
	// and for duplicate_checking
	bitmap_build_idx = unique_ptr<bool[]>(new bool[build_size]);
	memset(bitmap_build_idx.get(), 0, sizeof(bool) * build_size); // set false

	// Fill the selection vector using the build keys and create a sequential vector
	// todo: add check for fast pass when probe is part of build domain
	SelectionVector sel_build(keys_count + 1);
	SelectionVector sel_tuples(keys_count + 1);
	bool success = FillSelectionVectorSwitchBuild(build_vector, sel_build, sel_tuples, keys_count);
	// early out
	if (!success) {
		return false;
	}
	keys_count = unique_keys; // do not consider keys out of the range
	// Full scan the remaining build columns and fill the perfect hash table
	for (idx_t i = 0; i < ht->build_types.size(); i++) {
		auto build_size = perfect_join_statistics.build_range + 1;
		auto &vector = perfect_hash_table[i];
		D_ASSERT(vector.GetType() == ht->build_types[i]);
		const auto col_no = ht->condition_types.size() + i;
		const auto col_offset = ht->layout.GetOffsets()[col_no];
		RowOperations::Gather(tuples_addresses, sel_tuples, vector, sel_build, keys_count, col_offset, col_no,
		                      build_size);
	}
	return true;
}

std::tuple<uint16_t, uint16_t, uint32_t> PerfectHashJoinExecutor::DestructPhysicalId(uint64_t physical_id) {
	// TODO: safe casting? although it works
	uint16_t partition_id = physical_id >> 48;
	uint16_t extent_id = physical_id >> 32;
	uint32_t tuple_id = physical_id;
	return {partition_id, extent_id, tuple_id};
}

uint64_t PerfectHashJoinExecutor::GetPhysicalIdHash(uint16_t partition_id, uint16_t extent_id, uint32_t tuple_id) {
	if (partition_id > perfect_join_statistics.partition_max ||
		extent_id > perfect_join_statistics.extent_max ||
		tuple_id > perfect_join_statistics.tuple_max) {
		return std::numeric_limits<uint64_t>::max();  // return max so that input_value > build_max
	}

	uint64_t partition_base = (perfect_join_statistics.extent_range + 1) * (perfect_join_statistics.tuple_range + 1);
	uint64_t extent_base = perfect_join_statistics.tuple_range + 1;

	uint16_t partition_idx = partition_id - perfect_join_statistics.partition_min;
	uint16_t extent_idx = extent_id - perfect_join_statistics.extent_min;
	uint32_t tuple_idx = tuple_id - perfect_join_statistics.tuple_min;

	return partition_idx * partition_base + extent_idx * extent_base + tuple_idx;
}

uint64_t PerfectHashJoinExecutor::GetPhysicalIdHash(uint64_t physical_id) {
	auto [partition_id, extent_id, tuple_id] = DestructPhysicalId(physical_id);
	GetPhysicalIdHash(partition_id, extent_id, tuple_id);
}

void PerfectHashJoinExecutor::FillJoinStatForPhysicalId(Vector &source, idx_t count) {
	D_ASSERT(source.GetType().InternalType() == PhysicalType::UINT64);

	VectorData vector_data;
	source.Orrify(count, vector_data);
	auto data = reinterpret_cast<uint64_t *>(vector_data.data);

	for (idx_t i = 0; i < count; ++i) {
		auto data_idx = vector_data.sel->get_index(i);
		auto input_value = data[data_idx];
		auto [partition_id, extent_id, tuple_id] = DestructPhysicalId(input_value);

		perfect_join_statistics.partition_max = std::max(perfect_join_statistics.partition_max, partition_id);
		perfect_join_statistics.partition_min = std::min(perfect_join_statistics.partition_min, partition_id);
		perfect_join_statistics.partition_range = perfect_join_statistics.partition_max - perfect_join_statistics.partition_min;
		perfect_join_statistics.extent_max = std::max(perfect_join_statistics.extent_max, extent_id);
		perfect_join_statistics.extent_min = std::min(perfect_join_statistics.extent_min, extent_id);
		perfect_join_statistics.extent_range = perfect_join_statistics.extent_max - perfect_join_statistics.extent_min;
		perfect_join_statistics.tuple_max = std::max(perfect_join_statistics.tuple_max, tuple_id);
		perfect_join_statistics.tuple_min = std::min(perfect_join_statistics.tuple_min, tuple_id);
		perfect_join_statistics.tuple_range = perfect_join_statistics.tuple_max - perfect_join_statistics.tuple_min;

		perfect_join_statistics.build_max = GetPhysicalIdHash(perfect_join_statistics.partition_max, perfect_join_statistics.extent_max, perfect_join_statistics.tuple_max);
		if (perfect_join_statistics.build_max > PerfectHashJoinStats::MAX_BUILD_SIZE) {
			perfect_join_statistics.is_build_small = false;
			return;
		}
	}

	perfect_join_statistics.build_min = 0;
	perfect_join_statistics.build_range = perfect_join_statistics.build_max;
}

bool PerfectHashJoinExecutor::FillSelectionVectorSwitchBuild(Vector &source, SelectionVector &sel_vec,
                                                             SelectionVector &seq_sel_vec, idx_t count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		return TemplatedFillSelectionVectorBuild<int8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT16:
		return TemplatedFillSelectionVectorBuild<int16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT32:
		return TemplatedFillSelectionVectorBuild<int32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT64:
		return TemplatedFillSelectionVectorBuild<int64_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT8:
		return TemplatedFillSelectionVectorBuild<uint8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT16:
		return TemplatedFillSelectionVectorBuild<uint16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT32:
		return TemplatedFillSelectionVectorBuild<uint32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT64:
		return TemplatedFillSelectionVectorBuild<uint64_t>(source, sel_vec, seq_sel_vec, count);
	default:
		throw NotImplementedException("Type not supported for perfect hash join");
	}
}

template <typename T>
bool PerfectHashJoinExecutor::TemplatedFillSelectionVectorBuild(Vector &source, SelectionVector &sel_vec,
                                                                SelectionVector &seq_sel_vec, idx_t count) {
	auto min_value = perfect_join_statistics.build_min;
	auto max_value = perfect_join_statistics.build_max;
	VectorData vector_data;
	source.Orrify(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	// generate the selection vector
	for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
		auto data_idx = vector_data.sel->get_index(i);
		auto input_value = data[data_idx];

		if (perfect_join_statistics.is_physical_id) {
			input_value = GetPhysicalIdHash(input_value);
		}

		// add index to selection vector if value in the range
		if (min_value <= input_value && input_value <= max_value) {
			auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
			sel_vec.set_index(sel_idx, idx);
			if (bitmap_build_idx[idx]) {
				return false;
			} else {
				bitmap_build_idx[idx] = true;
				unique_keys++;
			}
			seq_sel_vec.set_index(sel_idx++, i);
		}
		else {
			return false;
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Probe
//===--------------------------------------------------------------------===//
class PerfectHashJoinState : public OperatorState {
public:
	DataChunk join_keys;
	ExpressionExecutor probe_executor;
	SelectionVector build_sel_vec;
	SelectionVector probe_sel_vec;
	SelectionVector seq_sel_vec;
};

unique_ptr<OperatorState> PerfectHashJoinExecutor::GetOperatorState(ExecutionContext &context) {
	auto state = make_unique<PerfectHashJoinState>();
	state->join_keys.Initialize(join.condition_types);
	for (auto &cond : join.conditions) {
		state->probe_executor.AddExpression(*cond.left);
	}
	state->build_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
	state->probe_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
	state->seq_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
	return move(state);
}

OperatorResultType PerfectHashJoinExecutor::ProbePerfectHashTable(ExecutionContext &context, DataChunk &input,
                                                                  DataChunk &result, OperatorState &state_p, unique_ptr<JoinHashTable> &ht) {
	auto &state = (PerfectHashJoinState &)state_p;
	// keeps track of how many probe keys have a match
	idx_t probe_sel_count = 0;

	// fetch the join keys from the chunk
	state.join_keys.Reset();
	state.probe_executor.Execute(input, state.join_keys);
	// select the keys that are in the min-max range
	auto &keys_vec = state.join_keys.data[0];
	auto keys_count = state.join_keys.size();
	// todo: add check for fast pass when probe is part of build domain
	FillSelectionVectorSwitchProbe(keys_vec, state.build_sel_vec, state.probe_sel_vec, keys_count, probe_sel_count);

	// S62 - Left projection mapping
	for (idx_t i = 0; i < ht->output_left_projection_map.size(); i++) {
		if (ht->output_left_projection_map[i] !=
			std::numeric_limits<uint32_t>::max()) {
			// If all probe side matches to build side, just reference probe
			if (keys_count == probe_sel_count) {
				result.data[ht->output_left_projection_map[i]].Reference(
					input.data[i]);
			} else {
				// otherwise, filter it out the values that do not match
				result.data[ht->output_left_projection_map[i]].Slice(
					input.data[i], state.probe_sel_vec,
					probe_sel_count);
			}
		}
	}

	// on the build side, we need to fetch the data and build dictionary vectors with the sel_vec
	idx_t i = 0;
	for (idx_t projection_map_idx = 0;
			projection_map_idx < ht->output_right_projection_map.size();
			projection_map_idx++) {
		if (ht->output_right_projection_map[projection_map_idx] !=
			std::numeric_limits<uint32_t>::max()) {
			auto &result_vector = result.data[ht->output_right_projection_map[projection_map_idx]];
			D_ASSERT(result_vector.GetType() == ht->build_types[i]);
			auto &build_vec = perfect_hash_table[i];
			result_vector.Reference(build_vec);
			result_vector.Slice(state.build_sel_vec, probe_sel_count);
			i++;
		}
	}
	result.SetCardinality(probe_sel_count);
	return OperatorResultType::NEED_MORE_INPUT;
}

void PerfectHashJoinExecutor::FillSelectionVectorSwitchProbe(Vector &source, SelectionVector &build_sel_vec,
                                                             SelectionVector &probe_sel_vec, idx_t count,
                                                             idx_t &probe_sel_count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedFillSelectionVectorProbe<int8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT16:
		TemplatedFillSelectionVectorProbe<int16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT32:
		TemplatedFillSelectionVectorProbe<int32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT64:
		TemplatedFillSelectionVectorProbe<int64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT8:
		TemplatedFillSelectionVectorProbe<uint8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT16:
		TemplatedFillSelectionVectorProbe<uint16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT32:
		TemplatedFillSelectionVectorProbe<uint32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT64:
		TemplatedFillSelectionVectorProbe<uint64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	default:
		throw NotImplementedException("Type not supported");
	}
}

template <typename T>
void PerfectHashJoinExecutor::TemplatedFillSelectionVectorProbe(Vector &source, SelectionVector &build_sel_vec,
                                                                SelectionVector &probe_sel_vec, idx_t count,
                                                                idx_t &probe_sel_count) {
	auto min_value = perfect_join_statistics.build_min;
	auto max_value = perfect_join_statistics.build_max;

	VectorData vector_data;
	source.Orrify(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	auto validity_mask = &vector_data.validity;
	// build selection vector for non-dense build
	if (validity_mask->AllValid()) {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			auto input_value = data[data_idx];

			if (perfect_join_statistics.is_physical_id) {
				input_value = GetPhysicalIdHash(input_value);
			}

			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	} else {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			if (!validity_mask->RowIsValid(data_idx)) {
				continue;
			}
			auto input_value = data[data_idx];

			if (perfect_join_statistics.is_physical_id) {
				input_value = GetPhysicalIdHash(input_value);
			}

			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	}
}

} // namespace duckdb
