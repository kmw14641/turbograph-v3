#include "execution/physical_operator/physical_perfect_hash_join.hpp"
#include "common/vector_operations/vector_operations.hpp"
#include "execution/expression_executor.hpp"
#include "function/aggregate/distributive_functions.hpp"
#include "main/client_context.hpp"
#include "parallel/thread_context.hpp"
#include "planner/expression/bound_aggregate_expression.hpp"
#include "storage/buffer_manager.hpp"
#include "storage/storage_manager.hpp"

#include "common/output_util.hpp"

namespace duckdb {

/**
 * PerfectHashJoin is enabled only when right key is not duplicated and left key always matches with any right key
*/
PhysicalPerfectHashJoin::PhysicalPerfectHashJoin(
    Schema sch, vector<JoinCondition> cond, JoinType join_type,
    vector<uint32_t> &output_left_projection_map,  // s62 style projection map
    vector<uint32_t> &output_right_projection_map,  // s62 style projection map (index matches build value's output, value indicates output schema's index)
    vector<LogicalType> &right_build_types,
    vector<idx_t> &right_build_map  // right column indexes to build (index matches build value's input, value indicates right input's index)
    )
    : PhysicalComparisonJoin(sch, PhysicalOperatorType::HASH_JOIN, move(cond),
                             join_type),
      build_types(right_build_types),
      build_map(right_build_map),
      output_left_projection_map(output_left_projection_map),
      output_right_projection_map(output_right_projection_map)
{
    D_ASSERT(conditions.size() == 1);  // TODO: implement PerfectHashFunction::CombineHash
    for (auto &condition : conditions) {
        condition_types.push_back(condition.left->return_type);
    }

    D_ASSERT(build_types.size() == build_map.size());
    if (join_type == JoinType::ANTI || join_type == JoinType::SEMI) {  // why mark is not here?
        D_ASSERT(build_types.size() == 0);
    }

    D_ASSERT(delim_types.size() == 0);
}


//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class PerfectHashJoinLocalSinkState : public LocalSinkState {  // Where is GlobalSinkState? We have to combine local hash table to global hash table!
   public:
    DataChunk build_chunk;
    DataChunk join_keys;
    ExpressionExecutor build_executor;

    //! The HT used by the join
    unique_ptr<JoinHashTable> hash_table;
    //! Whether or not the hash table has been finalized
    bool finalized = false;
};

unique_ptr<LocalSinkState> PhysicalPerfectHashJoin::GetLocalSinkState(
    ExecutionContext &context) const
{
    auto state = make_unique<PerfectHashJoinLocalSinkState>();
    if (!build_map.empty()) {
        state->build_chunk.Initialize(build_types);
    }
    for (auto &cond : conditions) {
        state->build_executor.AddExpression(*cond.right);
    }
    state->join_keys.Initialize(condition_types);

    // globals
    state->hash_table = make_unique<JoinHashTable>(
        BufferManager::GetBufferManager(*(context.client->db.get())),
        conditions, build_types, join_type, output_left_projection_map, output_right_projection_map);

    return move(state);
}

/**
 * TODO:
 * Bug in schemaless execution: NULL type data is not handled properly.
 * If I execute MATCH (m:Comment)-[r:HAS_CREATOR]->(p:Person)
		RETURN
			m.id AS messageId,
			p.lastName AS lastName,
			p.firstName AS firstName,
			p.id AS personId
    It occurs error in hash_table->Build(). It is because of NULL type data. 
    The type is VARCHAR, but there is NULL data in the column.
    However, null is not setted.
*/
SinkResultType PhysicalPerfectHashJoin::Sink(ExecutionContext &context,
                                      DataChunk &input,
                                      LocalSinkState &state) const
{
    auto &sink = (PerfectHashJoinLocalSinkState &)state;
    auto &lstate = (PerfectHashJoinLocalSinkState &)state;
    // resolve the join keys for the right chunk
    lstate.join_keys.Reset();
    lstate.build_executor.Execute(input, lstate.join_keys);
    // fill build_chunk from input, using build_map
    lstate.build_chunk.Reset();
    for (idx_t i = 0; i < build_map.size(); i++) {
        lstate.build_chunk.data[i].Reference(input.data[build_map[i]]);
    }
    lstate.build_chunk.SetCardinality(input);
    // build the HT
    sink.hash_table->Build(lstate.join_keys, lstate.build_chunk);
    return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalPerfectHashJoin::Combine(ExecutionContext &context,
                               LocalSinkState &lstate) const
{
    auto &state = (PerfectHashJoinLocalSinkState &)lstate;
    // finalize contexts
    auto &sink = (PerfectHashJoinLocalSinkState &)lstate;
    sink.hash_table->Finalize();  // why Sink only creates block, and combines block in Combine? Combine exists for combine different pipeline to global state. block is not helpful for parallel execution. is it for cache efficiency?
    sink.finalized = true;
}

DataChunk &PhysicalPerfectHashJoin::GetLastSinkedData(LocalSinkState &lstate) const
{
    auto &state = (PerfectHashJoinLocalSinkState &)lstate;
    return state.build_chunk;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class PerfectHashJoinOperatorState : public OperatorState {
   public:
    //! The join keys used to probe the HT
    DataChunk join_keys;
    //! expression executor that extracts left side of the join keys in the input
    ExpressionExecutor probe_executor;
    //! The scan structure used to scan the HT after probing
    unique_ptr<JoinHashTable::ScanStructure> scan_structure;

   public:
    void Finalize(PhysicalOperator *op, ExecutionContext &context) override
    {
        // context.thread.profiler.Flush(op, &probe_executor, "probe_executor", 0);
    }
};

unique_ptr<OperatorState> PhysicalPerfectHashJoin::GetOperatorState(
    ExecutionContext &context) const
{
    auto state = make_unique<PerfectHashJoinOperatorState>();
    state->join_keys.Initialize(condition_types);
    for (auto &cond : conditions) {
        state->probe_executor.AddExpression(*cond.left);
    }
    return move(state);
}

/**
 * TODO:
 * Where is output mapping code? It should be here.
*/
OperatorResultType PhysicalPerfectHashJoin::Execute(ExecutionContext &context,
                                             DataChunk &input, DataChunk &chunk,
                                             OperatorState &state_p,
                                             LocalSinkState &sink_state) const
{
    auto &state = (PerfectHashJoinOperatorState &)state_p;
    auto &sink = (PerfectHashJoinLocalSinkState &)sink_state;
    D_ASSERT(sink.finalized);

    if (sink.hash_table->Count() == 0 && EmptyResultIfRHSIsEmpty()) {
        return OperatorResultType::FINISHED;
    }

    /**
	 * NOTE
	 * See assertion in ScanStructure::NextInnerJoin (D_ASSERT(result.ColumnCount() == left.ColumnCount() + ht.build_types.size()))
	 * In the code, it slides left and concat to result. And then it concats the build chunk to the result.
	 * This means that DuckDB does not consider the case where the some columns in the lhs are not used in the output.
	 * For example, if the join key is not included in the final output, since it only used in the join, DuckDB outputs error.
	*/

    // DataChunk preprocessed_input;
    // // Get types. See output_left_projection_map. if std::numeric_limits<uint32_t>::max(), then it is not used in the output
    // vector<LogicalType> input_types = input.GetTypes();
    // vector<LogicalType> prep_input_types;
    // for (auto i = 0; i < output_left_projection_map.size(); i++) {
    //     if (output_left_projection_map[i] !=
    //         std::numeric_limits<uint32_t>::max()) {
    //         prep_input_types.push_back(input_types[i]);
    //     }
    // }
    // // Initialize and fill preprocessed_input
    // preprocessed_input.Initialize(prep_input_types);
    // idx_t prep_idx = 0;
    // for (idx_t input_idx = 0; input_idx < output_left_projection_map.size();
    //      input_idx++) {
    //     if (output_left_projection_map[input_idx] !=
    //         std::numeric_limits<uint32_t>::max()) {
    //         preprocessed_input.data[prep_idx++].Reference(
    //             input.data[input_idx]);
    //     }
    // }
    // preprocessed_input.SetCardinality(input.size());
    // preprocessed_input.SetSchemaIdx(input.GetSchemaIdx());

    // TODO: currently, for debug purpose, we assume the chunk is UNION schema.
    chunk.SetSchemaIdx(0);

    num_loops++;
    if (state.scan_structure) {
        // still have elements remaining from the previous probe (i.e. we got
        // >1024 elements in the previous probe)
        state.scan_structure->Next(state.join_keys, input, chunk);
        if (chunk.size() > 0) {
            return OperatorResultType::HAVE_MORE_OUTPUT;
        }
        state.scan_structure = nullptr;
        return OperatorResultType::NEED_MORE_INPUT;
    }
    // probe the HT
    if (sink.hash_table->Count() == 0) {  // number of tuples in a rhs
        ConstructEmptyJoinResult(sink.hash_table->join_type,
                                 sink.hash_table->has_null, input,
                                 chunk);
        return OperatorResultType::NEED_MORE_INPUT;
    }
    // resolve the join keys for the left chunk
    state.join_keys.Reset();
    state.probe_executor.Execute(input, state.join_keys);

    // perform the actual probe
    state.scan_structure = sink.hash_table->Probe(state.join_keys);
    state.scan_structure->Next(state.join_keys, input, chunk);
    return OperatorResultType::HAVE_MORE_OUTPUT;
}

std::string PhysicalPerfectHashJoin::ParamsToString() const
{
    std::string result = "";
    result += "output_left_projection_map.size()=" +
              std::to_string(output_left_projection_map.size()) + "[";
    for (auto &i : output_left_projection_map) {
        result += std::to_string(i) + ", ";
    }
    result += "], ";
    result += "output_right_projection_map.size()=" +
              std::to_string(output_right_projection_map.size()) + "[";
    for (auto &i : output_right_projection_map) {
        result += std::to_string(i) + ", ";
    }
    result += "], ";
    result += "build_map.size()=" +
              std::to_string(build_map.size()) + "[";
    for (auto &i : build_map) {
        result += std::to_string(i) + ", ";
    }
    result += "], ";
    result +=
        "condition_types.size()=" + std::to_string(condition_types.size()) +
        " [";
    for (auto &condition_type : condition_types) {
        result += condition_type.ToString() + ", ";
    }
    result += "], ";
    result += "build_types.size()=" + std::to_string(build_types.size()) + " [";
    for (auto &build_type : build_types) {
        result += build_type.ToString() + ", ";
    }
    result += "], join conditions: ";
    for (auto &expression: conditions) {
        result += expression.left->ToString() + ", ";
        result += expression.right->ToString() + ",";
        result += " / ";
    }
    return result;
}

std::string PhysicalPerfectHashJoin::ToString() const
{
    return "PerfectHashJoin";
}

}  // namespace duckdb
