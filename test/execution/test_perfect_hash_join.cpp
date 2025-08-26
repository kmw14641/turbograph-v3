#include "catch.hpp"
#include "test_config.hpp"

#include "execution/physical_operator/physical_hash_join.hpp"
#include "common/types/data_chunk.hpp"
#include "execution/execution_context.hpp"
#include "planner/expression/bound_reference_expression.hpp"
#include "main/database.hpp"
#include "common/limits.hpp"

using namespace duckdb;

class PHJConfig {
public:
    vector<LogicalType> left_input_types;
    vector<LogicalType> right_input_types;
    vector<LogicalType> output_types;
    idx_t left_key_idx;
    idx_t right_key_idx;
    vector<uint32_t> left_projection_map;
    vector<uint32_t> right_projection_map;
    vector<LogicalType> right_build_types;
    vector<idx_t> right_build_map;
    PerfectHashJoinStats perfect_join_statistics;

    PHJConfig() {
        SetDefaultConfig();
    }

    void SetDefaultConfig() {
        SetDefaultConfig(2, 2, 4);
        left_key_idx = 1;
        right_key_idx = 0;
        left_projection_map = {0, 1};
        right_projection_map = {2, 3};
        right_build_types = {LogicalType::INTEGER, LogicalType::INTEGER};
        right_build_map = {0, 1};
    }

    void SetDefaultConfig(size_t left_input_size, size_t right_input_size, size_t output_size) {
        left_input_types = vector<LogicalType>(left_input_size, LogicalType::INTEGER);
        right_input_types = vector<LogicalType>(right_input_size, LogicalType::INTEGER);
        output_types = vector<LogicalType>(output_size, LogicalType::INTEGER);
    }

    void SetDefaultStat(int64_t min, int64_t max) {
        perfect_join_statistics.build_min = min;
        perfect_join_statistics.build_max = max;
        perfect_join_statistics.build_range = max - min;
        perfect_join_statistics.is_build_small = true;
        perfect_join_statistics.is_physical_id = false;
    }
};


std::vector<std::vector<int>> DoPerfectHashJoin(PHJConfig phj_config,
                         std::vector<std::vector<int>> left_data,
                         std::vector<std::vector<int>> right_data) {
    // Create input DataChunk
    std::vector<std::unique_ptr<DataChunk>> left_inputs;
    idx_t left_data_idx = 0;
    while (left_data_idx < left_data.size()) {
        std::unique_ptr<DataChunk> left_input = std::make_unique<DataChunk>();
        idx_t left_input_idx = 0;
        left_input->Initialize(phj_config.left_input_types);
        while (left_input_idx < STANDARD_VECTOR_SIZE && left_data_idx < left_data.size()) {
            for (idx_t j = 0; j < phj_config.left_input_types.size(); ++j) {
                left_input->SetValue(j, left_input_idx, Value::INTEGER(left_data[left_data_idx][j]));
            }
            left_data_idx++;
            left_input_idx++;
        }
        left_input->SetCardinality(left_input_idx);
        left_inputs.push_back(std::move(left_input));
    }

    std::vector<std::unique_ptr<DataChunk>> right_inputs;
    idx_t right_data_idx = 0;
    while (right_data_idx < right_data.size()) {
        std::unique_ptr<DataChunk> right_input = std::make_unique<DataChunk>();
        idx_t right_input_idx = 0;
        right_input->Initialize(phj_config.right_input_types);
        while (right_input_idx < STANDARD_VECTOR_SIZE && right_data_idx < right_data.size()) {
            for (idx_t j = 0; j < phj_config.right_input_types.size(); ++j) {
                right_input->SetValue(j, right_input_idx, Value::INTEGER(right_data[right_data_idx][j]));
            }
            right_data_idx++;
            right_input_idx++;
        }
        right_input->SetCardinality(right_input_idx);
        right_inputs.push_back(std::move(right_input));
    }

    // Create join expression
    JoinCondition condition;
    condition.left = std::make_unique<BoundReferenceExpression>(phj_config.left_input_types[phj_config.left_key_idx], phj_config.left_key_idx);
    condition.right = std::make_unique<BoundReferenceExpression>(phj_config.right_input_types[phj_config.right_key_idx], phj_config.right_key_idx);
    condition.comparison = ExpressionType::COMPARE_EQUAL;

    vector<JoinCondition> conditions{};
    conditions.push_back(std::move(condition));

    // Set up schema
    Schema schema;
    schema.setStoredTypes(phj_config.output_types);

    // Create Execution context
    string dbdir = "/data/ldbc/sf1";
    auto db = make_unique<DuckDB>(dbdir.c_str());
    auto client_context = make_shared<ClientContext>(db->instance);
    ExecutionContext exec_context(client_context.get());

    // Start Timer
    auto start = std::chrono::high_resolution_clock::now();

    // Create operator
    PhysicalHashJoin perfect_hash_join(schema, std::move(conditions), JoinType::INNER,
                                              phj_config.left_projection_map, phj_config.right_projection_map,
                                              phj_config.right_build_types, phj_config.right_build_map, phj_config.perfect_join_statistics);

    // Create Execution state
    auto sink_state = perfect_hash_join.GetLocalSinkState(exec_context);
    auto op_state = perfect_hash_join.GetOperatorState(exec_context);

    // Build Hash Table
    for (int i = 0; i < right_inputs.size(); i++) {
        auto sink_result = perfect_hash_join.Sink(exec_context, *right_inputs[i], *sink_state);
    }

    // Combine sink state (do nothing indeed)
    perfect_hash_join.Combine(exec_context, *sink_state);

    // Create Output chunk
    vector<unique_ptr<DataChunk>> outputs;

    // Probe Hash Table
    vector<vector<int>> result_vector;
    for (int i = 0; i < left_inputs.size(); i++) {
        OperatorResultType result;
        do {
            unique_ptr<DataChunk> output = std::make_unique<DataChunk>();
            output->Initialize(schema.getStoredTypes());
            result = perfect_hash_join.Execute(exec_context, *left_inputs[i], *output, *op_state, *sink_state);

            // I don't want to serialize while timer is running, but output should be processed immediately, or overwrite occurs
            for (idx_t result_idx = 0; result_idx < output->size(); result_idx++) {
                result_vector.push_back(vector<int>());
                for (size_t j = 0; j < phj_config.output_types.size(); j++) {
                    result_vector[result_vector.size() - 1].push_back(output->GetValue(j, result_idx).GetValue<int>());
                }
            }
        } while (result == OperatorResultType::HAVE_MORE_OUTPUT);
    }

    // End Timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << duration.count() << " us\n";

    return result_vector;
}

TEST_CASE("Test perfect hash join. Match one build input with one probe input") {
    PHJConfig phj_config;
    phj_config.SetDefaultStat(10, 30);
    auto result = DoPerfectHashJoin(
        phj_config,
        {{10, 20}},  // left input
        {{20, 30}}  // right input
    );
    vector<vector<int>> expected_output = {{10, 20, 20, 30}};
    REQUIRE(result == expected_output);
}

TEST_CASE("Test perfect hash join. Test multiple inputs") {
    PHJConfig phj_config;
    phj_config.SetDefaultStat(10, 50);
    auto result = DoPerfectHashJoin(
        phj_config,
        {{10, 20}, {20, 40}, {30, 40}},
        {{10, 10}, {20, 20}, {30, 30}, {40, 40}, {50, 50}}
    );
    vector<vector<int>> expected_output = {{10, 20, 20, 20}, {20, 40, 40, 40}, {30, 40, 40, 40}};
    REQUIRE(result == expected_output);
}

TEST_CASE("Test perfect hash join. Test projection") {
    PHJConfig phj_config;
    phj_config.SetDefaultConfig(3, 3, 4);
    phj_config.left_projection_map = {1, 0, std::numeric_limits<uint32_t>::max()};
    phj_config.right_build_map = {2, 0};
    phj_config.right_projection_map = {3, 2};
    phj_config.SetDefaultStat(10, 50);
    auto result = DoPerfectHashJoin(
        phj_config,
        {{10, 20, 30}},
        {{20, 40, 50}}
    );
    vector<vector<int>> expected_output = {{20, 10, 20, 50}};
    REQUIRE(result == expected_output);
}
