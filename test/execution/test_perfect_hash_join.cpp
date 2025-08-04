#include "catch.hpp"
#include "test_config.hpp"

#include "execution/physical_operator/physical_perfect_hash_join.hpp"
#include "common/types/data_chunk.hpp"
#include "execution/execution_context.hpp"
#include "planner/expression/bound_reference_expression.hpp"

using namespace duckdb;

TEST_CASE("Test perfect hash join. Match one build input with one probe input") {
    // 1. Set up input types: 2 integers
    vector<LogicalType> left_input_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    vector<LogicalType> right_input_types = {LogicalType::INTEGER, LogicalType::INTEGER};

    // 2. Create input DataChunk
    DataChunk left_input;
    left_input.Initialize(left_input_types);
    left_input.SetCardinality(1);
    left_input.SetValue(0, 0, Value::INTEGER(10));  // left.col0 = 10
    left_input.SetValue(1, 0, Value::INTEGER(20));  // left.col1 = 20

    DataChunk right_input;
    right_input.Initialize(right_input_types);
    right_input.SetCardinality(1);
    right_input.SetValue(0, 0, Value::INTEGER(20));  // right.col0 = 20
    right_input.SetValue(1, 0, Value::INTEGER(30));  // right.col1 = 30

    // 3. Create join expression: left.col1 = right.col0
    JoinCondition condition;
    condition.left = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 1);  // ref col 1
    condition.right = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 0);  // ref col 0
    condition.comparison = ExpressionType::COMPARE_EQUAL;

    vector<JoinCondition> conditions{};
    conditions.push_back(std::move(condition));

    // 4. Create projections
    // left{col0, col1} + right{col0, col1} -> output{left.col0, left.col1, right.col0, right.col1}
    vector<uint32_t> left_projection_map = {0, 1};
    vector<uint32_t> right_projection_map = {2, 3};
    // set entire right input as build table value
    vector<LogicalType> right_build_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    vector<idx_t> right_build_map{0, 1};

    // 5. Set up schema
    Schema schema;
    schema.setStoredTypes({LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER});  // output types

    // 6. Create operator
    PhysicalPerfectHashJoin perfect_hash_join(schema, std::move(conditions), JoinType::INNER,
                                              left_projection_map, right_projection_map,
                                              right_build_types, right_build_map);

    // 7. Execution context/state
    ExecutionContext exec_context(nullptr);
    auto sink_state = perfect_hash_join.GetLocalSinkState(exec_context);
    auto op_state = perfect_hash_join.GetOperatorState(exec_context);

    // 8. Build Hash Table
    auto sink_result = perfect_hash_join.Sink(exec_context, right_input, *sink_state);

    // 9. Combine sink state (do nothing indeed)
    perfect_hash_join.Combine(exec_context, *sink_state);

    // 10. Create Output chunk
    DataChunk output;
    output.Initialize(schema.getStoredTypes(), 1);

    // 11. Probe Hash Table
    auto result = perfect_hash_join.Execute(exec_context, left_input, output, *op_state, *sink_state);

    // 12. Check result
    REQUIRE(result == OperatorResultType::NEED_MORE_INPUT);
    REQUIRE(output.size() == 1);
    REQUIRE(output.GetValue(0, 0) == Value::INTEGER(10));
    REQUIRE(output.GetValue(1, 0) == Value::INTEGER(20));
    REQUIRE(output.GetValue(2, 0) == Value::INTEGER(20));
    REQUIRE(output.GetValue(3, 0) == Value::INTEGER(30));
}
