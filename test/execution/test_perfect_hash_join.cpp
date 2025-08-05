#include "catch.hpp"
#include "test_config.hpp"

#include "execution/physical_operator/physical_perfect_hash_join.hpp"
#include "common/types/data_chunk.hpp"
#include "execution/execution_context.hpp"
#include "planner/expression/bound_reference_expression.hpp"
#include "main/database.hpp"
#include "common/limits.hpp"

using namespace duckdb;

void TestPerfectHashJoin(std::vector<std::vector<int>> left_data,
                         std::vector<std::vector<int>> right_data,
                         std::vector<std::vector<int>> expected_output) {
    // Set up input types: 2 integers
    vector<LogicalType> left_input_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    vector<LogicalType> right_input_types = {LogicalType::INTEGER, LogicalType::INTEGER};

    // Create input DataChunk
    DataChunk left_input;
    left_input.Initialize(left_input_types);
    left_input.SetCardinality(left_data.size());
    for (idx_t i = 0; i < left_data.size(); ++i) {
        left_input.SetValue(0, i, Value::INTEGER(left_data[i][0]));  // left.col0
        left_input.SetValue(1, i, Value::INTEGER(left_data[i][1]));  // left.col1
    }

    DataChunk right_input;
    right_input.Initialize(right_input_types);
    right_input.SetCardinality(right_data.size());
    for (idx_t i = 0; i < right_data.size(); ++i) {
        right_input.SetValue(0, i, Value::INTEGER(right_data[i][0]));  // right.col0
        right_input.SetValue(1, i, Value::INTEGER(right_data[i][1]));  // right.col1
    }

    // Create join expression: left.col1 = right.col0
    JoinCondition condition;
    condition.left = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 1);  // ref col 1
    condition.right = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 0);  // ref col 0
    condition.comparison = ExpressionType::COMPARE_EQUAL;

    vector<JoinCondition> conditions{};
    conditions.push_back(std::move(condition));

    // Create projections
    // left{col0, col1} + right{col0, col1} -> output{left.col0, left.col1, right.col0, right.col1}
    vector<uint32_t> left_projection_map = {0, 1};
    vector<uint32_t> right_projection_map = {2, 3};
    // set entire right input as build table value
    vector<LogicalType> right_build_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    vector<idx_t> right_build_map{0, 1};

    // Set up schema
    Schema schema;
    schema.setStoredTypes({LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER});  // output types

    // Create operator
    PhysicalPerfectHashJoin perfect_hash_join(schema, std::move(conditions), JoinType::INNER,
                                              left_projection_map, right_projection_map,
                                              right_build_types, right_build_map);

    // Execution context/state
    string dbdir = "/data/ldbc/sf1";
    auto db = make_unique<DuckDB>(dbdir.c_str());
    auto client_context = make_shared<ClientContext>(db->instance);
    ExecutionContext exec_context(client_context.get());
    auto sink_state = perfect_hash_join.GetLocalSinkState(exec_context);
    auto op_state = perfect_hash_join.GetOperatorState(exec_context);

    // Build Hash Table
    auto sink_result = perfect_hash_join.Sink(exec_context, right_input, *sink_state);

    // Combine sink state (do nothing indeed)
    perfect_hash_join.Combine(exec_context, *sink_state);

    // Create Output chunk
    DataChunk output;
    output.Initialize(schema.getStoredTypes());

    // Probe Hash Table
    auto result = perfect_hash_join.Execute(exec_context, left_input, output, *op_state, *sink_state);

    // Check result
    REQUIRE(result == OperatorResultType::HAVE_MORE_OUTPUT);
    REQUIRE(output.size() == expected_output.size());
    for (int i = 0; i < expected_output.size(); i++) {
        for (size_t j = 0; j < expected_output[i].size(); j++) {
            REQUIRE(output.GetValue(j, i) == expected_output[i][j]);
        }
    }
}

TEST_CASE("Test perfect hash join. Match one build input with one probe input") {
    TestPerfectHashJoin(
        {{10, 20}},  // left input
        {{20, 30}},  // right input
        {{10, 20, 20, 30}}  // expected output
    );
}

TEST_CASE("Test perfect hash join. Test multiple inputs") {
    TestPerfectHashJoin(
        {{10, 20}, {20, 40}, {30, 40}},
        {{10, 10}, {20, 20}, {30, 30}, {40, 40}, {50, 50}},
        {{10, 20, 20, 20}, {20, 40, 40, 40}, {30, 40, 40, 40}}
    );
}

// TODO: eliminate duplicate code!!

void PHJProjectionTest(std::vector<std::vector<int>> left_data,
                         std::vector<std::vector<int>> right_data,
                         std::vector<std::vector<int>> expected_output) {
    // Set up input types: 3 integers
    vector<LogicalType> left_input_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};
    vector<LogicalType> right_input_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};

    // Create input DataChunk
    DataChunk left_input;
    left_input.Initialize(left_input_types);
    left_input.SetCardinality(left_data.size());
    for (idx_t i = 0; i < left_data.size(); ++i) {
        left_input.SetValue(0, i, Value::INTEGER(left_data[i][0]));  // left.col0
        left_input.SetValue(1, i, Value::INTEGER(left_data[i][1]));  // left.col1
        left_input.SetValue(2, i, Value::INTEGER(left_data[i][2]));  // left.col2
    }

    DataChunk right_input;
    right_input.Initialize(right_input_types);
    right_input.SetCardinality(right_data.size());
    for (idx_t i = 0; i < right_data.size(); ++i) {
        right_input.SetValue(0, i, Value::INTEGER(right_data[i][0]));  // right.col0
        right_input.SetValue(1, i, Value::INTEGER(right_data[i][1]));  // right.col1
        right_input.SetValue(2, i, Value::INTEGER(right_data[i][2]));  // right.col2
    }

    // Create join expression: left.col1 = right.col0
    JoinCondition condition;
    condition.left = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 1);  // ref col 1
    condition.right = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 0);  // ref col 0
    condition.comparison = ExpressionType::COMPARE_EQUAL;

    vector<JoinCondition> conditions{};
    conditions.push_back(std::move(condition));

    // Create projections
    // left{col0, col1, col2} + right{col0, col1, col2} -> output{left.col1, left.col0, right.col0, right.col2}
    vector<uint32_t> left_projection_map = {1, 0, std::numeric_limits<uint32_t>::max()};
    vector<uint32_t> right_projection_map = {3, 2};
    // set entire right input as build table value
    vector<LogicalType> right_build_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    vector<idx_t> right_build_map{2, 0};

    // Set up schema
    Schema schema;
    schema.setStoredTypes({LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER});  // output types

    // Create operator
    PhysicalPerfectHashJoin perfect_hash_join(schema, std::move(conditions), JoinType::INNER,
                                              left_projection_map, right_projection_map,
                                              right_build_types, right_build_map);

    // Execution context/state
    string dbdir = "/data/ldbc/sf1";
    auto db = make_unique<DuckDB>(dbdir.c_str());
    auto client_context = make_shared<ClientContext>(db->instance);
    ExecutionContext exec_context(client_context.get());
    auto sink_state = perfect_hash_join.GetLocalSinkState(exec_context);
    auto op_state = perfect_hash_join.GetOperatorState(exec_context);

    // Build Hash Table
    auto sink_result = perfect_hash_join.Sink(exec_context, right_input, *sink_state);

    // Combine sink state (do nothing indeed)
    perfect_hash_join.Combine(exec_context, *sink_state);

    // Create Output chunk
    DataChunk output;
    output.Initialize(schema.getStoredTypes());

    // Probe Hash Table
    auto result = perfect_hash_join.Execute(exec_context, left_input, output, *op_state, *sink_state);

    // Check result
    REQUIRE(result == OperatorResultType::HAVE_MORE_OUTPUT);
    REQUIRE(output.size() == expected_output.size());
    for (int i = 0; i < expected_output.size(); i++) {
        for (size_t j = 0; j < expected_output[i].size(); j++) {
            REQUIRE(output.GetValue(j, i) == expected_output[i][j]);
        }
    }
}

TEST_CASE("Test perfect hash join. Test projection") {
    PHJProjectionTest(
        {{10, 20, 30}},
        {{20, 40, 50}},
        {{20, 10, 20, 50}}
    );
}

void PHJMultiCondTest(std::vector<std::vector<int>> left_data,
                         std::vector<std::vector<int>> right_data,
                         std::vector<std::vector<int>> expected_output) {
    // Set up input types: 3 integers
    vector<LogicalType> left_input_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};
    vector<LogicalType> right_input_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};

    // Create input DataChunk
    DataChunk left_input;
    left_input.Initialize(left_input_types);
    left_input.SetCardinality(left_data.size());
    for (idx_t i = 0; i < left_data.size(); ++i) {
        left_input.SetValue(0, i, Value::INTEGER(left_data[i][0]));  // left.col0
        left_input.SetValue(1, i, Value::INTEGER(left_data[i][1]));  // left.col1
        left_input.SetValue(2, i, Value::INTEGER(left_data[i][2]));  // left.col2
    }

    DataChunk right_input;
    right_input.Initialize(right_input_types);
    right_input.SetCardinality(right_data.size());
    for (idx_t i = 0; i < right_data.size(); ++i) {
        right_input.SetValue(0, i, Value::INTEGER(right_data[i][0]));  // right.col0
        right_input.SetValue(1, i, Value::INTEGER(right_data[i][1]));  // right.col1
        right_input.SetValue(2, i, Value::INTEGER(right_data[i][2]));  // right.col2
    }

    // Create join expression: left.col1 = right.col0
    JoinCondition condition1;
    condition1.left = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 1);  // ref col 1
    condition1.right = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 0);  // ref col 0
    condition1.comparison = ExpressionType::COMPARE_EQUAL;
    JoinCondition condition2;
    condition2.left = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 2);  // ref col 1
    condition2.right = std::make_unique<BoundReferenceExpression>(LogicalType::INTEGER, 2);  // ref col 0
    condition2.comparison = ExpressionType::COMPARE_EQUAL;

    vector<JoinCondition> conditions{};
    conditions.push_back(std::move(condition1));
    conditions.push_back(std::move(condition2));

    // Create projections
    // left{col0, col1, col2} + right{col0, col1, col2} -> output{left.col0, left.col1, left.col2, right.col0, right.col1, right.col2}
    vector<uint32_t> left_projection_map = {0, 1, 2};
    vector<uint32_t> right_projection_map = {3, 4, 5};
    // set entire right input as build table value
    vector<LogicalType> right_build_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};
    vector<idx_t> right_build_map{0, 1, 2};

    // Set up schema
    Schema schema;
    schema.setStoredTypes({LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER});  // output types

    // Create operator
    PhysicalPerfectHashJoin perfect_hash_join(schema, std::move(conditions), JoinType::INNER,
                                              left_projection_map, right_projection_map,
                                              right_build_types, right_build_map);

    // Execution context/state
    string dbdir = "/data/ldbc/sf1";
    auto db = make_unique<DuckDB>(dbdir.c_str());
    auto client_context = make_shared<ClientContext>(db->instance);
    ExecutionContext exec_context(client_context.get());
    auto sink_state = perfect_hash_join.GetLocalSinkState(exec_context);
    auto op_state = perfect_hash_join.GetOperatorState(exec_context);

    // Build Hash Table
    auto sink_result = perfect_hash_join.Sink(exec_context, right_input, *sink_state);

    // Combine sink state (do nothing indeed)
    perfect_hash_join.Combine(exec_context, *sink_state);

    // Create Output chunk
    DataChunk output;
    output.Initialize(schema.getStoredTypes());

    // Probe Hash Table
    auto result = perfect_hash_join.Execute(exec_context, left_input, output, *op_state, *sink_state);

    // Check result
    REQUIRE(result == OperatorResultType::HAVE_MORE_OUTPUT);
    REQUIRE(output.size() == expected_output.size());
    for (int i = 0; i < expected_output.size(); i++) {
        for (size_t j = 0; j < expected_output[i].size(); j++) {
            REQUIRE(output.GetValue(j, i) == expected_output[i][j]);
        }
    }
}

TEST_CASE("Test perfect hash join. Test multiple join condition") {
    PHJMultiCondTest(
        {{10, 20, 30}},
        {{20, 40, 30}},
        {{10, 20, 30, 20, 40, 30}}
    );
}
