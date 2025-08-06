#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/types.hpp"
#include "execution/cypher_pipeline.hpp"

namespace duckdb {
    
// Helper class for code generation with indentation
class CodeBuilder {
   public:
    void Add(const std::string &line)
    {
        for (int i = 0; i < nesting_level; ++i) {
            ss << "    ";  // 4 spaces per level
        }
        ss << line << "\n";
    }
    std::string str() const { return ss.str(); }
    void clear()
    {
        ss.str("");
        ss.clear();
    }
    void IncreaseNesting() { nesting_level++; }
    void DecreaseNesting() { nesting_level--; }

   private:
    std::stringstream ss;
    int nesting_level = 0;
};

// Structure to track pipeline state and dependencies
struct PipelineContext {
    // Pipeline-wide information
    int total_operators;
    int cur_op_idx;
    int current_sub_pipeline_index;

    // Pipeline
    CypherPipeline *current_pipeline = nullptr;

    // Sub pipeline
    std::vector<CypherPipeline> sub_pipelines;
    std::string current_tid_name;

    // Operator schemas (references to original schemas)
    std::vector<const std::vector<std::string> *> operator_column_names;
    std::vector<const std::vector<LogicalType> *> operator_column_types;

    // Current operator's input schema (from previous operator)
    std::vector<std::string> input_column_names;
    std::vector<LogicalType> input_column_types;

    // Current operator's output schema
    std::vector<std::string> output_column_names;
    std::vector<LogicalType> output_column_types;

    // Materialization status for each column (logical level)
    std::unordered_map<std::string, bool> column_materialized;
    std::vector<std::string> columns_to_be_materialized;
    std::vector<LogicalType> column_types_to_be_materialized;

    // Track which columns are actually used in expressions
    std::unordered_set<std::string> used_columns;

    std::unordered_map<std::string, std::string> column_to_param_mapping;

    PipelineContext() : total_operators(0), cur_op_idx(0) {}

    // Initialize pipeline context with all operator schemas
    void InitializePipeline(CypherPipeline &pipeline);

    // Move to next operator and update current schemas
    void AdvanceOperator();

    void GetReferencedColumns(Expression *expr,
                              std::vector<uint64_t> &referenced_columns);
};

} // namespace duckdb