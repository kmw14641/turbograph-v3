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
    int current_operator_index;

    // Sub pipeline
    std::vector<CypherPipeline> sub_pipelines;

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

    // Column mapping from input to output
    std::unordered_map<std::string, std::string> column_mapping;

    // Track which columns are actually used in expressions
    std::unordered_set<std::string> used_columns;

    // GPU memory management (integrated from LazyMaterializationInfo)
    std::unordered_map<std::string, bool> gpu_memory_loaded;
    std::unordered_map<std::string, std::string> column_to_param_mapping;
    std::unordered_map<std::string, std::string> column_to_table_mapping;
    std::unordered_map<std::string, idx_t> column_to_extent_mapping;
    std::unordered_map<std::string, idx_t> column_to_chunk_mapping;

    PipelineContext() : total_operators(0), current_operator_index(0) {}

    // Initialize pipeline context with all operator schemas
    void InitializePipeline(const CypherPipeline &pipeline);

    // Move to next operator and update current schemas
    void MoveToOperator(int op_idx);

    // Helper methods for GPU memory management
    void AddGPUColumn(const std::string &table_name,
                      const std::string &column_name, idx_t extent_id,
                      idx_t chunk_id, const std::string &param_name)
    {
        std::string key = table_name + "." + column_name;
        gpu_memory_loaded[key] = false;
        column_to_param_mapping[key] = param_name;
        column_to_table_mapping[key] = table_name;
        column_to_extent_mapping[key] = extent_id;
        column_to_chunk_mapping[key] = chunk_id;
    }

    bool IsGPUColumnLoaded(const std::string &table_name,
                           const std::string &column_name) const
    {
        std::string key = table_name + "." + column_name;
        auto it = gpu_memory_loaded.find(key);
        return it != gpu_memory_loaded.end() && it->second;
    }

    void MarkGPUColumnAsLoaded(const std::string &table_name,
                               const std::string &column_name)
    {
        std::string key = table_name + "." + column_name;
        gpu_memory_loaded[key] = true;
    }

    std::string GetColumnParamName(const std::string &table_name,
                                   const std::string &column_name) const
    {
        std::string key = table_name + "." + column_name;
        auto it = column_to_param_mapping.find(key);
        return it != column_to_param_mapping.end() ? it->second : "";
    }
};

} // namespace duckdb