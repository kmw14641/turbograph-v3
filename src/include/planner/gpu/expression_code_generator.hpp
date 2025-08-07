#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "common/types.hpp"
#include "common/types/value.hpp"
#include "planner/expression.hpp"
#include "planner/expression/bound_comparison_expression.hpp"
#include "planner/expression/bound_constant_expression.hpp"
#include "planner/expression/bound_function_expression.hpp"
#include "planner/expression/bound_operator_expression.hpp"
#include "planner/expression/bound_reference_expression.hpp"
#include "planner/expression/bound_conjunction_expression.hpp"
#include "planner/expression/bound_between_expression.hpp"
#include "planner/gpu/codegen_utils.hpp"

namespace duckdb {

// Expression code generator for generating GPU code from expressions
class ExpressionCodeGenerator {
   public:
    ExpressionCodeGenerator(ClientContext &context);

    // Generate GPU code for an expression and return the result variable name
    std::string GenerateExpressionCode(
        Expression *expr, CodeBuilder &code, PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    // Generate condition code for filter expressions
    std::string GenerateConditionCode(
        Expression *expr, CodeBuilder &code, PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

   private:
    ClientContext &context;
    int expr_counter;  // For generating unique variable names

    // Generate code for different expression types
    std::string GenerateReferenceExpression(
        BoundReferenceExpression *ref_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);
    
    std::string GenerateBetweenExpression(
        BoundBetweenExpression *between_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    std::string GenerateConstantExpression(
        BoundConstantExpression *const_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    std::string GenerateComparisonExpression(
        BoundComparisonExpression *comp_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    std::string GenerateFunctionExpression(
        BoundFunctionExpression *func_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    std::string GenerateOperatorExpression(
        BoundOperatorExpression *op_expr, CodeBuilder &code,
        PipelineContext &pipeline_ctx,
        std::unordered_map<uint64_t, std::string> &column_map);

    // Helper methods
    std::string GetUniqueVariableName(const std::string &prefix);
    std::string ConvertLogicalTypeToCUDAType(LogicalType type);
    std::string ConvertValueToCUDALiteral(const Value &value);
    std::string GetComparisonOperator(ExpressionType type);
    std::string GetArithmeticOperator(ExpressionType type);
    std::string GetLogicalOperator(ExpressionType type);

    // Lazy materialization helper
    void EnsureColumnMaterialized(const std::string &column_name,
                                  CodeBuilder &code,
                                  PipelineContext &pipeline_ctx);
};

}  // namespace duckdb