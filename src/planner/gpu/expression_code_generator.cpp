#include "planner/gpu/expression_code_generator.hpp"
#include <algorithm>
#include "common/string_util.hpp"
#include "main/client_context.hpp"

namespace duckdb {

// ExpressionCodeGenerator implementation
ExpressionCodeGenerator::ExpressionCodeGenerator(ClientContext &context)
    : context(context), expr_counter(0)
{}

std::string ExpressionCodeGenerator::GenerateExpressionCode(
    Expression *expr, CodeBuilder &code, PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    D_ASSERT(expr != nullptr);

    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF:
            return GenerateReferenceExpression(
                dynamic_cast<BoundReferenceExpression *>(expr), code,
                pipeline_ctx, column_map);
        
        case ExpressionClass::BOUND_BETWEEN:
            return GenerateBetweenExpression(
                dynamic_cast<BoundBetweenExpression *>(expr), code,
                pipeline_ctx, column_map);

        case ExpressionClass::BOUND_CONSTANT:
            return GenerateConstantExpression(
                dynamic_cast<BoundConstantExpression *>(expr), code,
                pipeline_ctx, column_map);

        case ExpressionClass::BOUND_COMPARISON:
            return GenerateComparisonExpression(
                dynamic_cast<BoundComparisonExpression *>(expr), code,
                pipeline_ctx, column_map);

        // case ExpressionClass::BOUND_FUNCTION:
        //     return GenerateFunctionExpression(
        //         dynamic_cast<BoundFunctionExpression *>(expr), code,
        //         pipeline_ctx, column_map);

        // case ExpressionClass::BOUND_OPERATOR:
        //     return GenerateOperatorExpression(
        //         dynamic_cast<BoundOperatorExpression *>(expr), code,
        //         pipeline_ctx, column_map);
        
        case ExpressionClass::BOUND_CONJUNCTION: {
            std::string result_code = "";
            auto bound_conj_expr = (duckdb::BoundConjunctionExpression *)expr;
            for (size_t i = 0; i < bound_conj_expr->children.size(); i++) {
                if (i > 0) {
                    if (bound_conj_expr->type == ExpressionType::CONJUNCTION_AND) {
                        result_code += " && ";
                    } else if (bound_conj_expr->type == ExpressionType::CONJUNCTION_OR) {
                        result_code += " || ";
                    }
                }
                result_code += GenerateExpressionCode(
                    bound_conj_expr->children[i].get(), code, pipeline_ctx,
                    column_map);
            }
            return result_code;
        }
        default: {
            throw NotImplementedException(
                "Unsupported expression type: " +
                std::to_string(static_cast<int>(expr->expression_class)));
        }
    }
}

std::string ExpressionCodeGenerator::GenerateConditionCode(
    Expression *expr, CodeBuilder &code, PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    std::string condition_var =
        GenerateExpressionCode(expr, code, pipeline_ctx, column_map);

    // If the result is not boolean, convert it
    if (expr && expr->return_type.id() != LogicalTypeId::BOOLEAN) {
        // std::string bool_condition = GetUniqueVariableName("bool_condition");
        // code.Add("bool " + bool_condition + " = (" + condition_var + " != 0);");
        // return bool_condition;
        throw NotImplementedException(
            "Condition expressions must return boolean type, but got: " +
            expr->return_type.ToString());
    }

    return condition_var;
}

std::string ExpressionCodeGenerator::GenerateReferenceExpression(
    BoundReferenceExpression *ref_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    if (!ref_expr) {
        throw InvalidInputException(
            "Reference expression cannot be null");
    }

    auto it = column_map.find(ref_expr->index);
    if (it != column_map.end()) {
        // If the column is already mapped, return its name
        return it->second;
    } else {
        throw InvalidInputException(
            "Invalid column index or not materialized: " +
            std::to_string(ref_expr->index));
    }
}

std::string ExpressionCodeGenerator::GenerateBetweenExpression(
    BoundBetweenExpression *between_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    // Generate code for left, lower, and upper expressions
    std::string input_var = GenerateExpressionCode(
        between_expr->input.get(), code, pipeline_ctx, column_map);
    std::string lower_var = GenerateExpressionCode(
        between_expr->lower.get(), code, pipeline_ctx, column_map);
    std::string upper_var = GenerateExpressionCode(
        between_expr->upper.get(), code, pipeline_ctx, column_map);

    // Generate the BETWEEN condition
    std::string lower_compare_str =
        between_expr->lower_inclusive ? " >= " : " > ";
    std::string upper_compare_str =
        between_expr->upper_inclusive ? " <= " : " < ";
    std::string result_code = "(" + input_var + lower_compare_str + lower_var +
                              " && " + input_var + upper_compare_str +
                              upper_var + ")";

    return result_code;
}

std::string ExpressionCodeGenerator::GenerateConstantExpression(
    BoundConstantExpression *const_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    if (!const_expr) {
        throw InvalidInputException("Constant expression cannot be null");
    }

    std::string value_literal = ConvertValueToCUDALiteral(const_expr->value);

    return value_literal;
}

std::string ExpressionCodeGenerator::GenerateComparisonExpression(
    BoundComparisonExpression *comp_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    // Generate code for left and right operands
    std::string left_var = GenerateExpressionCode(comp_expr->left.get(), code,
                                                  pipeline_ctx, column_map);
    std::string right_var = GenerateExpressionCode(
        comp_expr->right.get(), code, pipeline_ctx, column_map);

    // Generate comparison
    std::string operator_str = GetComparisonOperator(comp_expr->type);
    std::string result_code = (left_var + " " + operator_str + " " + right_var);

    return result_code;
}

std::string ExpressionCodeGenerator::GenerateFunctionExpression(
    BoundFunctionExpression *func_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    if (!func_expr) {
        throw InvalidInputException("Function expression cannot be null");
    }

    throw NotImplementedException(
        "Function expressions are not yet implemented in GPU code generation");

    std::string result_code;
    return result_code;
}

std::string ExpressionCodeGenerator::GenerateOperatorExpression(
    BoundOperatorExpression *op_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx,
    std::unordered_map<uint64_t, std::string> &column_map)
{
    if (!op_expr) {
        std::string result_var = GetUniqueVariableName("op_result");
        code.Add("int " + result_var + " = 0; // Invalid operator");
        return result_var;
    }

    code.Add("// Operator expression: " +
             std::to_string(static_cast<int>(op_expr->type)));

    std::string result_var = GetUniqueVariableName("op_result");
    std::string cuda_type = ConvertLogicalTypeToCUDAType(op_expr->return_type);

    if (op_expr->children.size() == 2) {
        // Binary operator
        std::string left_var = GenerateExpressionCode(
            op_expr->children[0].get(), code, pipeline_ctx, column_map);
        std::string right_var = GenerateExpressionCode(
            op_expr->children[1].get(), code, pipeline_ctx, column_map);

        // For now, just implement basic operations that we know exist
        switch (op_expr->type) {
            case ExpressionType::CONJUNCTION_AND:
                code.Add(cuda_type + " " + result_var + " = (" + left_var +
                         " && " + right_var + ");");
                break;
            case ExpressionType::CONJUNCTION_OR:
                code.Add(cuda_type + " " + result_var + " = (" + left_var +
                         " || " + right_var + ");");
                break;
            default:
                code.Add(cuda_type + " " + result_var + " = " + left_var +
                         "; // TODO: implement operator " +
                         std::to_string(static_cast<int>(op_expr->type)));
        }
    }
    else if (op_expr->children.size() == 1) {
        // Unary operator
        std::string operand_var = GenerateExpressionCode(
            op_expr->children[0].get(), code, pipeline_ctx, column_map);

        switch (op_expr->type) {
            case ExpressionType::OPERATOR_NOT:
                code.Add(cuda_type + " " + result_var + " = !(" + operand_var +
                         ");");
                break;
            default:
                code.Add(cuda_type + " " + result_var + " = " + operand_var +
                         "; // TODO: implement unary operator");
        }
    }
    else {
        code.Add(cuda_type + " " + result_var +
                 " = 0; // Invalid operator arity");
    }

    return result_var;
}

// Helper method implementations
std::string ExpressionCodeGenerator::GetUniqueVariableName(
    const std::string &prefix)
{
    return prefix + "_" + std::to_string(expr_counter++);
}

std::string ExpressionCodeGenerator::ConvertLogicalTypeToCUDAType(
    LogicalType type)
{
    switch (type.id()) {
        case LogicalTypeId::BOOLEAN:
            return "bool";
        case LogicalTypeId::TINYINT:
            return "int8_t";
        case LogicalTypeId::SMALLINT:
            return "int16_t";
        case LogicalTypeId::INTEGER:
            return "int32_t";
        case LogicalTypeId::BIGINT:
            return "int64_t";
        case LogicalTypeId::UBIGINT:
            return "uint64_t";
        case LogicalTypeId::FLOAT:
            return "float";
        case LogicalTypeId::DOUBLE:
            return "double";
        case LogicalTypeId::VARCHAR:
            return "char*";
        default:
            return "uint64_t";
    }
}

std::string ExpressionCodeGenerator::ConvertValueToCUDALiteral(
    const Value &value)
{
    switch (value.type().id()) {
        case LogicalTypeId::BOOLEAN:
            return value.GetValue<bool>() ? "true" : "false";
        case LogicalTypeId::TINYINT:
        case LogicalTypeId::SMALLINT:
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
            return std::to_string(value.GetValue<int64_t>());
        case LogicalTypeId::UBIGINT:
            return std::to_string(value.GetValue<uint64_t>());
        case LogicalTypeId::FLOAT:
            return std::to_string(value.GetValue<float>()) + "f";
        case LogicalTypeId::DOUBLE:
            return std::to_string(value.GetValue<double>());
        case LogicalTypeId::VARCHAR:
            return "\"" + value.GetValue<string>() + "\"";
        default:
            throw NotImplementedException(
                "Unsupported value type for CUDA literal: " +
                value.type().ToString());
    }
}

std::string ExpressionCodeGenerator::GetComparisonOperator(ExpressionType type)
{
    switch (type) {
        case ExpressionType::COMPARE_EQUAL:
            return "==";
        case ExpressionType::COMPARE_NOTEQUAL:
            return "!=";
        case ExpressionType::COMPARE_LESSTHAN:
            return "<";
        case ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return "<=";
        case ExpressionType::COMPARE_GREATERTHAN:
            return ">";
        case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return ">=";
        default:
            return "==";  // Default fallback
    }
}

std::string ExpressionCodeGenerator::GetArithmeticOperator(ExpressionType type)
{
    // Most arithmetic operations in DuckDB are handled as functions
    // For basic operators that might still exist:
    switch (type) {
        case ExpressionType::CONJUNCTION_AND:
            return "&&";
        case ExpressionType::CONJUNCTION_OR:
            return "||";
        default:
            return "+";  // Default fallback
    }
}

std::string ExpressionCodeGenerator::GetLogicalOperator(ExpressionType type)
{
    switch (type) {
        case ExpressionType::CONJUNCTION_AND:
            return "&&";
        case ExpressionType::CONJUNCTION_OR:
            return "||";
        default:
            return "&&";  // Default fallback
    }
}

void ExpressionCodeGenerator::EnsureColumnMaterialized(
    const std::string &column_name, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    // Check if column is already materialized
    auto it = pipeline_ctx.column_materialized.find(column_name);
    if (it != pipeline_ctx.column_materialized.end() && it->second) {
        return;  // Already materialized
    }

    // code.Add("// Ensure column " + column_name + " is materialized");

    // TODO: Implement actual lazy materialization logic
    // For now, just mark as materialized
    pipeline_ctx.column_materialized[column_name] = true;

    // code.Add("// Column " + column_name + " materialized");
}

}  // namespace duckdb