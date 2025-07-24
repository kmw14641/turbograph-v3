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
    const std::string &result_var_prefix)
{
    if (!expr) {
        std::string result_var = GetUniqueVariableName(result_var_prefix);
        code.Add("bool " + result_var + " = true; // Null expression");
        return result_var;
    }

    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF:
            return GenerateReferenceExpression(
                dynamic_cast<BoundReferenceExpression *>(expr), code,
                pipeline_ctx);

        case ExpressionClass::BOUND_CONSTANT:
            return GenerateConstantExpression(
                dynamic_cast<BoundConstantExpression *>(expr), code,
                pipeline_ctx);

        case ExpressionClass::BOUND_COMPARISON:
            return GenerateComparisonExpression(
                dynamic_cast<BoundComparisonExpression *>(expr), code,
                pipeline_ctx);

        case ExpressionClass::BOUND_FUNCTION:
            return GenerateFunctionExpression(
                dynamic_cast<BoundFunctionExpression *>(expr), code,
                pipeline_ctx);

        case ExpressionClass::BOUND_OPERATOR:
            return GenerateOperatorExpression(
                dynamic_cast<BoundOperatorExpression *>(expr), code,
                pipeline_ctx);
        
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
                    result_var_prefix);
            }
            return result_code;
        }

        default: {
            std::string result_var = GetUniqueVariableName(result_var_prefix);
            code.Add("// Unsupported expression type: " +
                     std::to_string(static_cast<int>(expr->expression_class)));
            code.Add(ConvertLogicalTypeToCUDAType(expr->return_type) + " " +
                     result_var + " = 0; // TODO: implement");
            return result_var;
        }
    }
}

std::string ExpressionCodeGenerator::GenerateConditionCode(
    Expression *expr, CodeBuilder &code, PipelineContext &pipeline_ctx)
{
    // code.Add("// Generate condition from expression");
    std::string condition_var =
        GenerateExpressionCode(expr, code, pipeline_ctx, "condition");

    // If the result is not boolean, convert it
    if (expr && expr->return_type.id() != LogicalTypeId::BOOLEAN) {
        std::string bool_condition = GetUniqueVariableName("bool_condition");
        code.Add("bool " + bool_condition + " = (" + condition_var + " != 0);");
        return bool_condition;
    }

    return condition_var;
}

std::string ExpressionCodeGenerator::GenerateReferenceExpression(
    BoundReferenceExpression *ref_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    if (!ref_expr) {
        std::string result_var = GetUniqueVariableName("ref_result");
        code.Add("int " + result_var + " = 0; // Invalid reference");
        return result_var;
    }

    // code.Add("// Reference expression (index: " +
    //          std::to_string(ref_expr->index) + ")");

    // TODO: Implement proper column resolution based on index
    // For now, create a placeholder
    if (ref_expr->index < pipeline_ctx.input_column_names.size()) {
        std::string column_name =
            pipeline_ctx.input_column_names[ref_expr->index];

        // Ensure column is materialized
        EnsureColumnMaterialized(column_name, code, pipeline_ctx);

        std::string result_var = GetUniqueVariableName("ref_result");
        std::string cuda_type =
            ConvertLogicalTypeToCUDAType(ref_expr->return_type);

        code.Add(cuda_type + " " + result_var + " = " + column_name +
                 "_ptr[i];");
        return result_var;
    }
    else {
        std::string result_var = GetUniqueVariableName("ref_result");
        code.Add("int " + result_var + " = 0; // Invalid column index");
        return result_var;
    }
}

std::string ExpressionCodeGenerator::GenerateConstantExpression(
    BoundConstantExpression *const_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    if (!const_expr) {
        std::string result_var = GetUniqueVariableName("const_result");
        code.Add("int " + result_var + " = 0; // Invalid constant");
        return result_var;
    }

    code.Add("// Constant expression");
    std::string result_var = GetUniqueVariableName("const_result");
    std::string cuda_type =
        ConvertLogicalTypeToCUDAType(const_expr->return_type);
    std::string value_literal = ConvertValueToCUDALiteral(const_expr->value);

    code.Add(cuda_type + " " + result_var + " = " + value_literal + ";");
    return result_var;
}

std::string ExpressionCodeGenerator::GenerateComparisonExpression(
    BoundComparisonExpression *comp_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    if (!comp_expr || !comp_expr->left || !comp_expr->right) {
        std::string result_var = GetUniqueVariableName("comp_result");
        code.Add("bool " + result_var + " = true; // Invalid comparison");
        return result_var;
    }

    // code.Add("// Comparison expression: " +
    //          GetComparisonOperator(comp_expr->type));

    // Generate code for left and right operands
    std::string left_var = GenerateExpressionCode(comp_expr->left.get(), code,
                                                  pipeline_ctx, "left_operand");
    std::string right_var = GenerateExpressionCode(
        comp_expr->right.get(), code, pipeline_ctx, "right_operand");

    // Generate comparison
    std::string result_var = GetUniqueVariableName("comp_result");
    std::string operator_str = GetComparisonOperator(comp_expr->type);

    code.Add("bool " + result_var + " = (" + left_var + " " + operator_str +
             " " + right_var + ");");
    return result_var;
}

std::string ExpressionCodeGenerator::GenerateFunctionExpression(
    BoundFunctionExpression *func_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    if (!func_expr) {
        std::string result_var = GetUniqueVariableName("func_result");
        code.Add("int " + result_var + " = 0; // Invalid function");
        return result_var;
    }

    code.Add("// Function expression: " + func_expr->function.name);

    // TODO: Implement specific function handling
    std::string result_var = GetUniqueVariableName("func_result");
    std::string cuda_type =
        ConvertLogicalTypeToCUDAType(func_expr->return_type);

    if (func_expr->function.name == "abs" && func_expr->children.size() == 1) {
        std::string arg_var = GenerateExpressionCode(
            func_expr->children[0].get(), code, pipeline_ctx, "func_arg");
        code.Add(cuda_type + " " + result_var + " = abs(" + arg_var + ");");
    }
    else if (func_expr->function.name == "sqrt" &&
             func_expr->children.size() == 1) {
        std::string arg_var = GenerateExpressionCode(
            func_expr->children[0].get(), code, pipeline_ctx, "func_arg");
        code.Add(cuda_type + " " + result_var + " = sqrt(" + arg_var + ");");
    }
    else {
        // Generate arguments
        std::vector<std::string> arg_vars;
        for (size_t i = 0; i < func_expr->children.size(); i++) {
            std::string arg_var = GenerateExpressionCode(
                func_expr->children[i].get(), code, pipeline_ctx,
                "func_arg" + std::to_string(i));
            arg_vars.push_back(arg_var);
        }

        code.Add(cuda_type + " " + result_var +
                 " = 0; // TODO: implement function " +
                 func_expr->function.name);
    }

    return result_var;
}

std::string ExpressionCodeGenerator::GenerateOperatorExpression(
    BoundOperatorExpression *op_expr, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
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
            op_expr->children[0].get(), code, pipeline_ctx, "left_op");
        std::string right_var = GenerateExpressionCode(
            op_expr->children[1].get(), code, pipeline_ctx, "right_op");

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
            op_expr->children[0].get(), code, pipeline_ctx, "unary_op");

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
            return "0";
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