#include "planner/expression_binder.hpp"
#include "parser/expression/variable_expression.hpp"
#include "planner/binder.hpp"
#include "planner/binder_scope.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    VariableExpression &expr)
{
    auto variable_name = expr.variable_name;
    auto variable_expr = binder->getContext().getScope().getExpression(variable_name);
    if (variable_expr == nullptr) {
        throw BinderException("Variable not found: " + variable_name);
    }
    return variable_expr->Copy();
}

}  // namespace duckdb