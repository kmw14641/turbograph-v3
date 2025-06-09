#include "parser/expression/constant_expression.hpp"
#include "planner/expression/bound_constant_expression.hpp"
#include "planner/expression_binder.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    ConstantExpression &expr)

{
    return make_unique<BoundConstantExpression>(expr.value);
}

}  // namespace duckdb