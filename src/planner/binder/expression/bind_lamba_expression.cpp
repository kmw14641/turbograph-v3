#include "planner/expression_binder.hpp"
#include "parser/expression/lambda_expression.hpp"
#include "parser/expression/operator_expression.hpp"
#include "common/logger.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(LambdaExpression &expr)
{
    spdlog::debug("Lambda expression is not implemented");
	OperatorExpression arrow_expr(ExpressionType::ARROW, std::move(expr.lhs), std::move(expr.expr));
	return ExpressionBinder::BindExpression(arrow_expr);
}

}  // namespace duckdb