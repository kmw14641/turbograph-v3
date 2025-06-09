#include "planner/expression_binder.hpp"
#include "parser/expression/case_expression.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression/bound_case_expression.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(CaseExpression& expr) {
	// first try to bind the children of the case expression
	for (auto &check : expr.case_checks) {
		BindChild(check.when_expr);
		BindChild(check.then_expr);
	}
	BindChild(expr.else_expr);

	// the children have been successfully resolved
	// figure out the result type of the CASE expression
	auto return_type = ((BoundExpression &)*expr.else_expr).expr->return_type;
	for (auto &check : expr.case_checks) {
		auto &then_expr = (BoundExpression &)*check.then_expr;
		return_type = LogicalType::MaxLogicalType(return_type, then_expr.expr->return_type);
	}
	ExpressionBinder::ResolveParameterType(return_type);

	// bind all the individual components of the CASE statement
	auto result = make_unique<BoundCaseExpression>(return_type);
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &check = expr.case_checks[i];
		auto &when_expr = (BoundExpression &)*check.when_expr;
		auto &then_expr = (BoundExpression &)*check.then_expr;
		BoundCaseCheck result_check;
		result_check.when_expr = BoundCastExpression::AddCastToType(std::move(when_expr.expr), LogicalType::BOOLEAN);
		result_check.then_expr = BoundCastExpression::AddCastToType(std::move(then_expr.expr), return_type);
		result->case_checks.push_back(std::move(result_check));
	}
	auto &else_expr = (BoundExpression &)*expr.else_expr;
	result->else_expr = BoundCastExpression::AddCastToType(std::move(else_expr.expr), return_type);
	return std::move(result);
}

}  // namespace duckdb