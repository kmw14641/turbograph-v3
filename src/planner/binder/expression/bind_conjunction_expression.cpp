#include "planner/expression_binder.hpp"
#include "planner/expression/bound_conjunction_expression.hpp"
#include "parser/expression/conjunction_expression.hpp"
#include "planner/expression/bound_cast_expression.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    ConjunctionExpression &expr)
{
	// first try to bind the children of the case expression
	for (idx_t i = 0; i < expr.children.size(); i++) {
		BindChild(expr.children[i]);
		if (expr.type == ExpressionType::CONJUNCTION_OR) currentORGroupID++;
	}

	// the children have been successfully resolved
	// cast the input types to boolean (if necessary)
	// and construct the bound conjunction expression
	auto result = make_unique<BoundConjunctionExpression>(expr.type);
	for (auto &child_expr : expr.children) {
		auto &child = (BoundExpression &)*child_expr;
		result->children.push_back(BoundCastExpression::AddCastToType(std::move(child.expr), LogicalType::BOOLEAN));
	}
	// now create the bound conjunction expression
    return std::move(result);
}

}