#include "common/types/decimal.hpp"
#include "parser/expression/list.hpp"
#include "planner/expression/bound_between_expression.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression/bound_comparison_expression.hpp"
#include "planner/expression_binder.hpp"
#include "common/logger.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(BetweenExpression &expr)
{
    // first try to bind the children of the case expression
    BindChild(expr.input);
    BindChild(expr.lower);
    BindChild(expr.upper);

    auto &input = (BoundExpression &)*expr.input;
    auto &lower = (BoundExpression &)*expr.lower;
    auto &upper = (BoundExpression &)*expr.upper;

    auto input_sql_type = input.expr->return_type;
    auto lower_sql_type = lower.expr->return_type;
    auto upper_sql_type = upper.expr->return_type;

    // cast the input types to the same type
    // now obtain the result type of the input types
    auto input_type = BoundComparisonExpression::BindComparison(input_sql_type,
                                                                lower_sql_type);
    input_type =
        BoundComparisonExpression::BindComparison(input_type, upper_sql_type);
    // add casts (if necessary)
    input.expr = BoundCastExpression::AddCastToType(std::move(input.expr), input_type);
    lower.expr = BoundCastExpression::AddCastToType(std::move(lower.expr), input_type);
    upper.expr = BoundCastExpression::AddCastToType(std::move(upper.expr), input_type);
    if (input_type.id() == LogicalTypeId::VARCHAR) {
        // handle collation
        auto collation = StringType::GetCollation(input_type);
        spdlog::debug("Skip collation push for between expression (TODO)");
    }
    return make_unique<BoundBetweenExpression>(
        std::move(input.expr), std::move(lower.expr), std::move(upper.expr), true, true);
}

}  // namespace duckdb