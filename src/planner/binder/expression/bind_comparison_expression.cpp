#include "common/types/decimal.hpp"
#include "parser/expression/list.hpp"
#include "planner/binder.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression/bound_comparison_expression.hpp"
#include "planner/expression/bound_pattern_element_expression.hpp"
#include "planner/expression/bound_property_expression.hpp"
#include "planner/expression_binder.hpp"

namespace duckdb {

LogicalType BoundComparisonExpression::BindComparison(LogicalType left_type,
                                                      LogicalType right_type)
{
    auto result_type = LogicalType::MaxLogicalType(left_type, right_type);
    switch (result_type.id()) {
        case LogicalTypeId::DECIMAL: {
            // result is a decimal: we need the maximum width and the maximum scale over width
            vector<LogicalType> argument_types = {left_type, right_type};
            uint8_t max_width = 0, max_scale = 0, max_width_over_scale = 0;
            for (idx_t i = 0; i < argument_types.size(); i++) {
                uint8_t width, scale;
                auto can_convert =
                    argument_types[i].GetDecimalProperties(width, scale);
                if (!can_convert) {
                    return result_type;
                }
                max_width = MaxValue<uint8_t>(width, max_width);
                max_scale = MaxValue<uint8_t>(scale, max_scale);
                max_width_over_scale =
                    MaxValue<uint8_t>(width - scale, max_width_over_scale);
            }
            max_width =
                MaxValue<uint8_t>(max_scale + max_width_over_scale, max_width);
            if (max_width > Decimal::MAX_WIDTH_DECIMAL) {
                // target width does not fit in decimal: truncate the scale (if possible) to try and make it fit
                max_width = Decimal::MAX_WIDTH_DECIMAL;
            }
            return LogicalType::DECIMAL(max_width, max_scale);
        }
        case LogicalTypeId::VARCHAR:
            // for comparison with strings, we prefer to bind to the numeric types
            if (left_type.IsNumeric() ||
                left_type.id() == LogicalTypeId::BOOLEAN) {
                return left_type;
            }
            else if (right_type.IsNumeric() ||
                     right_type.id() == LogicalTypeId::BOOLEAN) {
                return right_type;
            }
            else {
                // else: check if collations are compatible
                auto left_collation = StringType::GetCollation(left_type);
                auto right_collation = StringType::GetCollation(right_type);
                if (!left_collation.empty() && !right_collation.empty() &&
                    left_collation != right_collation) {
                    throw BinderException(
                        "Cannot combine types with different collation!");
                }
            }
            return result_type;
        case LogicalTypeId::UNKNOWN:
            // comparing two prepared statement parameters (e.g. SELECT ?=?)
            // default to VARCHAR
            return LogicalType::VARCHAR;
        default:
            return result_type;
    }
}

unique_ptr<Expression> ExpressionBinder::BindExpression(
    ComparisonExpression &expr)
{
    // first try to bind the children of the case expression
    BindChild(expr.left);
    BindChild(expr.right);

    // the children have been successfully resolved
    auto &left = (BoundExpression &)*expr.left;
    auto &right = (BoundExpression &)*expr.right;
    auto left_type = left.expr->return_type;
    auto right_type = right.expr->return_type;

    // Optimization: if the left and right are both node/rel, we can use the internal ID property
    if ((left_type == LogicalTypeId::NODE &&
         right_type == LogicalTypeId::NODE) ||
        (left_type == LogicalTypeId::REL && right_type == LogicalTypeId::REL)) {
        auto &left_pattern_element =
            (BoundPatternElementExpression &)*left.expr;
        auto &right_pattern_element =
            (BoundPatternElementExpression &)*right.expr;

        auto left_internal_id =
            CreateInternalIDPropertyExpression(left_pattern_element.bindingIdx);
        auto right_internal_id =
            CreateInternalIDPropertyExpression(right_pattern_element.bindingIdx);

        // Update the left and right expression to the internal ID property
        left.expr = std::move(left_internal_id);
        right.expr = std::move(right_internal_id);
        left_type = left.expr->return_type;
        right_type = right.expr->return_type;
    }

    // Mark columns as used for filter
    if (left.expr->GetExpressionClass() == ExpressionClass::PROPERTY) {
        auto &left_property = (BoundPropertyExpression &)*left.expr;
        binder->getContext().markColumnUsedInORGroup(
            left_property.patternElementBindingIdx, left_property.propertyKeyID,
            currentORGroupID);
    }

    if (right.expr->GetExpressionClass() == ExpressionClass::PROPERTY) {
        auto &right_property = (BoundPropertyExpression &)*right.expr;
        binder->getContext().markColumnUsedInORGroup(
            right_property.patternElementBindingIdx,
            right_property.propertyKeyID, currentORGroupID);
    }

    // cast the input types to the same type
    // now obtain the result type of the input types
    auto input_type =
        BoundComparisonExpression::BindComparison(left_type, right_type);
    // add casts (if necessary)
    left.expr =
        BoundCastExpression::AddCastToType(std::move(left.expr), input_type);
    right.expr =
        BoundCastExpression::AddCastToType(std::move(right.expr), input_type);
    if (input_type.id() == LogicalTypeId::VARCHAR) {
        // handle collation
        auto collation = StringType::GetCollation(input_type);
        spdlog::debug("Skip collation push for comparison expression (TODO)");
    }
    // now create the bound comparison expression
    return make_unique<BoundComparisonExpression>(
        expr.type, std::move(left.expr), std::move(right.expr));
}

}  // namespace duckdb