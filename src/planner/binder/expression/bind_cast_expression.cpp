#include "common/exception.hpp"
#include "parser/expression/cast_expression.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression_binder.hpp"
#include "planner/expression/bound_parameter_expression.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(CastExpression &expr)
{
    // first try to bind the child of the cast expression
    Bind(&expr.child);

    static const std::unordered_set<LogicalTypeId> unsupported_cast_types = {
        LogicalTypeId::LIST, LogicalTypeId::STRUCT, LogicalTypeId::MAP};

    if (unsupported_cast_types.find(expr.cast_type.id()) !=
        unsupported_cast_types.end()) {
        throw BinderException("Cast to type '" +
                              LogicalTypeIdToString(expr.cast_type.id()) +
                              "' is not supported.");
    }

    // the children have been successfully resolved
    auto &child = (BoundExpression &)*expr.child;
    if (expr.try_cast) {
        if (child.expr->return_type == expr.cast_type) {
            // no cast required: type matches
            return std::move(child.expr);
        }
        child.expr = make_unique<BoundCastExpression>(std::move(child.expr),
                                                      expr.cast_type, true);
    }
    else {
        if (child.expr->type == ExpressionType::VALUE_PARAMETER) {
            auto &parameter = (BoundParameterExpression &)*child.expr;
            // parameter: move types into the parameter expression itself
            parameter.return_type = expr.cast_type;
        }
        else {
            // otherwise add a cast to the target type
            child.expr = BoundCastExpression::AddCastToType(std::move(child.expr),
                                                            expr.cast_type);
        }
    }
    return std::move(child.expr);
}

}  // namespace duckdb