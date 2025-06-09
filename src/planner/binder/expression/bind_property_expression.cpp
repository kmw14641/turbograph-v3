#include "common/exception.hpp"
#include "parser/expression/property_expression.hpp"
#include "planner/expression_binder.hpp"
#include "planner/expression/bound_rel_expression.hpp"
#include "planner/binder.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    PropertyExpression &expr)
{
    if (expr.children.size() != 1)
        throw BinderException(
            "Property expression must have exactly one child");

    BindChild(expr.children[0]);

    auto &child = (BoundExpression &)*expr.children[0];
    auto child_type = child.expr->return_type;

    if (child_type == LogicalTypeId::NODE) {
        PropertyKeyID propertyKeyID = binder->getPropertyKeyID(expr.property_name);
        auto node_expr = (BoundNodeExpression *)child.expr.get();
        return CreatePropertyExpression(propertyKeyID, node_expr->bindingIdx);
    }
    else if (child_type == LogicalTypeId::REL) {
        PropertyKeyID propertyKeyID = binder->getPropertyKeyID(expr.property_name);
        auto rel_expr = (BoundRelExpression *)child.expr.get();
        return CreatePropertyExpression(propertyKeyID, rel_expr->bindingIdx);
    }
    else {
        throw BinderException("Property expression must have a node or rel child");
        return nullptr;
    }
}

}  // namespace duckdb