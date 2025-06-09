#include "common/enums/expression_type.hpp"
#include "parser/expression/function_expression.hpp"
#include "parser/expression/operator_expression.hpp"
#include "planner/binder.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression/bound_operator_expression.hpp"
#include "planner/expression/bound_property_expression.hpp"
#include "planner/expression_binder.hpp"

namespace duckdb {

static LogicalType ResolveNotType(OperatorExpression &op,
                                  vector<BoundExpression *> &children)
{
    // NOT expression, cast child to BOOLEAN
    D_ASSERT(children.size() == 1);
    children[0]->expr = BoundCastExpression::AddCastToType(
        move(children[0]->expr), LogicalType::BOOLEAN);
    return LogicalType(LogicalTypeId::BOOLEAN);
}

static LogicalType ResolveInType(OperatorExpression &op,
                                 vector<BoundExpression *> &children)
{
    if (children.empty()) {
        throw InternalException("IN requires at least a single child node");
    }
    // get the maximum type from the children
    LogicalType max_type = children[0]->expr->return_type;
    for (idx_t i = 1; i < children.size(); i++) {
        max_type = LogicalType::MaxLogicalType(max_type,
                                               children[i]->expr->return_type);
    }
    ExpressionBinder::ResolveParameterType(max_type);

    // cast all children to the same type
    for (idx_t i = 0; i < children.size(); i++) {
        children[i]->expr = BoundCastExpression::AddCastToType(
            move(children[i]->expr), max_type);
    }
    // (NOT) IN always returns a boolean
    return LogicalType::BOOLEAN;
}

static LogicalType ResolveOperatorType(OperatorExpression &op,
                                       vector<BoundExpression *> &children)
{
    switch (op.type) {
        case ExpressionType::OPERATOR_IS_NULL:
        case ExpressionType::OPERATOR_IS_NOT_NULL:
            // IS (NOT) NULL always returns a boolean, and does not cast its children
            ExpressionBinder::ResolveParameterType(children[0]->expr);
            return LogicalType::BOOLEAN;
        case ExpressionType::COMPARE_IN:
        case ExpressionType::COMPARE_NOT_IN:
            return ResolveInType(op, children);
        case ExpressionType::OPERATOR_COALESCE: {
            ResolveInType(op, children);
            return children[0]->expr->return_type;
        }
        case ExpressionType::OPERATOR_NOT:
            return ResolveNotType(op, children);
        default:
            throw InternalException(
                "Unrecognized expression type for ResolveOperatorType");
    }
}

unique_ptr<Expression> ExpressionBinder::BindExpression(
    OperatorExpression &expr)
{
    // bind the children of the operator expression
    for (idx_t i = 0; i < expr.children.size(); i++) {
        BindChild(expr.children[i]);
    }

    vector<BoundExpression *> children;
    for (idx_t i = 0; i < expr.children.size(); i++) {
        D_ASSERT(expr.children[i]->expression_class ==
                 ExpressionClass::BOUND_EXPRESSION);
        children.push_back((BoundExpression *)expr.children[i].get());
    }

    // S62 Optimization: set filter column if the operator is IS NOT NULL
    if (expr.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
        for (auto &child : children) {
            if (child->type == ExpressionType::PROPERTY) {
                auto property_expr =
                    (BoundPropertyExpression *)child->expr.get();
                binder->getContext().markColumnUsedInORGroup(
                    property_expr->patternElementBindingIdx,
                    property_expr->propertyIdx, currentORGroupID);
            }
        }
    }

    // now resolve the types
    LogicalType result_type = ResolveOperatorType(expr, children);
    if (expr.type == ExpressionType::OPERATOR_COALESCE) {
        if (children.empty()) {
            throw BinderException("COALESCE needs at least one child");
        }
        if (children.size() == 1) {
            return std::move(children[0]->expr);
        }
    }

    auto result = make_unique<BoundOperatorExpression>(expr.type, result_type);
    for (auto &child : children) {
        result->children.push_back(std::move(child->expr));
    }
    return std::move(result);
}

}  // namespace duckdb