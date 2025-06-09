#include "planner/expression_binder.hpp"
#include "planner/binder.hpp"
#include "planner/expression/bound_cast_expression.hpp"
#include "planner/expression/bound_property_expression.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    unique_ptr<ParsedExpression> *expr_ptr)
{
    // TODO: handle parameter binding (see Kuzu ParsedExpressionVisitor)
    bool is_root = currentORGroupID == 0;
    unique_ptr<Expression> expression;
    auto &expr_ref = **expr_ptr;
    ;
    switch (expr_ref.GetExpressionClass()) {
        case ExpressionClass::BETWEEN:
            expression = BindExpression((BetweenExpression &)expr_ref);
        case ExpressionClass::CASE:
            expression = BindExpression((CaseExpression &)expr_ref);
        case ExpressionClass::CAST:
            expression = BindExpression((CastExpression &)expr_ref);
        case ExpressionClass::COMPARISON:
            expression = BindExpression((ComparisonExpression &)expr_ref);
        case ExpressionClass::CONJUNCTION:
            expression = BindExpression((ConjunctionExpression &)expr_ref);
        case ExpressionClass::CONSTANT:
            expression = BindExpression((ConstantExpression &)expr_ref);
        case ExpressionClass::FUNCTION:
            expression = BindExpression((FunctionExpression &)expr_ref);
        case ExpressionClass::LAMBDA:
            expression = BindExpression((LambdaExpression &)expr_ref);
        case ExpressionClass::OPERATOR:
            expression = BindExpression((OperatorExpression &)expr_ref);
        case ExpressionClass::SUBQUERY:
            expression = BindExpression((SubqueryExpression &)expr_ref);
        case ExpressionClass::PARAMETER:
            expression = BindExpression((ParameterExpression &)expr_ref);
        case ExpressionClass::PROPERTY:
            expression = BindExpression((PropertyExpression &)expr_ref);
        default:
            throw NotImplementedException("Unimplemented expression class");
    }

    if (expr_ref.HasAlias()) {
        expression->SetAlias(expr_ref.GetAlias());
    }
    if (is_root) {
        // TODO: this is a temporal implementation for OR filter
        // improve this implementation
        currentORGroupID = 0;
    }
    return expression;
}

void ExpressionBinder::BindChild(unique_ptr<ParsedExpression> &expr)
{
    if (expr) {
        Bind(&expr);
    }
}

void ExpressionBinder::Bind(unique_ptr<ParsedExpression> *expr)
{
    auto &expression = **expr;
    auto alias = expression.alias;
    if (expression.GetExpressionClass() == ExpressionClass::BOUND_EXPRESSION) {
        return;
    }

    auto result = BindExpression(expr);
    auto be = (BoundExpression *)expr->get();
    be->alias = alias;
    if (!alias.empty()) {
        be->expr->alias = alias;
    }
    return;
}

bool ExpressionBinder::ContainsType(const LogicalType &type,
                                    LogicalTypeId target)
{
    if (type.id() == target) {
        return true;
    }
    switch (type.id()) {
        case LogicalTypeId::STRUCT:
        case LogicalTypeId::MAP: {
            auto child_count = StructType::GetChildCount(type);
            for (idx_t i = 0; i < child_count; i++) {
                if (ContainsType(StructType::GetChildType(type, i), target)) {
                    return true;
                }
            }
            return false;
        }
        case LogicalTypeId::LIST:
            return ContainsType(ListType::GetChildType(type), target);
        default:
            return false;
    }
}

LogicalType ExpressionBinder::ExchangeType(const LogicalType &type,
                                           LogicalTypeId target,
                                           LogicalType new_type)
{
    if (type.id() == target) {
        return new_type;
    }
    switch (type.id()) {
        case LogicalTypeId::STRUCT:
        case LogicalTypeId::MAP: {
            // we make a copy of the child types of the struct here
            auto child_types = StructType::GetChildTypes(type);
            for (auto &child_type : child_types) {
                child_type.second =
                    ExchangeType(child_type.second, target, new_type);
            }
            return type.id() == LogicalTypeId::MAP
                       ? LogicalType::MAP(std::move(child_types))
                       : LogicalType::STRUCT(std::move(child_types));
        }
        case LogicalTypeId::LIST:
            return LogicalType::LIST(
                ExchangeType(ListType::GetChildType(type), target, new_type));
        default:
            return type;
    }
}

void ExpressionBinder::ResolveParameterType(LogicalType &type)
{
    if (type.id() == LogicalTypeId::UNKNOWN) {
        type = LogicalType::VARCHAR;
    }
}

void ExpressionBinder::ResolveParameterType(unique_ptr<Expression> &expr)
{
    if (ContainsType(expr->return_type, LogicalTypeId::UNKNOWN)) {
        auto result_type = ExchangeType(
            expr->return_type, LogicalTypeId::UNKNOWN, LogicalType::VARCHAR);
        expr = BoundCastExpression::AddCastToType(std::move(expr), result_type);
    }
}

unique_ptr<Expression> ExpressionBinder::CreatePropertyExpression(
    PropertyKeyID propertyKeyID, idx_t patternElementBindingIdx)
{
    auto &bindContext = binder->getContext();

    auto propertyRef =
        bindContext.getProperty(patternElementBindingIdx, propertyKeyID);

    return make_unique<BoundPropertyExpression>(
        propertyRef.info.type, propertyKeyID, patternElementBindingIdx,
        propertyRef.index);
}

unique_ptr<Expression> ExpressionBinder::CreateInternalIDPropertyExpression(
    idx_t patternElementBindingIdx)
{
    return CreatePropertyExpression(INTERNAL_ID_PROPERTY_KEY_ID, patternElementBindingIdx);
}

}  // namespace duckdb