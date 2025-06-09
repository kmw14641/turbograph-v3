#include "planner/expression_binder.hpp"
#include "parser/expression/subquery_expression.hpp"
#include "common/logger.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(SubqueryExpression &expr)
{
    throw BinderException("Subquery binding is not implemented");
    return nullptr;
}

}