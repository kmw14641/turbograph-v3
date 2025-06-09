#include "planner/expression_binder.hpp"
#include "parser/expression/star_expression.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(StarExpression &expr)
{
    throw BinderException("Star expression binding is not implemented");
    return nullptr;
}

}