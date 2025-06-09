#include "planner/expression_binder.hpp"
#include "parser/expression/parameter_expression.hpp"
#include "common/logger.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(ParameterExpression &expr)
{
    throw BinderException("Parameter binding is not implemented");
    return nullptr;
}

}