//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/expression_binder.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common/types/value.hpp"
#include "parser/expression/bound_expression.hpp"
#include "parser/parsed_expression.hpp"
#include "parser/tokens.hpp"
#include "planner/expression.hpp"

namespace duckdb {

class Binder;

class ExpressionBinder {
    friend class Binder;

   public:
    ExpressionBinder(Binder *binder) : binder(binder) {}

    // DuckDB originally returns unique_ptr<Expression> (with support of Copy function).
    // However, in schemaless design, copying property expression is too expensive.
    // Thus, we return shared_ptr<Expression> instead.
    void Bind(unique_ptr<ParsedExpression> *expr);
    void BindChild(unique_ptr<ParsedExpression> &expr);
    unique_ptr<Expression> BindExpression(
        unique_ptr<ParsedExpression> *expr_ptr);

    static bool ContainsType(const LogicalType &type, LogicalTypeId target);
    static LogicalType ExchangeType(const LogicalType &type,
                                    LogicalTypeId target, LogicalType new_type);

    static void ResolveParameterType(LogicalType &type);
    static void ResolveParameterType(unique_ptr<Expression> &expr);

   public:
    unique_ptr<Expression> BindExpression(BetweenExpression &expr);
    unique_ptr<Expression> BindExpression(CaseExpression &expr);
    unique_ptr<Expression> BindExpression(CastExpression &expr);
    unique_ptr<Expression> BindExpression(ComparisonExpression &expr);
    unique_ptr<Expression> BindExpression(ConjunctionExpression &expr);
    unique_ptr<Expression> BindExpression(ConstantExpression &expr);
    unique_ptr<Expression> BindExpression(FunctionExpression &expr);
    unique_ptr<Expression> BindExpression(LambdaExpression &expr);
    unique_ptr<Expression> BindExpression(OperatorExpression &expr);
    unique_ptr<Expression> BindExpression(SubqueryExpression &expr);
    unique_ptr<Expression> BindExpression(ParameterExpression &expr);
    unique_ptr<Expression> BindExpression(StarExpression &expr);
    unique_ptr<Expression> BindExpression(PropertyExpression &expr);
    unique_ptr<Expression> BindExpression(VariableExpression &expr);

   public:
    unique_ptr<Expression> CreatePropertyExpression(
        PropertyKeyID propertyKeyID, idx_t patternElementBindingIdx);
    unique_ptr<Expression> CreateInternalIDPropertyExpression(
        idx_t patternElementBindingIdx);

    unique_ptr<Expression> BindScalarFunction(FunctionExpression &expr, vector<unique_ptr<Expression>> &children, vector<LogicalType> &child_types);
    unique_ptr<Expression> BindAggregateFunction(FunctionExpression &expr, vector<unique_ptr<Expression>> &children, vector<LogicalType> &child_types);

   private:
    Binder *binder;
    std::unordered_map<std::string, Value> parameterMap;
    uint64_t currentORGroupID = 0;
};

}  // namespace duckdb