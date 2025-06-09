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
    unique_ptr<Expression> BindExpression(NamedParameterExpression &expr);
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

    // /* Boolean Expressions */
    // shared_ptr<Expression> bindBooleanExpression(const ParsedExpression& parsedExpression);
    // shared_ptr<Expression> bindBooleanExpression(
    //     ExpressionType expressionType, const Expressions& children);

    // /* Comparison Expressions */
    // shared_ptr<Expression> bindComparisonExpression(const ParsedExpression& parsedExpression);
    // shared_ptr<Expression> bindComparisonExpression(
    //     ExpressionType expressionType, const Expressions& children);

    // /* Null Operator Expressions */
    // std::shared_ptr<Expression> bindNullOperatorExpression(
    //     const ParsedExpression& parsedExpression);
    // std::shared_ptr<Expression> bindNullOperatorExpression(ExpressionType expressionType,
    //     const Expressions& children);

    // /* Property Expressions */
    // Expressions bindPropertyStarExpression(const ParsedExpression& parsedExpression);
    // Expressions bindNodeOrRelPropertyStarExpression(const Expression& child);
    // std::shared_ptr<Expression> bindPropertyExpression(
    //     const ParsedExpression& parsedExpression);
    // std::shared_ptr<Expression> bindNodeOrRelPropertyExpression(const Expression& child,
    //     const std::string& propertyName);

    // /* Function Expressions */
    // std::shared_ptr<Expression> bindFunctionExpression(const ParsedExpression& expr);
    // std::shared_ptr<Expression> bindScalarFunctionExpression(
    //     const ParsedExpression& parsedExpression, const std::string& functionName);
    // std::shared_ptr<Expression> bindScalarFunctionExpression(const Expressions& children,
    //     const std::string& functionName,
    //     std::vector<std::string> optionalArguments = std::vector<std::string>{});
    // std::shared_ptr<Expression> bindRewriteFunctionExpression(const ParsedExpression& expr);
    // std::shared_ptr<Expression> bindAggregateFunctionExpression(
    //     const ParsedExpression& parsedExpression, const std::string& functionName,
    //     bool isDistinct);
    // shared_ptr<Expression> bindInternalIDExpression(const ParsedExpression& parsedExpression);
    // shared_ptr<Expression> bindInternalIDExpression(const Expression& expression);
    // unique_ptr<Expression> createInternalNodeIDExpression(const Expression& node,
    //     std::unordered_map<uint64_t, uint32_t>* propertyIDPerTable);

    // /* Parameter Expression */
    // shared_ptr<Expression> bindParameterExpression(const ParsedExpression& parsedExpression);

    // /* Constant Expression */
    // shared_ptr<Expression> bindConstantExpression(const ParsedExpression& parsedExpression);

    // /* Variable Expression */
    // shared_ptr<Expression> bindVariableExpression(const ParsedExpression& parsedExpression);

    // /* Subquery Expression */
    // shared_ptr<Expression> bindExistentialSubqueryExpression(
    //     const ParsedExpression& parsedExpression);

    // /* Case Expression */
    // shared_ptr<Expression> bindCaseExpression(const ParsedExpression& parsedExpression);

    // /* Cast */
    // std::shared_ptr<Expression> implicitCastIfNecessary(
    //     const std::shared_ptr<Expression>& expression, const LogicalType& targetType);
    // // Use implicitCast to cast to types you have obtained through known implicit casting rules.
    // // Use forceCast to cast to types you have obtained through other means, for example,
    // // through a maxLogicalType function
    // std::shared_ptr<Expression> implicitCast(const std::shared_ptr<Expression>& expression,
    //     const LogicalType& targetType);

   private:
    Binder *binder;
    std::unordered_map<std::string, Value> parameterMap;
    uint64_t currentORGroupID = 0;
};

}  // namespace duckdb