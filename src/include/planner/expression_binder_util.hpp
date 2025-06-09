#pragma once

#include "planner/expression.hpp"
#include "common/exception.hpp"

namespace duckdb {

struct ExpressionBinderUtil {

    static bool isNodeExpression(const std::shared_ptr<Expression>& expr) {
        return expr->type == ExpressionType::NODE;
    }
    static bool isRelExpression(const std::shared_ptr<Expression>& expr) {
        return expr->type == ExpressionType::REL;
    }

    static bool validateDataType(const std::shared_ptr<Expression>& expr, const LogicalType& type) {
        if (expr->return_type != type) {
            throw BinderException("Data type mismatch: " + expr->return_type.ToString() + " != " + type.ToString());
        }
    }
};

}