#pragma once

#include "planner/expression.hpp"

namespace duckdb {

class BoundPropertyExpression : public Expression {
public:
    BoundPropertyExpression(
        LogicalType type, PropertyKeyID propertyKeyID, idx_t patternElementBindingIdx, idx_t propertyIdx);

    std::string ToString() const override;
    bool Equals(const BaseExpression *other) const override;
    hash_t Hash() const override;
    unique_ptr<Expression> Copy() override;

public:
    PropertyKeyID propertyKeyID;
    idx_t patternElementBindingIdx;
    idx_t propertyIdx;
};

}  // namespace duckdb