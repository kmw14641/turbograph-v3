#pragma once

#include "planner/expression.hpp"
#include "planner/expression/bound_pattern_element_expression.hpp"

namespace duckdb {

class BoundNodeExpression : public BoundPatternElementExpression {
public:
    BoundNodeExpression(LogicalType type, std::string variableName, idx_t bindingIdx)
        : BoundPatternElementExpression(type, variableName, bindingIdx) {
    }

public:
    string ToString() const override;
	bool Equals(const BaseExpression *other) const override;
	hash_t Hash() const override;
	unique_ptr<Expression> Copy() override;
};

}