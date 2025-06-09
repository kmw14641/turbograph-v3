#pragma once

#include "planner/expression.hpp"

namespace duckdb {

class BoundPatternElementExpression : public Expression {
public:
	BoundPatternElementExpression(LogicalType type, std::string variableName, idx_t bindingIdx);
	
public:
	// Override functions
	string ToString() const override;

	bool Equals(const BaseExpression *other) const override;
	hash_t Hash() const override;

	unique_ptr<Expression> Copy() override;

public:
    std::string variableName;
	idx_t bindingIdx;
};
} // namespace duckdb