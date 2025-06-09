#include "planner/expression/bound_pattern_element_expression.hpp"
#include "common/types/hash.hpp"
#include "common/exception.hpp"
#include <set>


namespace duckdb {

BoundPatternElementExpression::BoundPatternElementExpression(LogicalType type, std::string _variableName, idx_t _bindingIdx)
		: Expression(ExpressionType::INVALID, ExpressionClass::BOUND_PATTERN_ELEMENT, type),
		  variableName(_variableName), bindingIdx(_bindingIdx)
{  
}

string BoundPatternElementExpression::ToString() const {
    throw NotImplementedException("BoundPatternElementExpression::ToString");
}

bool BoundPatternElementExpression::Equals(const BaseExpression *other) const {
    if (!Expression::Equals(other)) {
        return false;
    }
    auto other_ = (BoundPatternElementExpression *)other;
    if (variableName != other_->variableName) {
        return false;
    }
    if (bindingIdx != other_->bindingIdx) {
        return false;
    }
    return true;
}

hash_t BoundPatternElementExpression::Hash() const {
    hash_t result = Expression::Hash();
    result = CombineHash(result, duckdb::Hash(variableName.c_str()));
    result = CombineHash(result, duckdb::Hash(bindingIdx));
    return result;
}

unique_ptr<Expression> BoundPatternElementExpression::Copy() {
    auto copy = make_unique<BoundPatternElementExpression>(return_type, variableName, bindingIdx);
    copy->CopyProperties(*this);
    return std::move(copy);
}

} // namespace duckdb