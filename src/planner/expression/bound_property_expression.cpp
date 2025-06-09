#include "planner/expression/bound_property_expression.hpp"
#include "common/types/hash.hpp"
#include <string>

namespace duckdb {

BoundPropertyExpression::BoundPropertyExpression(
    LogicalType type, PropertyKeyID propertyKeyID, idx_t patternElementBindingIdx, idx_t propertyIdx)
    : Expression(ExpressionType::PROPERTY, ExpressionClass::BOUND_PROPERTY, type),
      propertyKeyID(propertyKeyID),
      patternElementBindingIdx(patternElementBindingIdx),
      propertyIdx(propertyIdx)
{}

string BoundPropertyExpression::ToString() const
{
    return std::to_string(patternElementBindingIdx) + "." + std::to_string(propertyIdx);
}

bool BoundPropertyExpression::Equals(const BaseExpression *other) const
{
    if (!Expression::Equals(other)) {
        return false;
    }
    auto other_ = (BoundPropertyExpression *)other;
    if (propertyKeyID != other_->propertyKeyID) {
        return false;
    }
    if (patternElementBindingIdx != other_->patternElementBindingIdx) {
        return false;
    }
    if (propertyIdx != other_->propertyIdx) {
        return false;
    }
    return true;
}

hash_t BoundPropertyExpression::Hash() const
{
    hash_t result = Expression::Hash();
    result = CombineHash(result, duckdb::Hash(propertyKeyID));
    result = CombineHash(result, duckdb::Hash(patternElementBindingIdx));
    result = CombineHash(result, duckdb::Hash(propertyIdx));
    return result;
}

unique_ptr<Expression> BoundPropertyExpression::Copy()
{
    auto copy = make_unique<BoundPropertyExpression>(
        return_type, propertyKeyID, patternElementBindingIdx, propertyIdx
    );
    copy->CopyProperties(*this);
    return std::move(copy);
}

}  // namespace duckdb