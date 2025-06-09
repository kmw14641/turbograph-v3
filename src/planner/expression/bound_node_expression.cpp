#include "planner/expression/bound_node_expression.hpp"
#include "common/types/hash.hpp"

namespace duckdb {

string BoundNodeExpression::ToString() const
{
    return "Node(" + variableName + ": " +
           std::to_string(bindingIdx) + ")";
}

bool BoundNodeExpression::Equals(const BaseExpression *other) const
{
    if (!BoundPatternElementExpression::Equals(other)) {
        return false;
    }
    return true;
}

hash_t BoundNodeExpression::Hash() const
{
    hash_t result = BoundPatternElementExpression::Hash();
    return result;
}

unique_ptr<Expression> BoundNodeExpression::Copy()
{
    auto copy = make_unique<BoundNodeExpression>(return_type, variableName,
                                                 bindingIdx);
    copy->CopyProperties(*this);
    return std::move(copy);
}

}  // namespace duckdb