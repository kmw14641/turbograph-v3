#include "planner/expression/bound_rel_expression.hpp"
#include "common/types/hash.hpp"

namespace duckdb {

BoundRelExpression::BoundRelExpression(
    LogicalType dataType, std::string variableName, idx_t bindingIdx,
    std::shared_ptr<BoundNodeExpression> &srcNode,
    std::shared_ptr<BoundNodeExpression> &dstNode,
    RelDirectionType directionType, QueryRelType relType, Bound lowerBound,
    Bound upperBound)
    : BoundPatternElementExpression(dataType, variableName, bindingIdx),
      srcNode(std::move(srcNode)),
      dstNode(std::move(dstNode)),
      directionType(directionType),
      relType(relType),
      lowerBound(lowerBound),
      upperBound(upperBound)
{}

string BoundRelExpression::ToString() const
{
    // TODO: update this to include srcNode and dstNode information
    switch (directionType) {
        case RelDirectionType::SINGLE:
            return "SINGLE(" + variableName + ": " + std::to_string(bindingIdx) +
                   " * " + std::to_string(lowerBound) + "..." +
                   std::to_string(upperBound) + ")";
        case RelDirectionType::BOTH:
            return "BOTH(" + variableName + ": " + std::to_string(bindingIdx) +
                   " * " + std::to_string(lowerBound) + "..." +
                   std::to_string(upperBound) + ")";
        default:
            return "UNKNOWN(" + variableName + ": " + std::to_string(bindingIdx) +
                   " * " + std::to_string(lowerBound) + "..." +
                   std::to_string(upperBound) + ")";
    }
}

bool BoundRelExpression::Equals(const BaseExpression *other) const
{
    if (!BoundPatternElementExpression::Equals(other)) {
        return false;
    }
    auto other_ = (BoundRelExpression *)other;
    if (srcNode != other_->srcNode) {
        return false;
    }
    if (dstNode != other_->dstNode) {
        return false;
    }
    if (directionType != other_->directionType) {
        return false;
    }
    if (relType != other_->relType) {
        return false;
    }
    return true;
}

hash_t BoundRelExpression::Hash() const
{
    hash_t result = BoundPatternElementExpression::Hash();
    result = CombineHash(result, srcNode->Hash());
    result = CombineHash(result, dstNode->Hash());
    result = CombineHash(result, duckdb::Hash((uint8_t)directionType));
    result = CombineHash(result, duckdb::Hash((uint8_t)relType));
    return result;
}

unique_ptr<Expression> BoundRelExpression::Copy()
{
    auto copy = std::make_unique<BoundRelExpression>(
        return_type, variableName, bindingIdx, srcNode, dstNode, directionType,
        relType, lowerBound, upperBound);
    copy->CopyProperties(*this);
    return std::move(copy);
}

}  // namespace duckdb