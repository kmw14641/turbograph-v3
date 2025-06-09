#pragma once

#include "planner/expression.hpp"
#include "planner/expression/bound_pattern_element_expression.hpp"
#include "planner/expression/bound_node_expression.hpp"
#include "common/enums/query_rel_type.hpp"
#include "common/constants.hpp"
#include "common/typedefs.hpp"

namespace duckdb {

enum class RelDirectionType : uint8_t {
    SINGLE = 0,
    BOTH = 1,
    UNKNOWN = 2,
};

class BoundRelExpression : public BoundPatternElementExpression {
public:
    BoundRelExpression(LogicalType dataType, std::string variableName, 
        idx_t bindingIdx, std::shared_ptr<BoundNodeExpression> &srcNode,
        std::shared_ptr<BoundNodeExpression> &dstNode, RelDirectionType directionType, QueryRelType relType,
        Bound lowerBound, Bound upperBound);

	string ToString() const override;

	bool Equals(const BaseExpression *other) const override;
	hash_t Hash() const override;

	unique_ptr<Expression> Copy() override;

public:
    std::shared_ptr<BoundNodeExpression> getSrcNode() const { return srcNode; }
    std::string getSrcNodeName() const { return srcNode->variableName; }
    void setDstNode(std::shared_ptr<BoundNodeExpression> node) { dstNode = std::move(node); }
    std::shared_ptr<BoundNodeExpression> getDstNode() const { return dstNode; }
    std::string getDstNodeName() const { return dstNode->variableName; }

    void setLeftNode(std::shared_ptr<BoundNodeExpression> node) { leftNode = std::move(node); }
    std::shared_ptr<BoundNodeExpression> getLeftNode() const { return leftNode; }
    void setRightNode(std::shared_ptr<BoundNodeExpression> node) { rightNode = std::move(node); }
    std::shared_ptr<BoundNodeExpression> getRightNode() const { return rightNode; }

    QueryRelType getRelType() const { return relType; }

    void setDirectionExpr(std::shared_ptr<Expression> expr) { directionExpr = std::move(expr); }
    bool hasDirectionExpr() const { return directionExpr != nullptr; }
    std::shared_ptr<Expression> getDirectionExpr() const { return directionExpr; }
    RelDirectionType getDirectionType() const { return directionType; }

    bool isSelfLoop() const { return *srcNode == *dstNode; }

    inline Bound getLowerBound() const { return lowerBound; }
    inline Bound getUpperBound() const { return upperBound; }

private:
    // Start node if a directed arrow is given. Left node otherwise.
    std::shared_ptr<BoundNodeExpression> srcNode;
    std::shared_ptr<BoundNodeExpression> leftNode;
    // End node if a directed arrow is given. Right node otherwise.
    std::shared_ptr<BoundNodeExpression> dstNode;
    std::shared_ptr<BoundNodeExpression> rightNode;
    // Whether relationship is directed.
    RelDirectionType directionType;
    // Direction expr is nullptr when direction type is SINGLE
    std::shared_ptr<Expression> directionExpr;
    // Whether relationship type is recursive.
    QueryRelType relType;
    // Recursive join information
    Bound lowerBound;
    Bound upperBound;  
};

}