#pragma once

#include <unordered_map>
#include "planner/expression.hpp"
#include "planner/expression/bound_node_expression.hpp"

namespace duckdb {

class BinderScope {
   public:
    BinderScope() = default;

    BinderScope(const BinderScope &other)
        : expressions{other.expressions},
          nameToExprIdx{other.nameToExprIdx}
    {}
    
    bool empty() const { return expressions.empty(); }
    bool contains(const std::string &varName) const
    {
        return nameToExprIdx.count(varName) > 0;
    }
    std::shared_ptr<Expression> getExpression(const std::string &varName) const
    {
        return expressions[nameToExprIdx.at(varName)];
    }

    Expressions getExpressions() const { return expressions; }
    void addExpression(const std::string &varName,
                       std::shared_ptr<Expression> expression)
    {
        nameToExprIdx[varName] = expressions.size();
        expressions.push_back(expression);
    }
    void replaceExpression(const std::string &oldName,
                           const std::string &newName,
                           std::shared_ptr<Expression> expression)
    {
        auto idx = nameToExprIdx.at(oldName);
        expressions[idx] = std::move(expression);
        nameToExprIdx.erase(oldName);
        nameToExprIdx[newName] = idx;
    }

    void addNodeReplacement(std::shared_ptr<BoundNodeExpression> node) {
        nodeReplacement.insert({node->variableName, node});
    }
    bool hasNodeReplacement(const std::string& name) const {
        return nodeReplacement.count(name) > 0;
    }
    std::shared_ptr<BoundNodeExpression> getNodeReplacement(const std::string& name) const {
        return nodeReplacement.at(name);
    }

    void clear()
    {
        expressions.clear();
        nameToExprIdx.clear();
        nodeReplacement.clear();
    }

   private:
    // Expressions in scope. Order should be preserved.
    Expressions expressions;
    std::unordered_map<std::string, idx_t> nameToExprIdx;
    // A node pattern may not always be bound as a node expression, e.g. in the above query,
    // (new_a) is bound as a variable rather than node expression.
    std::unordered_map<std::string, std::shared_ptr<BoundNodeExpression>>
        nodeReplacement;
};

}  // namespace duckdb
