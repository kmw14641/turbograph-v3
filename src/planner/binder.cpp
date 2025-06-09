#include "planner/binder.hpp"
#include "common/exception.hpp"
#include "parser/cypher_statement.hpp"
#include <set>

namespace duckdb {

std::unique_ptr<BoundStatement> Binder::bind(const CypherStatement& statement) {
    switch (statement.type) {
        case StatementType::SELECT_STATEMENT:
            return bindQuery((const RegularQuery&) statement);
        default: 
            throw BinderException("Unsupported statement type");
            return nullptr;
    }
}


std::shared_ptr<Expression> Binder::bindWhereExpression(const ParsedExpression& parsedExpression) {
    return nullptr;
}

void Binder::addToScope(const std::vector<std::string>& names, const Expressions& exprs) {
    for (size_t i = 0; i < names.size(); ++i) {
        addToScope(names[i], exprs[i]);
    }
}

void Binder::addToScope(const std::string& name, std::shared_ptr<Expression> expr) {
    bindContext.scope.addExpression(name, expr);
}

BinderScope Binder::saveScope() const {
    return bindContext.scope;
}

void Binder::restoreScope(BinderScope prevScope) {
    bindContext.scope = prevScope;
}

void Binder::replaceExpressionInScope(const std::string& oldName, const std::string& newName,
    std::shared_ptr<Expression> expression) {
    bindContext.scope.replaceExpression(oldName, newName, expression);
}

std::string Binder::getUniqueExpressionName(const std::string &name) {
    return "_" + to_string(bindContext.lastExpressionId++) + "_" + name;
}

}