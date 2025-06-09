#pragma once

#include <utility>
#include "catalog/catalog.hpp"
#include "main/client_context.hpp"
#include "parser/cypher_statement.hpp"
#include "parser/query/graph_pattern/rel_pattern.hpp"
#include "parser/tokens.hpp"
#include "planner/bound_statement.hpp"
#include "planner/bound_tokens.hpp"
#include "planner/bind_context.hpp"

namespace duckdb {

class Binder {
   public:
    explicit Binder(std::shared_ptr<ClientContext> &client)
        : client{client}, bindContext{}
    {}

    std::unique_ptr<BoundStatement> bind(const CypherStatement &statement);

    /*** bind query ***/
    std::unique_ptr<BoundStatement> bindQuery(const RegularQuery &regularQuery);
    std::shared_ptr<NormalizedSingleQuery> bindSingleQuery(
        const SingleQuery &singleQuery);
    std::shared_ptr<NormalizedQueryPart> bindQueryPart(
        const QueryPart &queryPart);

    /*** bind where ***/
    std::shared_ptr<Expression> bindWhereExpression(
        const ParsedExpression &parsedExpression);

    /*** bind reading clause ***/
    std::shared_ptr<BoundReadingClause> bindReadingClause(
        const ReadingClause &readingClause);
    std::shared_ptr<BoundReadingClause> bindMatchClause(
        const ReadingClause &readingClause);
    void rewriteMatchPattern(
        std::shared_ptr<BoundGraphPattern> &boundGraphPattern);

    /*** bind projection clause ***/
    std::shared_ptr<BoundWithClause> bindWithClause(
        const WithClause &withClause);
    std::shared_ptr<BoundReturnClause> bindReturnClause(
        const ReturnClause &returnClause);

    /*** bind updating clause ***/
    std::shared_ptr<BoundUpdatingClause> bindUpdatingClause(
        const UpdatingClause &updatingClause);

    /*** bind graph pattern ***/
    std::shared_ptr<BoundGraphPattern> bindGraphPattern(
        const std::vector<PatternElement> &graphPattern);
    std::shared_ptr<QueryGraph> bindPatternElement(
        const PatternElement &patternElement);
    std::shared_ptr<BoundNodeExpression> bindQueryNode(
        const NodePattern &nodePattern,
        std::shared_ptr<QueryGraph> &queryGraph);
    std::shared_ptr<BoundRelExpression> bindQueryRel(
        const RelPattern &relPattern,
        std::shared_ptr<BoundNodeExpression> &leftNode,
        std::shared_ptr<BoundNodeExpression> &rightNode,
        std::shared_ptr<QueryGraph> &queryGraph);
    std::shared_ptr<Expression> createPath(const std::string &pathName,
                                           const Expressions &children);
    std::shared_ptr<BoundNodeExpression> createQueryNode(
        const NodePattern &nodePattern);
    std::shared_ptr<BoundNodeExpression> createQueryNode(
        const std::string &parsedName,
        const PatternElementGraphletInfo &graphletInfo);
    void bindRelAndConnectedNodePartitionIDs(
        const vector<string> &partitionNames,
        const shared_ptr<BoundNodeExpression> &srcNode,
        const shared_ptr<BoundNodeExpression> &dstNode,
        vector<CatalogObjectID> &partitionOIDs);
    std::pair<Bound, Bound> bindVariableLengthRelBound(
        const RelPattern &relPattern);
    void bindQueryNodeSchema(shared_ptr<BoundNodeExpression> queryNode,
                             bool hasConnectedEdges);
    void bindQueryRelSchema(shared_ptr<BoundRelExpression> queryRel);

    /** scope functions **/
    void addToScope(const std::vector<std::string> &names,
                    const Expressions &exprs);
    void addToScope(const std::string &name, std::shared_ptr<Expression> expr);
    BinderScope saveScope() const;
    void restoreScope(BinderScope prevScope);
    void replaceExpressionInScope(const std::string &oldName,
                                  const std::string &newName,
                                  std::shared_ptr<Expression> expression);

    /** helper functions **/
    std::string getUniqueExpressionName(const std::string &name);

    inline BindContext& getContext() { return bindContext; }
    inline shared_ptr<ClientContext> getClient() { return client; }

   private:
    std::shared_ptr<ClientContext> client;
    BindContext bindContext;
};

}  // namespace duckdb