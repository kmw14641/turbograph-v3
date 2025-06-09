#include <set>
#include "catalog/catalog_wrapper.hpp"
#include "common/exception.hpp"
#include "common/logger.hpp"
#include "common/operator/numeric_cast.hpp"
#include "main/database.hpp"
#include "parser/query/graph_pattern/node_pattern.hpp"
#include "parser/query/graph_pattern/pattern_element.hpp"
#include "parser/query/graph_pattern/rel_pattern.hpp"
#include "planner/binder.hpp"
#include "planner/expression/bound_node_expression.hpp"
#include "planner/expression/bound_pattern_element_expression.hpp"
#include "planner/expression_binder_util.hpp"
#include "planner/query/graph_pattern/query_graph.hpp"

namespace duckdb {

// A graph pattern contains node/rel and a set of key-value pairs associated with the variable. We
// bind node/rel as query graph and key-value pairs as a separate collection. This collection is
// interpreted in two different ways.
//    - In MATCH clause, these are additional predicates to WHERE clause
//    - In UPDATE clause, there are properties to set.
// We do not store key-value pairs in query graph primarily because we will merge key-value
// std::pairs with other predicates specified in WHERE clause.
std::shared_ptr<BoundGraphPattern> Binder::bindGraphPattern(
    const std::vector<PatternElement> &graphPattern)
{
    auto queryGraphCollection = std::make_shared<QueryGraphCollection>();
    for (auto &patternElement : graphPattern) {
        queryGraphCollection->addAndMergeQueryGraphIfConnected(
            bindPatternElement(patternElement));
    }
    queryGraphCollection->finalize();
    auto boundGraphPattern = std::make_shared<BoundGraphPattern>();
    boundGraphPattern->queryGraphCollection = queryGraphCollection;
    return boundGraphPattern;
}

std::shared_ptr<QueryGraph> Binder::bindPatternElement(
    const PatternElement &patternElement)
{
    auto queryGraph = std::make_shared<QueryGraph>();
    Expressions nodeAndRels;
    auto leftNode =
        bindQueryNode(*patternElement.getFirstNodePattern(), queryGraph);
    nodeAndRels.push_back(std::static_pointer_cast<Expression>(leftNode));
    for (uint32_t i = 0; i < patternElement.getNumPatternElementChains(); ++i) {
        const auto *patternElementChain =
            patternElement.getPatternElementChain(i);
        auto rightNode =
            bindQueryNode(*patternElementChain->getNodePattern(), queryGraph);
        auto rel = bindQueryRel(*patternElementChain->getRelPattern(), leftNode,
                                rightNode, queryGraph);
        nodeAndRels.push_back(std::static_pointer_cast<Expression>(rel));
        nodeAndRels.push_back(std::static_pointer_cast<Expression>(rightNode));
        leftNode = rightNode;
    }

    if (patternElement.hasPathName()) {
        auto pathName = patternElement.getPathName();
        auto path = createPath(pathName, nodeAndRels);
        addToScope(pathName, path);
        throw BinderException("Path variable is not supported yet");
        // TODO: (jhha) pass path expression to query graph
    }

    bindQueryNodeSchema(queryGraph->getQueryNode(0),
                        patternElement.getNumPatternElementChains() > 0);
    for (uint32_t i = 0; i < patternElement.getNumPatternElementChains(); ++i) {
        bindQueryNodeSchema(queryGraph->getQueryNode(i + 1),
                            patternElement.getNumPatternElementChains() > 0);
        bindQueryRelSchema(queryGraph->getQueryRel(i));
    }

    return queryGraph;
}

std::shared_ptr<Expression> Binder::createPath(const std::string &pathName,
                                               const Expressions &children)
{
    return nullptr;
}

std::shared_ptr<BoundNodeExpression> Binder::bindQueryNode(
    const NodePattern &nodePattern, std::shared_ptr<QueryGraph> &queryGraph)
{
    auto parsedName = nodePattern.getVariableName();
    std::shared_ptr<BoundNodeExpression> queryNode;
    if (bindContext.scope.contains(parsedName)) {
        auto prevVariable = bindContext.scope.getExpression(parsedName);
        if (!ExpressionBinderUtil::isNodeExpression(prevVariable)) {
            // TODO: (jhha)we should support node replacement case
            throw BinderException("Cannot bind " + parsedName +
                                  " as node pattern.");
        }
        else {
            queryNode =
                std::static_pointer_cast<BoundNodeExpression>(prevVariable);
            if (!nodePattern.getLabelNames().empty()) {
                throw BinderException("Cannot bind additional labels to " +
                                      parsedName);
            }
        }
    }
    else {
        queryNode = createQueryNode(nodePattern);
        if (!parsedName.empty()) {
            addToScope(parsedName, queryNode);
        }
    }
    if (nodePattern.getPropertyKeyVals().size() > 0) {
        throw BinderException(
            "Specifying properties for a node is not supported yet");
    }
    queryGraph->addQueryNode(queryNode);
    return queryNode;
}

std::shared_ptr<BoundNodeExpression> Binder::createQueryNode(
    const NodePattern &nodePattern)
{
    auto parsedName = nodePattern.getVariableName();
    PatternElementGraphletInfo graphletInfo;
    // Bind partition OIDs only; graphletOIDs and numGrahpletsPerPartition will be set later
    client->db->GetCatalogWrapper().GetPartitionIDs(
        *client, nodePattern.getLabelNames(), graphletInfo.partitionOIDs,
        GraphComponentType::VERTEX);
    return createQueryNode(parsedName, graphletInfo);
}

std::shared_ptr<BoundNodeExpression> Binder::createQueryNode(
    const std::string &parsedName,
    const PatternElementGraphletInfo &graphletInfo)
{
    idx_t bindingIdx = bindContext.addPatternElementBinding(graphletInfo);
    auto queryNode = std::make_shared<BoundNodeExpression>(
        LogicalType::NODE, getUniqueExpressionName(parsedName), bindingIdx);
    queryNode->SetAlias(parsedName);
    if (!parsedName.empty()) {
        addToScope(parsedName, queryNode);
    }
    return queryNode;
}

void Binder::bindRelAndConnectedNodePartitionIDs(
    const vector<string> &partitionNames,
    const shared_ptr<BoundNodeExpression> &srcNode,
    const shared_ptr<BoundNodeExpression> &dstNode,
    vector<CatalogObjectID> &relPartitionOIDs)
{
    vector<CatalogObjectID> srcPartitionOIDs, dstPartitionOIDs;
    vector<CatalogObjectID> curSrcPartitionOIDS = bindContext.getPartitionOIDs(srcNode->bindingIdx);
    vector<CatalogObjectID> curDstPartitionOIDS = bindContext.getPartitionOIDs(dstNode->bindingIdx);
    if (partitionNames.size() == 0) {
        // get edges that connected with srcNode
        client->db->GetCatalogWrapper().GetConnectedEdgeSubPartitionIDs(
            *client, curSrcPartitionOIDS, relPartitionOIDs, dstPartitionOIDs);

        // prune unnecessary partition IDs
        vector<CatalogObjectID> finalDstPartitionOIDS;
        std::set_intersection(curDstPartitionOIDS.begin(),
                              curDstPartitionOIDS.end(),
                              dstPartitionOIDs.begin(), dstPartitionOIDs.end(),
                              std::back_inserter(finalDstPartitionOIDS));
        bindContext.replacePartitionOIDs(dstNode->bindingIdx, finalDstPartitionOIDS);
    }
    else {
        client->db->GetCatalogWrapper().GetEdgeAndConnectedSrcDstPartitionIDs(
            *client, partitionNames, relPartitionOIDs, srcPartitionOIDs,
            dstPartitionOIDs, GraphComponentType::EDGE);

        // prune unnecessary partition IDs
        vector<CatalogObjectID> finalDstPartitionOIDS;
        std::set_intersection(curDstPartitionOIDS.begin(),
                              curDstPartitionOIDS.end(),
                              dstPartitionOIDs.begin(), dstPartitionOIDs.end(),
                              std::back_inserter(finalDstPartitionOIDS));
        bindContext.replacePartitionOIDs(dstNode->bindingIdx, finalDstPartitionOIDS);

        vector<CatalogObjectID> finalSrcPartitionOIDS;
        std::set_intersection(curSrcPartitionOIDS.begin(),
                              curSrcPartitionOIDS.end(),
                              srcPartitionOIDs.begin(), srcPartitionOIDs.end(),
                              std::back_inserter(finalSrcPartitionOIDS));
        bindContext.replacePartitionOIDs(srcNode->bindingIdx, finalSrcPartitionOIDS);
    }
}

std::shared_ptr<BoundRelExpression> Binder::bindQueryRel(
    const RelPattern &relPattern,
    std::shared_ptr<BoundNodeExpression> &leftNode,
    std::shared_ptr<BoundNodeExpression> &rightNode,
    std::shared_ptr<QueryGraph> &queryGraph)
{
    auto parsedName = relPattern.getVariableName();
    if (bindContext.scope.contains(parsedName)) {
        auto prevVariable = bindContext.scope.getExpression(parsedName);
        ExpressionBinderUtil::validateDataType(prevVariable,
                                               LogicalTypeId::REL);
        throw BinderException(
            "Bind relationship " + parsedName +
            " to relationship with the same name is not supported.");
    }

    // bind src & dst node
    RelDirectionType directionType = RelDirectionType::UNKNOWN;
    std::shared_ptr<BoundNodeExpression> srcNode;
    std::shared_ptr<BoundNodeExpression> dstNode;
    switch (relPattern.getDirection()) {
        case ArrowDirection::LEFT: {
            srcNode = rightNode;
            dstNode = leftNode;
            directionType = RelDirectionType::SINGLE;
        } break;
        case ArrowDirection::RIGHT: {
            srcNode = leftNode;
            dstNode = rightNode;
            directionType = RelDirectionType::SINGLE;
        } break;
        case ArrowDirection::BOTH: {
            // For both direction, left and right will be written with the same label set. So either one
            // being src will be correct.
            srcNode = leftNode;
            dstNode = rightNode;
            directionType = RelDirectionType::BOTH;
        } break;
        default:
            throw BinderException("Invalid direction type");
    }

    if (srcNode->variableName == dstNode->variableName) {
        throw BinderException("Self-loop is not supported: " +
                              srcNode->variableName);
    }

    PatternElementGraphletInfo graphletInfo;
    bindRelAndConnectedNodePartitionIDs(relPattern.getTypeNames(), leftNode,
                                        rightNode, graphletInfo.partitionOIDs);
    client->db->GetCatalogWrapper().GetSubPartitionIDsFromPartitions(
        *client, graphletInfo.partitionOIDs, graphletInfo.graphletOIDs,
        graphletInfo.numGrahpletsPerPartition, GraphComponentType::EDGE);

    auto boundPair = bindVariableLengthRelBound(relPattern);
    idx_t bindingIdx = bindContext.addPatternElementBinding(graphletInfo);
    auto queryRel = make_shared<BoundRelExpression>(
        LogicalType::REL, getUniqueExpressionName(parsedName), bindingIdx,
        srcNode, dstNode, directionType, relPattern.getRelType(),
        boundPair.first, boundPair.second);
    queryRel->SetAlias(parsedName);
    if (!parsedName.empty()) {
        addToScope(parsedName, queryRel);
    }
    queryGraph->addQueryRel(queryRel);
    return queryRel;
}

std::pair<Bound, Bound> Binder::bindVariableLengthRelBound(
    const RelPattern &relPattern)
{
    const auto *recursiveInfo = relPattern.getRecursiveInfo();

    Bound lowerBound = 0;
    Bound upperBound = 0;
    bool strict = true;

    // Parse lowerBound
    if (recursiveInfo->lowerBound.empty()) {
        lowerBound = 1;
    }
    else {
        string_t lowerBound_s(recursiveInfo->lowerBound);
        if (!TryCast::Operation<string_t, uint64_t>(lowerBound_s, lowerBound,
                                                    strict)) {
            throw duckdb::BinderException(
                "Failed to parse lower bound '" + recursiveInfo->lowerBound +
                "' for variable length relationship.");
        }
    }

    // Parse upperBound
    if (recursiveInfo->upperBound.empty()) {
        upperBound = std::numeric_limits<Bound>::max();
    }
    else {
        string_t upperBound_s(recursiveInfo->upperBound);
        if (!TryCast::Operation<string_t, uint64_t>(upperBound_s, upperBound,
                                                    strict)) {
            throw duckdb::BinderException(
                "Failed to parse upper bound '" + recursiveInfo->upperBound +
                "' for variable length relationship.");
        }
    }

    // Validate bounds
    if (lowerBound > upperBound) {
        throw duckdb::BinderException(
            "Lower bound " + std::to_string(lowerBound) +
            " cannot be greater than upper bound " +
            std::to_string(upperBound) + " for variable length relationship.");
    }

    return std::make_pair(lowerBound, upperBound);
}

void Binder::bindQueryNodeSchema(shared_ptr<BoundNodeExpression> queryNode,
                                 bool hasConnectedEdges)
{
    if (!bindContext.isBounded(queryNode->bindingIdx)) {
        vector<CatalogObjectID> partitionOIDs = bindContext.getPartitionOIDs(queryNode->bindingIdx);
        vector<CatalogObjectID> graphletOIDs;
        vector<size_t> numGrahpletsPerPartition;
        client->db->GetCatalogWrapper().GetSubPartitionIDsFromPartitions(
            *client, partitionOIDs, graphletOIDs, numGrahpletsPerPartition,
            GraphComponentType::VERTEX);
        bindContext.addGraphletOIDs(queryNode->bindingIdx, graphletOIDs, numGrahpletsPerPartition);

        duckdb::GraphCatalogEntry *graph_catalog_entry =
            (duckdb::GraphCatalogEntry *)client->db->GetCatalog().GetEntry(
                *client, duckdb::CatalogType::GRAPH_ENTRY, DEFAULT_SCHEMA,
                DEFAULT_GRAPH);

        // Tip: we don't need ID expression APIs. Just treat it as a normal property

        // TODO: (jhha) change type names for readability
        std::shared_ptr<
            unordered_map<duckdb::idx_t, unordered_map<uint64_t, uint32_t>>>
            property_schema_index;
        std::shared_ptr<unordered_map<uint64_t, uint32_t>>
            physical_id_property_schema_index;
        std::shared_ptr<vector<duckdb::idx_t>> universal_schema_ids;
        std::shared_ptr<vector<duckdb::LogicalTypeId>> universal_types_id;
        client->db->GetCatalogWrapper().GetPropertyKeyToPropertySchemaMap(
            *client, partitionOIDs, graphletOIDs, property_schema_index,
            physical_id_property_schema_index, universal_schema_ids,
            universal_types_id);
        {}
    }
}

void Binder::bindQueryRelSchema(shared_ptr<BoundRelExpression> queryRel) {

}


}  // namespace duckdb