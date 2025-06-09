#pragma once

#include "planner/binder_scope.hpp"
#include "planner/expression.hpp"

namespace duckdb {

struct PatternElementGraphletInfo {
    std::vector<CatalogObjectID> partitionOIDs;
    std::vector<CatalogObjectID> graphletOIDs;
    std::vector<size_t> numGrahpletsPerPartition;
};

struct PropertyInfo {
    LogicalType type;
    PropertyKeyID propertyKeyID;
    std::unordered_map<GraphletID, idx_t> idxPerGraphlet;
};

struct PatternElementBinding {
    PatternElementGraphletInfo graphletInfo;
    // Index over propertyExprs on property name.
    std::unordered_map<PropertyKeyID, idx_t> propertyKeyIDToIdx;
    // Property expressions with order.
    std::vector<PropertyInfo> propertyInfos;
    // Column pruning
    std::vector<bool> columnUsageMask;
    std::unordered_map<idx_t, vector<bool>> filterColumnUsageByORGroup;
};

struct PropertyRef {
    idx_t index;
    PropertyInfo &info;
};

class BindContext {
    friend class Binder;

   public:
    BindContext() : lastExpressionId{0} {}

    // Graphlet Info
    inline vector<CatalogObjectID> &getPartitionOIDs(
        idx_t patternElementBindingIdx)
    {
        return patternElementBindings[patternElementBindingIdx]
            .graphletInfo.partitionOIDs;
    }
    inline vector<CatalogObjectID> &getGraphletOIDs(
        idx_t patternElementBindingIdx)
    {
        return patternElementBindings[patternElementBindingIdx]
            .graphletInfo.graphletOIDs;
    }
    inline vector<size_t> &getNumGraphletsPerPartition(
        idx_t patternElementBindingIdx)
    {
        return patternElementBindings[patternElementBindingIdx]
            .graphletInfo.numGrahpletsPerPartition;
    }
    PropertyRef getProperty(idx_t patternElementBindingIdx,
                                         PropertyKeyID propertyKeyID);

    inline bool isMultiLabeled(idx_t patternElementBindingIdx)
    {
        return patternElementBindings[patternElementBindingIdx]
                   .graphletInfo.partitionOIDs.size() > 1;
    }
    bool hasProperty(idx_t patternElementBindingIdx,
                     PropertyKeyID propertyKeyID) const;
    inline bool isBounded(idx_t patternElementBindingIdx)
    {
        return patternElementBindings[patternElementBindingIdx]
                   .propertyInfos.size() > 0;
    }

    idx_t addPatternElementBinding(const PatternElementGraphletInfo &graphletInfo);
    void addPartitionOIDs(idx_t patternElementBindingIdx,
                          const vector<CatalogObjectID> &partitionOIDs);
    void addGraphletOIDs(idx_t patternElementBindingIdx,
                         const vector<CatalogObjectID> &graphletOIDs,
                         const vector<size_t> &numGraphletsPerPartition);
    idx_t addProperty(idx_t patternElementBindingIdx,
                      PropertyInfo &propertyInfo);
    void replacePartitionOIDs(idx_t patternElementBindingIdx,
                              vector<CatalogObjectID> &partitionOIDs);

    // Column pruning
    void markColumnUnused(idx_t patternElementBindingIdx, idx_t colIdx);
    bool isColumnUsed(idx_t patternElementBindingIdx, idx_t colIdx);
    void markColumnUsedInORGroup(idx_t patternElementBindingIdx,
                                idx_t colIdx,
                                idx_t groupIdx = 0);
    bool isColumnUsedInORGroup(idx_t patternElementBindingIdx, idx_t colIdx,
                               idx_t groupIdx = 0);
    vector<idx_t> getReferencedORGroupIDs(idx_t patternElementBindingIdx);

   private:
    std::vector<PatternElementBinding> patternElementBindings;
    idx_t lastExpressionId;
    BinderScope scope;
};

}  // namespace duckdb