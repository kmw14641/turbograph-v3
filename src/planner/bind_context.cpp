#include "planner/bind_context.hpp"
#include <set>
#include "common/exception.hpp"

namespace duckdb {

bool BindContext::hasProperty(idx_t patternElementBindingIdx,
                              PropertyKeyID propertyKeyID) const
{
    return patternElementBindings[patternElementBindingIdx]
               .propertyKeyIDToIdx.find(propertyKeyID) !=
           patternElementBindings[patternElementBindingIdx]
               .propertyKeyIDToIdx.end();
}

idx_t BindContext::addPatternElementBinding(
    const PatternElementGraphletInfo &graphletInfo)
{
    idx_t patternElementBindingIdx = patternElementBindings.size();
    patternElementBindings.push_back(PatternElementBinding());
    patternElementBindings[patternElementBindingIdx].graphletInfo =
        graphletInfo;
    return patternElementBindingIdx;
}

void BindContext::addPartitionOIDs(idx_t patternElementBindingIdx,
                                   const vector<CatalogObjectID> &partitionOIDs)
{
    patternElementBindings[patternElementBindingIdx]
        .graphletInfo.partitionOIDs.insert(
            patternElementBindings[patternElementBindingIdx]
                .graphletInfo.partitionOIDs.end(),
            partitionOIDs.begin(), partitionOIDs.end());
}

void BindContext::addGraphletOIDs(
    idx_t patternElementBindingIdx, const vector<CatalogObjectID> &graphletOIDs,
    const vector<size_t> &numGraphletsPerPartition)
{
    patternElementBindings[patternElementBindingIdx]
        .graphletInfo.graphletOIDs.insert(
            patternElementBindings[patternElementBindingIdx]
                .graphletInfo.graphletOIDs.end(),
            graphletOIDs.begin(), graphletOIDs.end());
    patternElementBindings[patternElementBindingIdx]
        .graphletInfo.numGrahpletsPerPartition.insert(
            patternElementBindings[patternElementBindingIdx]
                .graphletInfo.numGrahpletsPerPartition.end(),
            numGraphletsPerPartition.begin(), numGraphletsPerPartition.end());
}

PropertyRef BindContext::getProperty(idx_t patternElementBindingIdx,
                                     PropertyKeyID propertyKeyID)
{
    auto &binding = patternElementBindings[patternElementBindingIdx];
    auto it = binding.propertyKeyIDToIdx.find(propertyKeyID);
    if (it == binding.propertyKeyIDToIdx.end()) {
        throw BinderException("Property not found in binding.");
    }
    idx_t index = it->second;
    return {index, binding.propertyInfos[index]};
}

void BindContext::replacePartitionOIDs(idx_t patternElementBindingIdx,
                                       vector<CatalogObjectID> &partitionOIDs)
{
    std::swap(patternElementBindings[patternElementBindingIdx]
                  .graphletInfo.partitionOIDs,
              partitionOIDs);
}

void BindContext::markColumnUnused(idx_t patternElementBindingIdx,
                                   idx_t col_idx)
{
    D_ASSERT(patternElementBindings[patternElementBindingIdx]
                 .columnUsageMask.size() > col_idx);
    patternElementBindings[patternElementBindingIdx].columnUsageMask[col_idx] =
        false;
}

bool BindContext::isColumnUsed(idx_t patternElementBindingIdx, idx_t col_idx)
{
    D_ASSERT(patternElementBindings[patternElementBindingIdx]
                 .columnUsageMask.size() > col_idx);
    return patternElementBindings[patternElementBindingIdx]
        .columnUsageMask[col_idx];
}

void BindContext::markColumnUsedInORGroup(idx_t patternElementBindingIdx,
                                          idx_t colIdx, idx_t groupIdx)
{
    D_ASSERT(colIdx < patternElementBindings[patternElementBindingIdx]
                          .columnUsageMask.size());
    patternElementBindings[patternElementBindingIdx].columnUsageMask[colIdx] =
        true;
    patternElementBindings[patternElementBindingIdx]
        .filterColumnUsageByORGroup[groupIdx][colIdx] = true;
}

bool BindContext::isColumnUsedInORGroup(idx_t patternElementBindingIdx,
                                        idx_t colIdx, idx_t groupIdx)
{
    D_ASSERT(colIdx < patternElementBindings[patternElementBindingIdx]
                          .columnUsageMask.size());
    return patternElementBindings[patternElementBindingIdx]
        .filterColumnUsageByORGroup[groupIdx][colIdx];
}

vector<idx_t> BindContext::getReferencedORGroupIDs(
    idx_t patternElementBindingIdx)
{
    std::vector<idx_t> keys;
    for (auto &[key, value] : patternElementBindings[patternElementBindingIdx]
                                  .filterColumnUsageByORGroup) {
        keys.push_back(key);
    }
    return keys;
}
}  // namespace duckdb