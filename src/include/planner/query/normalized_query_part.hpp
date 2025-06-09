#pragma once

#include "planner/query/reading_clause/bound_reading_clause.hpp"
#include "planner/query/return_with_clause/bound_projection_body.hpp"
#include "planner/query/updating_clause/bound_updating_clause.hpp"
#include "planner/expression.hpp"

namespace duckdb {

class NormalizedQueryPart {
public:
    NormalizedQueryPart() = default;

    void addReadingClause(std::shared_ptr<BoundReadingClause> boundReadingClause) {
        readingClauses.push_back(std::move(boundReadingClause));
    }
    bool hasReadingClause() const { return !readingClauses.empty(); }
    uint32_t getNumReadingClause() const { return readingClauses.size(); }
    std::shared_ptr<BoundReadingClause> getReadingClause(uint32_t idx) const { 
        return readingClauses[idx]; 
    }

    void addUpdatingClause(std::shared_ptr<BoundUpdatingClause> boundUpdatingClause) {
        updatingClauses.push_back(std::move(boundUpdatingClause));
    }
    bool hasUpdatingClause() const { return !updatingClauses.empty(); }
    uint32_t getNumUpdatingClause() const { return updatingClauses.size(); }

    void setProjectionBody(std::shared_ptr<BoundProjectionBody> boundProjectionBody) {
        projectionBody = std::move(boundProjectionBody);
    }
    bool hasProjectionBody() const { return projectionBody != nullptr; }
    std::shared_ptr<BoundProjectionBody> getProjectionBody() {
        return projectionBody;
    }

    bool hasProjectionBodyPredicate() const { return projectionBodyPredicate != nullptr; }
    std::shared_ptr<Expression> getProjectionBodyPredicate() const {
        return projectionBodyPredicate;
    }
    void setProjectionBodyPredicate(const std::shared_ptr<Expression>& predicate) {
        projectionBodyPredicate = predicate;
    }

private:
    std::vector<std::shared_ptr<BoundReadingClause>> readingClauses;
    std::vector<std::shared_ptr<BoundUpdatingClause>> updatingClauses;
    std::shared_ptr<BoundProjectionBody> projectionBody;
    std::shared_ptr<Expression> projectionBodyPredicate;
};

}
