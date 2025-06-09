#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "catalog/catalog_wrapper.hpp"
#include "main/database.hpp"
#include "parser/expression/function_expression.hpp"
#include "planner/binder.hpp"
#include "planner/expression/bound_aggregate_expression.hpp"
#include "planner/expression/bound_function_expression.hpp"
#include "planner/expression_binder.hpp"

namespace duckdb {

unique_ptr<Expression> ExpressionBinder::BindExpression(
    FunctionExpression &function)
{
    for (auto &child : function.children) {
        BindChild(child);
    }

    vector<unique_ptr<Expression>> children;
    vector<LogicalType> child_types;
    for (idx_t i = 0; i < function.children.size(); i++) {
        auto &child = (BoundExpression &)*function.children[i];
        D_ASSERT(child.expr);
        children.push_back(std::move(child.expr));
        child_types.push_back(child.expr->return_type);
    }

    auto client = binder->getClient();
    auto &catalog = client->db->GetCatalog();
    auto func =
        catalog.GetEntry(*client, function.schema, function.function_name);
    switch (func->type) {
        case CatalogType::SCALAR_FUNCTION_ENTRY:
            return BindScalarFunction(function, children, child_types);
        case CatalogType::AGGREGATE_FUNCTION_ENTRY:
            return BindAggregateFunction(function, children, child_types);
        default:
            throw BinderException(
                "Function %s is not a scalar or aggregate function",
                function.function_name.c_str());
    }
}

unique_ptr<Expression> ExpressionBinder::BindScalarFunction(
    FunctionExpression &function, vector<unique_ptr<Expression>> &children,
    vector<LogicalType> &child_types)
{
    auto client = binder->getClient();
    duckdb::idx_t func_mdid_id =
        client->db->GetCatalogWrapper().GetScalarFuncMdId(
            *client, function.function_name, child_types);
    ScalarFunctionCatalogEntry *func_catalog_entry;
    idx_t function_idx;
    client->db->GetCatalogWrapper().GetScalarFuncAndIdx(
        *client, func_mdid_id, func_catalog_entry, function_idx);
    auto catalog_function =
        func_catalog_entry->functions.get()->functions[function_idx];
    unique_ptr<duckdb::FunctionData> bind_info;
    if (catalog_function.bind) {
        bind_info = catalog_function.bind(*client, catalog_function, children);
        children.resize(
            std::min(catalog_function.arguments.size(), children.size()));
    }

    return make_unique<BoundFunctionExpression>(
        catalog_function.return_type, catalog_function, std::move(children),
        std::move(bind_info));
}

unique_ptr<Expression> ExpressionBinder::BindAggregateFunction(
    FunctionExpression &function, vector<unique_ptr<Expression>> &children,
    vector<LogicalType> &child_types)
{
    auto client = binder->getClient();
    duckdb::idx_t func_mdid_id = client->db->GetCatalogWrapper().GetAggFuncMdId(
        *client, function.function_name, child_types);
    duckdb::AggregateFunctionCatalogEntry *aggfunc_catalog_entry;
    duckdb::idx_t function_idx;
    client->db->GetCatalogWrapper().GetAggFuncAndIdx(
        *client, func_mdid_id, aggfunc_catalog_entry, function_idx);

    auto catalog_function =
        aggfunc_catalog_entry->functions.get()->functions[function_idx];
    unique_ptr<duckdb::FunctionData> bind_info;
    if (catalog_function.bind) {
        bind_info = catalog_function.bind(*client, catalog_function, children);
        children.resize(
            std::min(catalog_function.arguments.size(), children.size()));
    }

    return make_unique<duckdb::BoundAggregateExpression>(
        std::move(catalog_function), std::move(children), nullptr,
        std::move(bind_info), function.distinct);
}
}  // namespace duckdb