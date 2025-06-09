#include "parser/expression/parameter_expression.hpp"

#include "common/exception.hpp"
#include "common/field_writer.hpp"
#include "common/types/hash.hpp"
#include "common/types/string_type.hpp"

namespace duckdb {

ParameterExpression::ParameterExpression()
    : ParsedExpression(ExpressionType::VALUE_PARAMETER, ExpressionClass::PARAMETER), parameter_name("") {
}

string ParameterExpression::ToString() const {
	return "$" + parameter_name;
}

unique_ptr<ParsedExpression> ParameterExpression::Copy() const {
	auto copy = make_unique<ParameterExpression>();
	copy->parameter_name = parameter_name;
	copy->CopyProperties(*this);
	return std::move(copy);
}

bool ParameterExpression::Equals(const ParameterExpression *a, const ParameterExpression *b) {
	return a->parameter_name == b->parameter_name;
}

hash_t ParameterExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	return CombineHash(duckdb::Hash(duckdb::string_t(parameter_name)), result);
}

void ParameterExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(parameter_name);
}

unique_ptr<ParsedExpression> ParameterExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<ParameterExpression>();
	expression->parameter_name = reader.ReadRequired<string>();
	return std::move(expression);
}

} // namespace duckdb
