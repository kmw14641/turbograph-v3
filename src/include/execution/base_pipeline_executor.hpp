#ifndef BASE_PIPELINE_EXECUTOR_H
#define BASE_PIPELINE_EXECUTOR_H

#include "execution/cypher_pipeline.hpp"

namespace duckdb {

class ExecutionContext;
class SchemaFlowGraph;

//! Base class for pipeline executors - minimal interface
class BasePipelineExecutor {
   public:
    //! The pipeline to process
    CypherPipeline *pipeline;

	ExecutionContext *context;

	unique_ptr<LocalSinkState> local_sink_state;

   public:
    BasePipelineExecutor() = default;
    virtual ~BasePipelineExecutor() = default;

    //! Fully execute a pipeline with a source and a sink until the source is completely exhausted
    virtual void ExecutePipeline() = 0;

    //! Get pipeline pointer
    virtual CypherPipeline *GetPipeline() const = 0;

    //! Get context pointer
    virtual ExecutionContext *GetContext() const = 0;

    //! Get schema flow graph (if applicable)
    virtual SchemaFlowGraph *GetSchemaFlowGraph() { return nullptr; }

    virtual std::string GetPipelineToString() const = 0;
};

}  // namespace duckdb

#endif  // BASE_PIPELINE_EXECUTOR_H