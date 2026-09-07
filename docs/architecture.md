Architecture

Overview

The Local AI Data Analyst Agent uses a hybrid architecture that combines an LLM for intent planning with deterministic Python logic for control, execution, and final response construction.

The goal is to keep the flexibility of natural-language interaction while reducing unnecessary tool calls, planner instability, repeated execution, and hallucinated final answers.

High-Level Flow

flowchart TD
    A[User Request] --> B[Structured Planner]
    B --> C[Deterministic Plan Minimiser]
    C --> D[Operation List]
    D --> E[Execution Request Builder]
    E --> F[Fresh Single-Tool Strands Agent]
    F --> G[Python Analysis Tool]
    G --> H[Verified Tool Result]
    H --> I[Deterministic Final Composer]
    I --> J[User Response]

    B -. local model .-> K[Ollama / Qwen3 8B]
    G --> L[output/]
    A --> M[logs/agent.log]
    D --> M
    H --> M

1. User Request

The application accepts a natural-language request such as:

Calculate the average Score by Country in data/Customers.csv
and create a bar chart.

The request is sent to the structured planner.

2. Structured Planner

The planner uses the local qwen3:8b model through Ollama.

Instead of returning free text, the model is constrained to return JSON containing an operations array.

Allowed operations are:

inspect_csv
summary_statistics
check_data_quality
group_average
value_counts
find_maximum
find_minimum
calculate_correlation
create_grouped_bar_chart
create_pie_chart
create_line_chart
create_scatter_plot

Conceptual planner output:

{
  "operations": [
    "group_average",
    "create_grouped_bar_chart"
  ]
}

Only operations present in the internal TOOL_REGISTRY are accepted.

Duplicate operations are removed.

Invalid JSON returns an empty plan instead of crashing the application.

3. Deterministic Plan Minimisation

LLMs can over-select tools.

For example, a chart request could theoretically be planned as:

inspect_csv
value_counts
create_pie_chart

The minimiser applies deterministic rules to remove unnecessary operations.

For a chart-only request:

Create a pie chart showing customers by Country.

the final plan becomes:

create_pie_chart

Similarly, check_data_quality is removed unless the user explicitly asks about missing values, nulls, duplicates, or data quality.

The minimiser therefore acts as a control layer between probabilistic planning and deterministic execution.

4. Tool Registry

The application maintains a fixed mapping from operation names to Python tool functions.

Conceptually:

TOOL_REGISTRY = {
    "inspect_csv": inspect_csv,
    "summary_statistics": summary_statistics,
    "check_data_quality": check_data_quality,
    "group_average": group_average,
    "value_counts": value_counts,
    "find_maximum": find_maximum,
    "find_minimum": find_minimum,
    "calculate_correlation": calculate_correlation,
    "create_grouped_bar_chart": create_grouped_bar_chart,
    "create_pie_chart": create_pie_chart,
    "create_line_chart": create_line_chart,
    "create_scatter_plot": create_scatter_plot,
}

This registry forms the boundary between the planner and executable application behaviour.

5. Single-Tool Executor

Each selected operation receives a fresh Strands Agent.

That executor is given exactly one tool.

This is intentionally different from exposing all tools to one large agent.

Benefits:

smaller tool-selection surface

fewer accidental calls

easier debugging

clearer logs

more deterministic testing

reduced cross-tool confusion

6. Tool-Call Guard

The executor uses a Strands hook to track tool usage.

The selected tool is permitted to execute once during the invocation.

If the model attempts to call it again, the second call is cancelled and the executor is instructed to use the first verified result.

This was added to prevent repeated execution after error responses such as missing-column validation.

7. Python Tool Layer

Data Tools

tools/data_tools.py

Responsibilities:

CSV inspection

descriptive statistics

data-quality checks

Analysis Tools

tools/analysis_tools.py

Responsibilities:

grouped averages

value counts

maximum row lookup

minimum row lookup

Pearson correlation

The correlation tool validates:

column existence

numeric data types

sufficient complete observations

constant-column edge cases

Chart Tools

tools/chart_tools.py

Responsibilities:

grouped bar charts

pie charts

line charts

scatter plots

Charts are written into the output/ directory.

The chart layer also validates column existence and numeric requirements where appropriate.

8. Deterministic Final Composer

The final output is not sent back through the LLM for another synthesis step.

Instead, verified operation results are mapped to readable section names and combined directly.

Example:

### Grouped Average Analysis

Country
USA        825.0
Germany    425.0

### Bar Chart

Chart saved to: output\bar_chart_average_by_group.png

This architecture prevents the final model from:

omitting successful operations

inventing values

duplicating analysis

leaking previous generation text

9. Logging

The application writes runtime information to:

logs/agent.log

Logged events include:

Agent started
User request
Selected plan
Executing operation
Completed operation
Operation result
Request execution time
Unhandled exceptions

Logging provides an audit trail for troubleshooting planner and execution behaviour.

10. Testing Strategy

The project uses layered testing.

Tool Unit Tests

Each tool family is tested independently.

test_data_tools.py       9 tests
test_analysis_tools.py  25 tests
test_chart_tools.py     16 tests

Planner Tests

test_planner.py contains 15 tests.

The Ollama planner is mocked so the tests focus on:

planner parsing

valid-operation filtering

duplicate removal

minimisation

malformed output handling

Workflow Tests

test_agent_workflows.py contains 7 tests.

The live Strands executor is mocked to test:

planner
→ minimiser
→ operation dispatch
→ result collection
→ final response

Total

72 automated tests

11. Design Philosophy

The system intentionally separates probabilistic and deterministic responsibilities.

Probabilistic

Qwen3 is used for:

Natural language
→ operation selection

Deterministic

Python controls:

operation validation
plan minimisation
tool registry
execution boundaries
result composition
logging
testing

This separation makes the application more reliable than a design where one LLM is responsible for planning, tool selection, repeated execution, analysis, and final synthesis at the same time.

12. Current Architecture Boundary

The current system is a local CSV analysis agent.

It does not yet provide:

database execution

arbitrary Python execution

browser access

automatic code generation

multi-user authentication

hosted API deployment

Those capabilities should be added as separate, controlled components rather than weakening the current tool boundary.

13. Future Architecture

A possible future production architecture is:

flowchart LR
    A[Web UI] --> B[FastAPI]
    B --> C[Planner Service]
    C --> D[Policy / Plan Validator]
    D --> E[Tool Executor]
    E --> F1[CSV Tools]
    E --> F2[SQL Tools]
    E --> F3[Data Quality Tools]
    E --> F4[Visualisation Tools]
    F1 --> G[Result Store]
    F2 --> G
    F3 --> G
    F4 --> G
    G --> H[Response Composer]
    H --> B

This preserves the same central principle:

LLM for intent, deterministic software for control.