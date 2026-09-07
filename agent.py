import json
import logging
import os
import time
import ollama

from threading import Lock

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.hooks import (
    HookProvider,
    HookRegistry,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
)

from tools.data_tools import (
    inspect_csv,
    summary_statistics,
    check_data_quality,
)

from tools.analysis_tools import (
    group_average,
    value_counts,
    find_maximum,
    find_minimum,
    calculate_correlation,
)

from tools.chart_tools import (
    create_grouped_bar_chart,
    create_pie_chart,
    create_line_chart,
    create_scatter_plot,
)


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_analyst_agent")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(file_handler)


# ============================================================
# MODEL
# ============================================================

model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:8b",
    additional_args={"think": False},
)


# ============================================================
# TOOL REGISTRY
# ============================================================

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


# ============================================================
# PLANNER
# ============================================================
PLANNER_SYSTEM_PROMPT = """
You are the planner for a Data Analyst AI Agent.

Choose the minimum operations required to satisfy the user's
explicit request.

Available operations:

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

Routing rules:

- inspect, preview, view, describe, or show dataset structure -> inspect_csv
- summary or descriptive statistics -> summary_statistics
- missing values, null values, duplicates, or data quality -> check_data_quality
- average or mean grouped by a category -> group_average
- count or frequency by category -> value_counts
- highest, largest, or maximum -> find_maximum
- lowest, smallest, or minimum -> find_minimum
- correlation or numeric relationship analysis -> calculate_correlation
- bar chart -> create_grouped_bar_chart
- pie chart -> create_pie_chart
- line chart -> create_line_chart
- scatter plot -> create_scatter_plot

Do not add inspect_csv unless inspection is explicitly requested.

Do not add a chart unless a chart, graph, plot, or visualisation
is explicitly requested.

For chart-only requests, select only the requested chart operation
unless another analysis operation is explicitly requested.

If the user explicitly requests multiple tasks, include each
required operation.

Select only the minimum operations required.
"""


PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "inspect_csv",
                    "summary_statistics",
                    "check_data_quality",
                    "group_average",
                    "value_counts",
                    "find_maximum",
                    "find_minimum",
                    "calculate_correlation",
                    "create_grouped_bar_chart",
                    "create_pie_chart",
                    "create_line_chart",
                    "create_scatter_plot",
                ],
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}



# ============================================================
# PLAN VALIDATION / MINIMISATION
# ============================================================

def minimize_plan(user_request, operations):
    """
    Apply deterministic safeguards after the LLM planner.

    The structured planner chooses valid operations, while this
    function removes unnecessary operations that were not explicitly
    requested by the user.
    """

    request = user_request.lower().strip()

    # --------------------------------------------------------
    # 1. Remove inspect_csv unless inspection was explicitly asked
    # --------------------------------------------------------

    inspection_phrases = [
        "inspect",
        "preview",
        "view the dataset",
        "show the dataset",
        "describe the dataset",
        "describe data/",
        "show the structure",
        "show the columns",
    ]

    inspection_requested = any(
        phrase in request
        for phrase in inspection_phrases
    )

    if not inspection_requested:
        operations = [
            operation
            for operation in operations
            if operation != "inspect_csv"
        ]

    # --------------------------------------------------------
    # Remove check_data_quality unless data quality was
    # explicitly requested
    # --------------------------------------------------------

    data_quality_phrases = [
        "data quality",
        "missing values",
        "missing value",
        "null values",
        "null value",
        "duplicates",
        "duplicate rows",
        "duplicate row",
    ]

    data_quality_requested = any(
        phrase in request
        for phrase in data_quality_phrases
    )

    if not data_quality_requested:
        operations = [
            operation
            for operation in operations
            if operation != "check_data_quality"
        ]

    # --------------------------------------------------------
    # 2. Detect an explicitly requested chart type
    # --------------------------------------------------------

    chart_mapping = {
        "pie chart": "create_pie_chart",
        "line chart": "create_line_chart",
        "scatter plot": "create_scatter_plot",
        "bar chart": "create_grouped_bar_chart",
    }

    requested_chart = None

    for phrase, operation in chart_mapping.items():
        if phrase in request:
            requested_chart = operation
            break

    # --------------------------------------------------------
    # 3. For direct chart-only requests, keep only that chart tool
    # --------------------------------------------------------
    #
    # Example:
    # "Create a pie chart showing customers by Country"
    #
    # should become:
    # ["create_pie_chart"]
    #
    # rather than:
    # ["inspect_csv", "value_counts", "create_pie_chart"]
    # --------------------------------------------------------

    if requested_chart:
        chart_only_starts = (
            "create",
            "generate",
            "make",
            "plot",
            "draw",
        )

        chart_only_request = request.startswith(
            chart_only_starts
        )

        explicit_analysis_phrases = [
            "calculate the correlation",
            "calculate correlation",
            "calculate the average",
            "calculate average",
            "calculate the mean",
            "calculate mean",
            "find the highest",
            "find the maximum",
            "find the lowest",
            "find the minimum",
            "check the data quality",
            "check data quality",
            "summary statistics",
            "descriptive statistics",
            "inspect",
            "preview",
        ]

        analysis_requested = any(
            phrase in request
            for phrase in explicit_analysis_phrases
        )

        if chart_only_request and not analysis_requested:
            return [requested_chart]

    return operations


# ============================================================
# CREATE PLAN
# ============================================================

def create_plan(user_request):

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_request,
            },
        ],
        format=PLANNER_SCHEMA,
        think=False,
    )

    try:
        planner_data = json.loads(
            response["message"]["content"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

    operations = []

    for operation in planner_data.get("operations", []):
        if (
            operation in TOOL_REGISTRY
            and operation not in operations
        ):
            operations.append(operation)

    return minimize_plan(
        user_request,
        operations,
    )


# ============================================================
# EXECUTION REQUEST
# ============================================================

def create_execution_request(user_request, operation_name):

    return f"""
Original user request:

{user_request}

Your assigned operation:

{operation_name}

Perform ONLY your assigned operation.

Ignore other operations in the original request.

Use your available tool.

Use the exact file path and column names supplied by the user.

Report only the result of your assigned operation.

Do not claim another operation was completed.
"""


# ============================================================
# TOOL-CALL SAFEGUARD
# ============================================================

class LimitToolCounts(HookProvider):
    """
    Limit how many times each tool may run during one agent invocation.

    Each specialised executor in this project receives exactly one tool,
    so the configured maximum is one successful invocation per request.
    """

    def __init__(self, max_tool_counts):
        self.max_tool_counts = max_tool_counts
        self.tool_counts = {}
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(
            BeforeInvocationEvent,
            self.reset_counts,
        )
        registry.add_callback(
            BeforeToolCallEvent,
            self.intercept_tool,
        )

    def reset_counts(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            self.tool_counts = {}

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]

        with self._lock:
            current_count = self.tool_counts.get(tool_name, 0) + 1
            self.tool_counts[tool_name] = current_count

            max_count = self.max_tool_counts.get(tool_name)

        if (
            max_count is not None
            and current_count > max_count
        ):
            event.cancel_tool = (
                f"Tool '{tool_name}' has already been executed once. "
                "Do not call it again. Use the first tool result and "
                "provide your final response now."
            )


# ============================================================
# EXECUTE PLAN
# ============================================================

def execute_plan(user_request, operations):

    results = []

    for operation_name in operations:

        logger.info("Executing operation: %s", operation_name)

        tool_function = TOOL_REGISTRY[operation_name]

        tool_limit = LimitToolCounts(
            max_tool_counts={
                operation_name: 1,
            }
        )

        executor = Agent(
            model=model,
            tools=[tool_function],
            hooks=[tool_limit],
            system_prompt=f"""
You are a specialised Data Analyst tool executor.

You have exactly ONE available tool:

{operation_name}

Perform only the operation assigned to you.

Use the exact file path and column names from the user's request.

You MUST use your available tool exactly once.

Never call the tool more than once during this request.

If the tool returns an error or validation message, that is still
the verified result. Report it directly and do not retry the tool.

Do not calculate or invent dataset results yourself.

Do not discuss operations outside your assigned responsibility.

After the single tool execution, report only the result of your
assigned operation.
""",
        )

        focused_request = create_execution_request(
            user_request,
            operation_name,
        )

        response = executor(focused_request)

        result_text = str(response)

        logger.info(
            "Completed operation: %s",
            operation_name,
        )
        logger.info(
            "Operation result [%s]: %s",
            operation_name,
            result_text.replace("\n", " | "),
        )

        results.append(
            {
                "operation": operation_name,
                "result": result_text,
            }
        )

    return results


# ============================================================
# FINAL SYNTHESIS
# ============================================================

def create_final_response(user_request, results):

    if not results:
        final_text = "No verified analysis results were produced."
        print(final_text)
        return final_text

    operation_titles = {
        "inspect_csv": "Dataset Inspection",
        "summary_statistics": "Summary Statistics",
        "check_data_quality": "Data Quality Analysis",
        "group_average": "Grouped Average Analysis",
        "value_counts": "Value Counts",
        "find_maximum": "Maximum Value Analysis",
        "find_minimum": "Minimum Value Analysis",
        "calculate_correlation": "Correlation Analysis",
        "create_grouped_bar_chart": "Bar Chart",
        "create_pie_chart": "Pie Chart",
        "create_line_chart": "Line Chart",
        "create_scatter_plot": "Scatter Plot",
    }

    sections = []

    for item in results:

        operation = item["operation"]
        result = item["result"].strip()

        title = operation_titles.get(
            operation,
            operation.replace("_", " ").title()
        )

        sections.append(
            f"### {title}\n\n{result}"
        )

    final_text = "\n\n".join(sections)

    print(final_text)

    return final_text

# ============================================================
# PROCESS USER REQUEST
# ============================================================

def process_request(user_request):

    start_time = time.perf_counter()

    logger.info("User request: %s", user_request)

    try:
        operations = create_plan(user_request)

        logger.info("Selected plan: %s", operations)

        if not operations:
            message = (
                "I could not determine which data-analysis "
                "operation is required for that request."
            )

            logger.warning(
                "No valid operations selected for request: %s",
                user_request,
            )

            print(f"\nAgent: {message}")
            return

        print("\nPlan:")
        for operation in operations:
            print(f"  - {operation}")

        print()

        results = execute_plan(
            user_request,
            operations,
        )

        print("\nAgent:")
        create_final_response(
            user_request,
            results,
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Request completed successfully in %.2f seconds",
            elapsed,
        )

    except Exception:
        elapsed = time.perf_counter() - start_time

        logger.exception(
            "Request failed after %.2f seconds",
            elapsed,
        )

        raise


# ============================================================
# INTERACTIVE CHAT
# ============================================================

def main():

    logger.info("Agent started")

    print("========================================")
    print("     Local Data Analyst AI Agent")
    print("========================================")

    print("\nModel: Qwen3 8B")
    print("Framework: Strands Agents")
    print("Runtime: Ollama")

    print("\nType 'exit' to close the agent.")

    while True:

        try:
            user_request = input("\nYou: ").strip()

            if not user_request:
                continue

            if user_request.lower() in {
                "exit",
                "quit",
                "q",
            }:
                logger.info("Agent stopped by user")
                print("\nAgent stopped.")
                break

            process_request(user_request)

        except KeyboardInterrupt:
            logger.info("Agent stopped by keyboard interrupt")
            print("\n\nAgent stopped.")
            break

        except Exception as error:
            logger.exception("Unhandled error in interactive loop")
            print(f"\nError: {error}")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()