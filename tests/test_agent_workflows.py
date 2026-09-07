import json

import pytest

import agent


class DummyLogger:
    """Prevent workflow tests from writing to the real agent log."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


@pytest.fixture
def fake_executor_runtime(monkeypatch):
    """
    Replace the live Strands executor with a deterministic fake.

    The individual data, analysis, and chart tools already have their
    own unit tests. These workflow tests focus on orchestration:
    planner -> minimisation -> executor -> final response.
    """

    calls = []
    result_map = {}

    class FakeExecutorAgent:
        def __init__(self, *args, **kwargs):
            tools = kwargs.get("tools", [])

            if not tools:
                self.operation = None
                return

            selected_tool = tools[0]

            self.operation = next(
                name
                for name, tool_function in agent.TOOL_REGISTRY.items()
                if tool_function is selected_tool
            )

        def __call__(self, prompt):
            if self.operation is None:
                return ""

            calls.append(self.operation)

            return result_map.get(
                self.operation,
                f"Verified result for {self.operation}",
            )

    monkeypatch.setattr(agent, "Agent", FakeExecutorAgent)
    monkeypatch.setattr(agent, "logger", DummyLogger())

    return calls, result_map


def _mock_planner(monkeypatch, operations):
    """Mock the structured Ollama planner response."""

    def fake_chat(*args, **kwargs):
        return {
            "message": {
                "content": json.dumps(
                    {
                        "operations": operations,
                    }
                )
            }
        }

    monkeypatch.setattr(
        agent.ollama,
        "chat",
        fake_chat,
    )


def test_workflow_value_counts(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    _mock_planner(
        monkeypatch,
        ["value_counts"],
    )

    result_map["value_counts"] = (
        "Country\n"
        "USA        3\n"
        "Germany    2"
    )

    agent.process_request(
        "Count the number of customers in each Country "
        "in data/Customers.csv."
    )

    output = capsys.readouterr().out

    assert calls == ["value_counts"]
    assert "Plan:" in output
    assert "- value_counts" in output
    assert "### Value Counts" in output
    assert "USA" in output
    assert "Germany" in output


def test_workflow_inspection_and_summary_statistics(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    _mock_planner(
        monkeypatch,
        [
            "inspect_csv",
            "summary_statistics",
        ],
    )

    result_map["inspect_csv"] = (
        "CSV successfully loaded.\n"
        "Rows: 5\n"
        "Columns: 5"
    )

    result_map["summary_statistics"] = (
        "CustomerID mean 3.0\n"
        "Score mean 625.0"
    )

    agent.process_request(
        "Inspect data/Customers.csv and give me summary statistics."
    )

    output = capsys.readouterr().out

    assert calls == [
        "inspect_csv",
        "summary_statistics",
    ]
    assert "### Dataset Inspection" in output
    assert "Rows: 5" in output
    assert "### Summary Statistics" in output
    assert "Score mean 625.0" in output


def test_workflow_average_and_bar_chart(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    _mock_planner(
        monkeypatch,
        [
            "group_average",
            "create_grouped_bar_chart",
        ],
    )

    result_map["group_average"] = (
        "Country\n"
        "USA        825.0\n"
        "Germany    425.0"
    )

    result_map["create_grouped_bar_chart"] = (
        "Bar chart created successfully.\n"
        "Chart saved to: output\\bar_chart_average_by_group.png"
    )

    agent.process_request(
        "Calculate the average Score by Country in "
        "data/Customers.csv and create a bar chart showing the result."
    )

    output = capsys.readouterr().out

    assert calls == [
        "group_average",
        "create_grouped_bar_chart",
    ]
    assert "### Grouped Average Analysis" in output
    assert "825.0" in output
    assert "425.0" in output
    assert "### Bar Chart" in output
    assert "output\\bar_chart_average_by_group.png" in output


def test_workflow_pie_chart_minimises_overselected_plan(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    # Simulate an LLM that over-selects operations.
    _mock_planner(
        monkeypatch,
        [
            "inspect_csv",
            "value_counts",
            "create_pie_chart",
        ],
    )

    result_map["create_pie_chart"] = (
        "Pie chart created successfully.\n"
        "Chart saved to: output\\pie_chart.png"
    )

    agent.process_request(
        "Create a pie chart showing the number of customers "
        "by Country in data/Customers.csv."
    )

    output = capsys.readouterr().out

    assert calls == ["create_pie_chart"]
    assert "- create_pie_chart" in output
    assert "- inspect_csv" not in output
    assert "- value_counts" not in output
    assert "### Pie Chart" in output
    assert "### Dataset Inspection" not in output
    assert "### Value Counts" not in output


def test_workflow_revenue_error_removes_unrequested_quality_check(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    # Simulate the planner over-selecting check_data_quality.
    _mock_planner(
        monkeypatch,
        [
            "group_average",
            "check_data_quality",
        ],
    )

    result_map["group_average"] = (
        "Column 'Revenue' does not exist in the dataset."
    )

    agent.process_request(
        "Calculate the average Revenue by Country "
        "in data/Customers.csv."
    )

    output = capsys.readouterr().out

    assert calls == ["group_average"]
    assert "- group_average" in output
    assert "- check_data_quality" not in output
    assert "### Grouped Average Analysis" in output
    assert "Column 'Revenue' does not exist" in output
    assert "### Data Quality Analysis" not in output


def test_workflow_correlation_and_scatter_plot(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, result_map = fake_executor_runtime

    _mock_planner(
        monkeypatch,
        [
            "calculate_correlation",
            "create_scatter_plot",
        ],
    )

    result_map["calculate_correlation"] = (
        "Pearson correlation analysis\n"
        "CustomerID vs Score\n"
        "Correlation coefficient: 0.1570\n"
        "Relationship: very weak positive correlation"
    )

    result_map["create_scatter_plot"] = (
        "Scatter plot created successfully.\n"
        "Chart saved to: output\\scatter_plot.png"
    )

    agent.process_request(
        "Calculate the correlation between CustomerID and Score "
        "and create a scatter plot."
    )

    output = capsys.readouterr().out

    assert calls == [
        "calculate_correlation",
        "create_scatter_plot",
    ]
    assert "### Correlation Analysis" in output
    assert "0.1570" in output
    assert "### Scatter Plot" in output
    assert "output\\scatter_plot.png" in output


def test_workflow_invalid_planner_response_is_handled(
    monkeypatch,
    capsys,
    fake_executor_runtime,
):
    calls, _ = fake_executor_runtime

    def fake_chat(*args, **kwargs):
        return {
            "message": {
                "content": "not valid json"
            }
        }

    monkeypatch.setattr(
        agent.ollama,
        "chat",
        fake_chat,
    )

    agent.process_request(
        "Count customers by Country."
    )

    output = capsys.readouterr().out

    assert calls == []
    assert (
        "I could not determine which data-analysis operation "
        "is required for that request."
    ) in output
