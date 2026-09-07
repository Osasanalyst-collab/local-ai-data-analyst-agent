import json

import agent


class FakeOllamaResponse(dict):
    """Simple dictionary-like response matching ollama.chat output."""
    pass


def _mock_chat_with_operations(operations):
    """
    Return a fake ollama.chat function that emits structured planner JSON.
    """
    def fake_chat(*args, **kwargs):
        return FakeOllamaResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "operations": operations,
                        }
                    )
                }
            }
        )

    return fake_chat


def test_count_by_country_routes_to_value_counts(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(["value_counts"]),
    )

    result = agent.create_plan(
        "Count the number of customers in each Country "
        "in data/Customers.csv."
    )

    assert result == ["value_counts"]


def test_pie_chart_removes_unnecessary_operations(monkeypatch):
    """
    Even if the LLM over-selects inspect_csv and value_counts,
    the deterministic minimiser should keep only create_pie_chart.
    """
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "inspect_csv",
                "value_counts",
                "create_pie_chart",
            ]
        ),
    )

    result = agent.create_plan(
        "Create a pie chart showing the number of customers "
        "by Country in data/Customers.csv."
    )

    assert result == ["create_pie_chart"]


def test_line_chart_removes_unnecessary_inspection(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "inspect_csv",
                "create_line_chart",
            ]
        ),
    )

    result = agent.create_plan(
        "Create a line chart showing Score by CustomerID "
        "in data/Customers.csv."
    )

    assert result == ["create_line_chart"]


def test_scatter_plot_removes_unnecessary_inspection(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "inspect_csv",
                "create_scatter_plot",
            ]
        ),
    )

    result = agent.create_plan(
        "Create a scatter plot showing the relationship between "
        "CustomerID and Score in data/Customers.csv."
    )

    assert result == ["create_scatter_plot"]


def test_explicit_inspection_and_summary_are_preserved(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "inspect_csv",
                "summary_statistics",
            ]
        ),
    )

    result = agent.create_plan(
        "Inspect data/Customers.csv and give me summary statistics."
    )

    assert result == [
        "inspect_csv",
        "summary_statistics",
    ]


def test_average_and_bar_chart_preserve_both_operations(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "group_average",
                "create_grouped_bar_chart",
            ]
        ),
    )

    result = agent.create_plan(
        "Calculate the average Score by Country in data/Customers.csv "
        "and create a bar chart showing the result."
    )

    assert result == [
        "group_average",
        "create_grouped_bar_chart",
    ]


def test_unrequested_data_quality_is_removed(monkeypatch):
    """
    This reproduces the earlier Revenue routing problem.
    """
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "group_average",
                "check_data_quality",
            ]
        ),
    )

    result = agent.create_plan(
        "Calculate the average Revenue by Country "
        "in data/Customers.csv."
    )

    assert result == ["group_average"]


def test_explicit_data_quality_is_preserved(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "check_data_quality",
            ]
        ),
    )

    result = agent.create_plan(
        "Check the data quality of data/Customers.csv."
    )

    assert result == ["check_data_quality"]


def test_explicit_correlation_and_scatter_preserve_both(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "calculate_correlation",
                "create_scatter_plot",
            ]
        ),
    )

    result = agent.create_plan(
        "Calculate the correlation between CustomerID and Score "
        "and create a scatter plot."
    )

    assert result == [
        "calculate_correlation",
        "create_scatter_plot",
    ]


def test_duplicate_operations_are_removed(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "find_maximum",
                "find_maximum",
            ]
        ),
    )

    result = agent.create_plan(
        "Find the customer with the highest Score "
        "in data/Customers.csv."
    )

    assert result == ["find_maximum"]


def test_unknown_operation_is_rejected(monkeypatch):
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        _mock_chat_with_operations(
            [
                "invented_tool",
                "find_minimum",
            ]
        ),
    )

    result = agent.create_plan(
        "Find the customer with the lowest Score "
        "in data/Customers.csv."
    )

    assert result == ["find_minimum"]


def test_invalid_json_returns_empty_plan(monkeypatch):
    def fake_chat(*args, **kwargs):
        return {
            "message": {
                "content": "this is not valid json"
            }
        }

    monkeypatch.setattr(
        agent.ollama,
        "chat",
        fake_chat,
    )

    result = agent.create_plan(
        "Count customers by Country."
    )

    assert result == []


def test_missing_message_content_returns_empty_plan(monkeypatch):
    def fake_chat(*args, **kwargs):
        return {}

    monkeypatch.setattr(
        agent.ollama,
        "chat",
        fake_chat,
    )

    result = agent.create_plan(
        "Check the data quality."
    )

    assert result == []


def test_minimize_plan_direct_chart_only():
    result = agent.minimize_plan(
        "Create a pie chart showing customers by Country.",
        [
            "inspect_csv",
            "value_counts",
            "create_pie_chart",
        ],
    )

    assert result == ["create_pie_chart"]


def test_minimize_plan_keeps_explicit_inspection():
    result = agent.minimize_plan(
        "Inspect the dataset and give me summary statistics.",
        [
            "inspect_csv",
            "summary_statistics",
        ],
    )

    assert result == [
        "inspect_csv",
        "summary_statistics",
    ]
