

import pandas as pd

from tools.chart_tools import (
    create_grouped_bar_chart,
    create_pie_chart,
    create_line_chart,
    create_scatter_plot,
)


def _make_customer_csv(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5],
            "FirstName": ["Jossef", "Kevin", "Mary", "Mark", "Anna"],
            "LastName": ["Goldberg", "Brown", None, "Schwarz", "Adams"],
            "Country": ["Germany", "USA", "USA", "Germany", "USA"],
            "Score": [350.0, 900.0, 750.0, 500.0, None],
        }
    ).to_csv(csv_path, index=False)

    return csv_path


def test_grouped_bar_chart_creates_png(tmp_path, monkeypatch):
    csv_path = _make_customer_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = create_grouped_bar_chart(
        str(csv_path),
        "Country",
        "Score",
    )

    chart_path = tmp_path / "output" / "bar_chart_average_by_group.png"

    assert "Bar chart created successfully." in result
    assert "Country" in result
    assert "Score" in result
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0


def test_grouped_bar_chart_forces_output_directory(tmp_path, monkeypatch):
    csv_path = _make_customer_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = create_grouped_bar_chart(
        str(csv_path),
        "Country",
        "Score",
        "data/custom_bar.png",
    )

    expected = tmp_path / "output" / "custom_bar.png"

    assert expected.exists()
    assert not (tmp_path / "data" / "custom_bar.png").exists()
    assert "custom_bar.png" in result


def test_grouped_bar_chart_missing_group_column(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_grouped_bar_chart(
        str(csv_path),
        "Region",
        "Score",
    )

    assert "Column 'Region' does not exist" in result
    assert "Available columns:" in result


def test_grouped_bar_chart_missing_value_column(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_grouped_bar_chart(
        str(csv_path),
        "Country",
        "Revenue",
    )

    assert "Column 'Revenue' does not exist" in result
    assert "Available columns:" in result


def test_grouped_bar_chart_requires_numeric_value(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_grouped_bar_chart(
        str(csv_path),
        "Country",
        "FirstName",
    )

    assert "must be numeric" in result


def test_pie_chart_creates_png(tmp_path, monkeypatch):
    csv_path = _make_customer_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = create_pie_chart(
        str(csv_path),
        "Country",
    )

    chart_path = tmp_path / "output" / "pie_chart.png"

    assert "Pie chart created successfully." in result
    assert "Country" in result
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0


def test_pie_chart_missing_column(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_pie_chart(
        str(csv_path),
        "Region",
    )

    assert "Column 'Region' does not exist" in result
    assert "Available columns:" in result


def test_pie_chart_all_missing_values(tmp_path):
    csv_path = tmp_path / "missing_country.csv"

    pd.DataFrame(
        {
            "Country": [None, None, None],
        }
    ).to_csv(csv_path, index=False)

    result = create_pie_chart(
        str(csv_path),
        "Country",
    )

    assert "No valid non-missing data is available" in result


def test_line_chart_creates_png(tmp_path, monkeypatch):
    csv_path = _make_customer_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = create_line_chart(
        str(csv_path),
        "CustomerID",
        "Score",
    )

    chart_path = tmp_path / "output" / "line_chart.png"

    assert "Line chart created successfully." in result
    assert "CustomerID" in result
    assert "Score" in result
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0


def test_line_chart_missing_x_column(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_line_chart(
        str(csv_path),
        "OrderID",
        "Score",
    )

    assert "Column 'OrderID' does not exist" in result


def test_line_chart_requires_numeric_y(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_line_chart(
        str(csv_path),
        "CustomerID",
        "FirstName",
    )

    assert "must be numeric" in result


def test_scatter_plot_creates_png(tmp_path, monkeypatch):
    csv_path = _make_customer_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = create_scatter_plot(
        str(csv_path),
        "CustomerID",
        "Score",
    )

    chart_path = tmp_path / "output" / "scatter_plot.png"

    assert "Scatter plot created successfully." in result
    assert "CustomerID" in result
    assert "Score" in result
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0


def test_scatter_plot_requires_numeric_x(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_scatter_plot(
        str(csv_path),
        "FirstName",
        "Score",
    )

    assert "Column 'FirstName' must be numeric" in result


def test_scatter_plot_requires_numeric_y(tmp_path):
    csv_path = _make_customer_csv(tmp_path)

    result = create_scatter_plot(
        str(csv_path),
        "CustomerID",
        "FirstName",
    )

    assert "Column 'FirstName' must be numeric" in result


def test_chart_tools_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    bar_result = create_grouped_bar_chart(
        str(missing_path),
        "Country",
        "Score",
    )

    pie_result = create_pie_chart(
        str(missing_path),
        "Country",
    )

    line_result = create_line_chart(
        str(missing_path),
        "CustomerID",
        "Score",
    )

    scatter_result = create_scatter_plot(
        str(missing_path),
        "CustomerID",
        "Score",
    )

    assert bar_result == f"CSV file not found: {missing_path}"
    assert pie_result == f"CSV file not found: {missing_path}"
    assert line_result == f"CSV file not found: {missing_path}"
    assert scatter_result == f"CSV file not found: {missing_path}"


def test_chart_tools_empty_csv(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    result = create_pie_chart(
        str(csv_path),
        "Country",
    )

    assert result == f"CSV file is empty: {csv_path}"
