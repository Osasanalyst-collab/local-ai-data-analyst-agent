import pandas as pd

from tools.analysis_tools import (
    group_average,
    value_counts,
    find_maximum,
    find_minimum,
    calculate_correlation,
)


def test_group_average_valid_file(tmp_path):
    csv_path = tmp_path / "customers.csv"

    df = pd.DataFrame(
        {
            "Country": ["USA", "USA", "Germany", "Germany"],
            "Score": [900.0, 750.0, 350.0, 500.0],
        }
    )
    df.to_csv(csv_path, index=False)

    result = group_average(
        str(csv_path),
        "Country",
        "Score",
    )

    assert "USA" in result
    assert "Germany" in result
    assert "825.0" in result
    assert "425.0" in result


def test_group_average_missing_group_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Country": ["USA", "Germany"],
            "Score": [900, 350],
        }
    ).to_csv(csv_path, index=False)

    result = group_average(
        str(csv_path),
        "Region",
        "Score",
    )

    assert result == "Column 'Region' does not exist."


def test_group_average_missing_value_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Country": ["USA", "Germany"],
            "Score": [900, 350],
        }
    ).to_csv(csv_path, index=False)

    result = group_average(
        str(csv_path),
        "Country",
        "Revenue",
    )

    assert result == "Column 'Revenue' does not exist."


def test_group_average_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = group_average(
        str(missing_path),
        "Country",
        "Score",
    )

    assert result.startswith("Error calculating grouped average:")


def test_value_counts_valid_file(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Country": ["USA", "Germany", "USA", "USA", "Germany"],
        }
    ).to_csv(csv_path, index=False)

    result = value_counts(
        str(csv_path),
        "Country",
    )

    assert "USA" in result
    assert "Germany" in result
    assert "3" in result
    assert "2" in result


def test_value_counts_includes_missing_values(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Country": ["USA", None, "USA"],
        }
    ).to_csv(csv_path, index=False)

    result = value_counts(
        str(csv_path),
        "Country",
    )

    assert "USA" in result
    assert "2" in result
    assert "NaN" in result


def test_value_counts_missing_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Country": ["USA", "Germany"],
        }
    ).to_csv(csv_path, index=False)

    result = value_counts(
        str(csv_path),
        "Region",
    )

    assert result == "Column 'Region' does not exist."


def test_value_counts_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = value_counts(
        str(missing_path),
        "Country",
    )

    assert result.startswith("Error counting values:")


def test_find_maximum_returns_correct_row(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "FirstName": ["Jossef", "Kevin", "Mary"],
            "Country": ["Germany", "USA", "USA"],
            "Score": [350.0, 900.0, 750.0],
        }
    ).to_csv(csv_path, index=False)

    result = find_maximum(
        str(csv_path),
        "Score",
    )

    assert "CustomerID" in result
    assert "2" in result
    assert "Kevin" in result
    assert "USA" in result
    assert "900.0" in result


def test_find_maximum_ignores_missing_numeric_values(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "Score": [350.0, None, 750.0],
        }
    ).to_csv(csv_path, index=False)

    result = find_maximum(
        str(csv_path),
        "Score",
    )

    assert "CustomerID" in result
    assert "3.0" in result or "3" in result
    assert "750.0" in result


def test_find_maximum_missing_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Score": [100, 200],
        }
    ).to_csv(csv_path, index=False)

    result = find_maximum(
        str(csv_path),
        "Revenue",
    )

    assert result == "Column 'Revenue' does not exist."


def test_find_minimum_returns_correct_row(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "FirstName": ["Jossef", "Kevin", "Mary"],
            "Country": ["Germany", "USA", "USA"],
            "Score": [350.0, 900.0, 750.0],
        }
    ).to_csv(csv_path, index=False)

    result = find_minimum(
        str(csv_path),
        "Score",
    )

    assert "CustomerID" in result
    assert "1" in result
    assert "Jossef" in result
    assert "Germany" in result
    assert "350.0" in result


def test_find_minimum_missing_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Score": [100, 200],
        }
    ).to_csv(csv_path, index=False)

    result = find_minimum(
        str(csv_path),
        "Revenue",
    )

    assert result == "Column 'Revenue' does not exist."


def test_find_maximum_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = find_maximum(
        str(missing_path),
        "Score",
    )

    assert result.startswith("Error finding maximum value:")


def test_find_minimum_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = find_minimum(
        str(missing_path),
        "Score",
    )

    assert result.startswith("Error finding minimum value:")


def test_calculate_correlation_positive(tmp_path):
    csv_path = tmp_path / "positive.csv"

    pd.DataFrame(
        {
            "X": [1, 2, 3, 4],
            "Y": [10, 20, 30, 40],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "X",
        "Y",
    )

    assert "Correlation coefficient: 1.0000" in result
    assert "Relationship: very strong positive correlation" in result
    assert "Complete observations used: 4" in result


def test_calculate_correlation_negative(tmp_path):
    csv_path = tmp_path / "negative.csv"

    pd.DataFrame(
        {
            "X": [1, 2, 3, 4],
            "Y": [40, 30, 20, 10],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "X",
        "Y",
    )

    assert "Correlation coefficient: -1.0000" in result
    assert "Relationship: very strong negative correlation" in result


def test_calculate_correlation_drops_missing_rows(tmp_path):
    csv_path = tmp_path / "missing_values.csv"

    pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [10.0, None, 30.0, 40.0],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "X",
        "Y",
    )

    assert "Complete observations used: 3" in result


def test_calculate_correlation_missing_x_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "Score": [100, 200, 300],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "CustomerID",
        "Score",
    )

    assert result == "Column 'CustomerID' does not exist in the dataset."


def test_calculate_correlation_missing_y_column(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "CustomerID",
        "Score",
    )

    assert result == "Column 'Score' does not exist in the dataset."


def test_calculate_correlation_requires_numeric_x(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "FirstName": ["Anna", "Ben", "Cara"],
            "Score": [100, 200, 300],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "FirstName",
        "Score",
    )

    assert result == "Column 'FirstName' must be numeric."


def test_calculate_correlation_requires_numeric_y(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "FirstName": ["Anna", "Ben", "Cara"],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "CustomerID",
        "FirstName",
    )

    assert result == "Column 'FirstName' must be numeric."


def test_calculate_correlation_requires_two_complete_rows(tmp_path):
    csv_path = tmp_path / "too_few.csv"

    pd.DataFrame(
        {
            "X": [1.0, 2.0],
            "Y": [10.0, None],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "X",
        "Y",
    )

    assert result == (
        "At least two complete observations are required "
        "to calculate correlation."
    )


def test_calculate_correlation_constant_column(tmp_path):
    csv_path = tmp_path / "constant.csv"

    pd.DataFrame(
        {
            "X": [1, 1, 1, 1],
            "Y": [10, 20, 30, 40],
        }
    ).to_csv(csv_path, index=False)

    result = calculate_correlation(
        str(csv_path),
        "X",
        "Y",
    )

    assert result == (
        "Correlation could not be calculated. "
        "One of the columns may contain constant values."
    )


def test_calculate_correlation_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = calculate_correlation(
        str(missing_path),
        "X",
        "Y",
    )

    assert result == f"CSV file not found: {missing_path}"
