import pandas as pd

from tools.data_tools import (
    inspect_csv,
    summary_statistics,
    check_data_quality,
)


def test_inspect_csv_valid_file(tmp_path):
    csv_path = tmp_path / "customers.csv"

    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "FirstName": ["Anna", "Ben", "Cara"],
            "LastName": ["Smith", None, "Jones"],
            "Country": ["UK", "USA", "UK"],
            "Score": [500.0, 700.0, None],
        }
    )
    df.to_csv(csv_path, index=False)

    result = inspect_csv(str(csv_path))

    assert "CSV successfully loaded." in result
    assert "Rows: 3" in result
    assert "Columns: 5" in result
    assert "CustomerID" in result
    assert "FirstName" in result
    assert "LastName" in result
    assert "Country" in result
    assert "Score" in result
    assert "Missing values:" in result
    assert "First 5 rows:" in result


def test_inspect_csv_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    result = inspect_csv(str(missing_path))

    assert result.startswith("Error reading CSV:")
    assert "does_not_exist.csv" in result


def test_inspect_csv_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    result = inspect_csv(str(csv_path))

    assert result.startswith("Error reading CSV:")


def test_summary_statistics_numeric_columns(tmp_path):
    csv_path = tmp_path / "numeric.csv"

    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "Score": [10.0, 20.0, 30.0],
            "Country": ["UK", "USA", "UK"],
        }
    )
    df.to_csv(csv_path, index=False)

    result = summary_statistics(str(csv_path))

    assert "CustomerID" in result
    assert "Score" in result
    assert "count" in result
    assert "mean" in result
    assert "min" in result
    assert "max" in result
    assert "20.0" in result


def test_summary_statistics_no_numeric_columns(tmp_path):
    csv_path = tmp_path / "text_only.csv"

    df = pd.DataFrame(
        {
            "FirstName": ["Anna", "Ben"],
            "Country": ["UK", "USA"],
        }
    )
    df.to_csv(csv_path, index=False)

    result = summary_statistics(str(csv_path))

    assert result == "The dataset contains no numeric columns."


def test_summary_statistics_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    result = summary_statistics(str(missing_path))

    assert result.startswith("Error calculating statistics:")
    assert "missing.csv" in result


def test_check_data_quality_detects_missing_and_duplicates(tmp_path):
    csv_path = tmp_path / "quality.csv"

    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 2, 3],
            "FirstName": ["Anna", "Ben", "Ben", "Cara"],
            "LastName": ["Smith", "Brown", "Brown", None],
            "Country": ["UK", "USA", "USA", "UK"],
            "Score": [500.0, 700.0, 700.0, None],
        }
    )
    df.to_csv(csv_path, index=False)

    result = check_data_quality(str(csv_path))

    assert "Data Quality Report" in result
    assert "Total rows: 4" in result
    assert "Duplicate rows:" in result
    assert "\n1\n" in result
    assert "LastName" in result
    assert "Score" in result

    # Verify the missing counts themselves are present.
    quality_df = pd.read_csv(csv_path)
    expected_missing = quality_df.isnull().sum()

    assert expected_missing["LastName"] == 1
    assert expected_missing["Score"] == 1
    assert quality_df.duplicated().sum() == 1


def test_check_data_quality_clean_file(tmp_path):
    csv_path = tmp_path / "clean.csv"

    df = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "FirstName": ["Anna", "Ben"],
            "Country": ["UK", "USA"],
            "Score": [500, 700],
        }
    )
    df.to_csv(csv_path, index=False)

    result = check_data_quality(str(csv_path))

    assert "Total rows: 2" in result
    assert "Duplicate rows:" in result
    assert "\n0\n" in result
    assert "CustomerID" in result
    assert "FirstName" in result
    assert "Country" in result
    assert "Score" in result


def test_check_data_quality_missing_file(tmp_path):
    missing_path = tmp_path / "missing_quality.csv"

    result = check_data_quality(str(missing_path))

    assert result.startswith("Error checking data quality:")
    assert "missing_quality.csv" in result
