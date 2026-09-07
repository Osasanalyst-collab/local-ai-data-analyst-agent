import pandas as pd
from strands import tool


@tool
def inspect_csv(file_path: str) -> str:
    """
    Inspect a CSV file and return its basic structure.

    Args:
        file_path: Path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        return f"""
CSV successfully loaded.

Rows: {len(df)}
Columns: {len(df.columns)}

Column names:
{list(df.columns)}

Data types:
{df.dtypes.to_string()}

Missing values:
{df.isnull().sum().to_string()}

First 5 rows:
{df.head().to_string()}
"""
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


@tool
def summary_statistics(file_path: str) -> str:
    """
    Calculate descriptive statistics for numeric columns in a CSV file.

    Args:
        file_path: Path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return "The dataset contains no numeric columns."

        return numeric_df.describe().to_string()

    except Exception as e:
        return f"Error calculating statistics: {str(e)}"


@tool
def check_data_quality(file_path: str) -> str:
    """
    Check missing values and duplicate rows in a CSV file.

    Args:
        file_path: Path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        missing = df.isnull().sum()
        duplicates = df.duplicated().sum()

        return f"""
Data Quality Report

Total rows: {len(df)}

Duplicate rows:
{duplicates}

Missing values by column:
{missing.to_string()}
"""
    except Exception as e:
        return f"Error checking data quality: {str(e)}"