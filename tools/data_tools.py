
from strands import tool

from tools.csv_utils import read_csv_safely


@tool
def inspect_csv(file_path: str) -> str:
    """
    Inspect a CSV dataset and return exact structural information.

    This version keeps the original output contract used by the
    automated tests while also providing the richer inspection details
    needed by the Streamlit interface.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        A deterministic dataset inspection report.
    """
    try:
        df = read_csv_safely(file_path)

        row_count = len(df)
        column_count = len(df.columns)

        dtype_lines = [
            f"{column}: {df[column].dtype}"
            for column in df.columns
        ]

        missing_counts = df.isna().sum()

        missing_lines = [
            f"{column}: {int(missing_counts[column])}"
            for column in df.columns
        ]

        if row_count > 0:
            preview = df.head(5).to_string(index=False)
        else:
            preview = "No rows available."

        return (
            "CSV successfully loaded.\n\n"
            f"File: {file_path}\n"
            f"Rows: {row_count}\n"
            f"Columns: {column_count}\n\n"
            f"Column names: {list(df.columns)}\n\n"
            "Data types:\n"
            + "\n".join(dtype_lines)
            + "\n\nMissing values:\n"
            + "\n".join(missing_lines)
            + "\n\nFirst 5 rows:\n"
            + preview
        )

    except Exception as e:
        return f"Error reading CSV: {str(e)}"


@tool
def summary_statistics(file_path: str) -> str:
    """
    Generate descriptive statistics for numeric columns in a CSV file.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        Descriptive statistics for numeric columns.
    """
    try:
        df = read_csv_safely(file_path)

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return "The dataset contains no numeric columns."

        return numeric_df.describe().to_string()

    except Exception as e:
        return f"Error calculating statistics: {str(e)}"


@tool
def check_data_quality(file_path: str) -> str:
    """
    Check a CSV dataset for missing values and duplicate rows.

    The report format intentionally preserves the wording expected by
    the existing automated tests.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        A deterministic data-quality report.
    """
    try:
        df = read_csv_safely(file_path)

        missing_values = df.isna().sum()
        duplicate_rows = int(df.duplicated().sum())

        return (
            "Data Quality Report\n\n"
            f"Total rows: {len(df)}\n"
            f"Total columns: {len(df.columns)}\n\n"
            "Missing values:\n"
            f"{missing_values.to_string()}\n\n"
            "Duplicate rows:\n"
            f"{duplicate_rows}\n"
        )

    except Exception as e:
        return f"Error checking data quality: {str(e)}"
