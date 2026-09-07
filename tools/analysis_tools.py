import pandas as pd
from strands import tool


@tool
def group_average(
    file_path: str,
    group_column: str,
    value_column: str,
) -> str:
    """
    Calculate and return the average or mean of a numeric column
    for each category in another column.

    Use this tool when the user asks questions such as:
    - What is the average Score by Country?
    - What is the mean salary by department?
    - What is the average sales for each region?

    This tool returns numerical results only.
    Use this instead of a chart tool unless the user explicitly asks
    for a chart, graph, plot, or visualisation.

    Args:
        file_path: Path to the CSV dataset.
        group_column: Categorical column used to group the data.
        value_column: Numeric column whose average will be calculated.

    Returns:
        The grouped average values.
    """
    try:
        df = pd.read_csv(file_path)

        if group_column not in df.columns:
            return f"Column '{group_column}' does not exist."

        if value_column not in df.columns:
            return f"Column '{value_column}' does not exist."

        result = (
            df.groupby(group_column)[value_column]
            .mean()
            .sort_values(ascending=False)
        )

        return result.to_string()

    except Exception as e:
        return f"Error calculating grouped average: {str(e)}"


@tool
def value_counts(file_path: str, column: str) -> str:
    """
    Count how many times each value appears in a column.

    Args:
        file_path: Path to the CSV file.
        column: Column to count.
    """
    try:
        df = pd.read_csv(file_path)

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        result = df[column].value_counts(dropna=False)

        return result.to_string()

    except Exception as e:
        return f"Error counting values: {str(e)}"


@tool
def find_maximum(file_path: str, column: str) -> str:
    """
    Find the row containing the maximum value in a numeric column.

    Args:
        file_path: Path to the CSV file.
        column: Numeric column to search.
    """
    try:
        df = pd.read_csv(file_path)

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        max_index = df[column].idxmax()
        row = df.loc[max_index]

        return row.to_string()

    except Exception as e:
        return f"Error finding maximum value: {str(e)}"


@tool
def find_minimum(file_path: str, column: str) -> str:
    """
    Find the row containing the minimum value in a numeric column.

    Args:
        file_path: Path to the CSV file.
        column: Numeric column to search.
    """
    try:
        df = pd.read_csv(file_path)

        if column not in df.columns:
            return f"Column '{column}' does not exist."

        min_index = df[column].idxmin()
        row = df.loc[min_index]

        return row.to_string()

    except Exception as e:
        return f"Error finding minimum value: {str(e)}"


@tool
def calculate_correlation(
    file_path: str,
    x_column: str,
    y_column: str,
) -> str:
    """
    Calculate the Pearson correlation between two numeric columns.

    Use this tool when the user explicitly asks about:
    - correlation
    - relationship strength
    - positive correlation
    - negative correlation
    - how strongly two numeric variables are related

    Do NOT use this tool just because the user asks for a scatter plot.
    Use create_scatter_plot when a visualisation is requested.

    Args:
        file_path: Path to the CSV dataset.
        x_column: First numeric column.
        y_column: Second numeric column.

    Returns:
        The Pearson correlation coefficient and an interpretation
        of the relationship.
    """

    try:
        df = pd.read_csv(file_path)

        if x_column not in df.columns:
            return f"Column '{x_column}' does not exist in the dataset."

        if y_column not in df.columns:
            return f"Column '{y_column}' does not exist in the dataset."

        if not pd.api.types.is_numeric_dtype(df[x_column]):
            return f"Column '{x_column}' must be numeric."

        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return f"Column '{y_column}' must be numeric."

        correlation_data = df[[x_column, y_column]].dropna()

        if len(correlation_data) < 2:
            return (
                "At least two complete observations are required "
                "to calculate correlation."
            )

        # Prevent NumPy/Pandas runtime warnings when one of the
        # numeric columns contains only a single unique value.
        if (
            correlation_data[x_column].nunique() < 2
            or correlation_data[y_column].nunique() < 2
        ):
            return (
                "Correlation could not be calculated. "
                "One of the columns may contain constant values."
            )

        correlation = correlation_data[x_column].corr(
            correlation_data[y_column]
        )

        if pd.isna(correlation):
            return (
                "Correlation could not be calculated. "
                "One of the columns may contain constant values."
            )

        abs_correlation = abs(correlation)

        if abs_correlation >= 0.8:
            strength = "very strong"
        elif abs_correlation >= 0.6:
            strength = "strong"
        elif abs_correlation >= 0.4:
            strength = "moderate"
        elif abs_correlation >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        if correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "no linear"

        return (
            f"Pearson correlation analysis\n"
            f"{x_column} vs {y_column}\n"
            f"Correlation coefficient: {correlation:.4f}\n"
            f"Relationship: {strength} {direction} correlation\n"
            f"Complete observations used: {len(correlation_data)}"
        )

    except FileNotFoundError:
        return f"CSV file not found: {file_path}"

    except Exception as e:
        return f"Error calculating correlation: {str(e)}"
