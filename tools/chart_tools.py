import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from strands import tool


OUTPUT_DIR = "output"


def _prepare_output_path(output_file: str, default_filename: str) -> str:
    """
    Force all generated charts into the output/ directory.

    If a custom output filename is supplied, only its filename is kept.
    This prevents charts from being scattered across data/, charts/,
    or the project root.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = os.path.basename(output_file.strip()) if output_file else default_filename

    if not filename:
        filename = default_filename

    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"

    return os.path.join(OUTPUT_DIR, filename)


def _load_csv(file_path: str):
    """
    Load a CSV file and return either a DataFrame or an error message.
    """
    if not os.path.exists(file_path):
        return None, f"CSV file not found: {file_path}"

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return None, f"CSV file is empty: {file_path}"
    except Exception as exc:
        return None, f"Unable to read CSV file '{file_path}': {exc}"

    if df.empty:
        return None, f"CSV file contains no rows: {file_path}"

    return df, None


def _missing_column_message(df: pd.DataFrame, column: str) -> str:
    available = ", ".join(map(str, df.columns))
    return (
        f"Column '{column}' does not exist in the dataset. "
        f"Available columns: {available}"
    )


@tool
def create_grouped_bar_chart(
    file_path: str,
    group_column: str,
    value_column: str,
    output_file: str = "bar_chart_average_by_group.png",
) -> str:
    """
    Create a bar chart of the average numeric value for each category.

    Use this tool only when the user explicitly requests a bar chart,
    graph, plot, or visualisation of grouped averages.

    Args:
        file_path: Path to the CSV dataset.
        group_column: Categorical column used to group the data.
        value_column: Numeric column whose average is calculated.
        output_file: Optional chart filename. The chart is always saved
            inside the output/ directory.

    Returns:
        A message confirming success and the exact saved chart path.
    """
    df, error = _load_csv(file_path)

    if error:
        return error

    if group_column not in df.columns:
        return _missing_column_message(df, group_column)

    if value_column not in df.columns:
        return _missing_column_message(df, value_column)

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"Column '{value_column}' must be numeric to create a bar chart."

    chart_data = (
        df[[group_column, value_column]]
        .dropna()
        .groupby(group_column)[value_column]
        .mean()
        .sort_values(ascending=False)
    )

    if chart_data.empty:
        return (
            f"No valid non-missing data is available for "
            f"'{group_column}' and '{value_column}'."
        )

    saved_path = _prepare_output_path(
        output_file,
        "bar_chart_average_by_group.png",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        chart_data.plot(kind="bar", ax=ax)

        ax.set_title(f"Average {value_column} by {group_column}")
        ax.set_xlabel(group_column)
        ax.set_ylabel(f"Average {value_column}")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return (
        f"Bar chart created successfully.\n"
        f"Group column: {group_column}\n"
        f"Value column: {value_column}\n"
        f"Chart saved to: {saved_path}"
    )


@tool
def create_pie_chart(
    file_path: str,
    column: str,
    output_file: str = "pie_chart.png",
) -> str:
    """
    Create a pie chart showing the distribution of a categorical column.

    Use this tool only when the user explicitly requests a pie chart.

    Args:
        file_path: Path to the CSV dataset.
        column: Categorical column to visualise.
        output_file: Optional chart filename. The chart is always saved
            inside the output/ directory.

    Returns:
        A message confirming success and the exact saved chart path.
    """
    df, error = _load_csv(file_path)

    if error:
        return error

    if column not in df.columns:
        return _missing_column_message(df, column)

    chart_data = df[column].dropna().value_counts()

    if chart_data.empty:
        return f"No valid non-missing data is available in column '{column}'."

    saved_path = _prepare_output_path(
        output_file,
        "pie_chart.png",
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    try:
        ax.pie(
            chart_data.values,
            labels=chart_data.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title(f"Distribution of {column}")

        fig.tight_layout()
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return (
        f"Pie chart created successfully.\n"
        f"Column: {column}\n"
        f"Chart saved to: {saved_path}"
    )


@tool
def create_line_chart(
    file_path: str,
    x_column: str,
    y_column: str,
    output_file: str = "line_chart.png",
) -> str:
    """
    Create a line chart showing a numeric y-column across an ordered x-axis.

    Use this tool only when the user explicitly requests a line chart,
    trend chart, graph, plot, or visualisation.

    Args:
        file_path: Path to the CSV dataset.
        x_column: Column used on the x-axis.
        y_column: Numeric column used on the y-axis.
        output_file: Optional chart filename. The chart is always saved
            inside the output/ directory.

    Returns:
        A message confirming success and the exact saved chart path.
    """
    df, error = _load_csv(file_path)

    if error:
        return error

    if x_column not in df.columns:
        return _missing_column_message(df, x_column)

    if y_column not in df.columns:
        return _missing_column_message(df, y_column)

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"Column '{y_column}' must be numeric to create a line chart."

    chart_data = df[[x_column, y_column]].dropna()

    if chart_data.empty:
        return (
            f"No valid non-missing data is available for "
            f"'{x_column}' and '{y_column}'."
        )

    saved_path = _prepare_output_path(
        output_file,
        "line_chart.png",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        ax.plot(
            chart_data[x_column],
            chart_data[y_column],
            marker="o",
        )

        ax.set_title(f"{y_column} by {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return (
        f"Line chart created successfully.\n"
        f"X column: {x_column}\n"
        f"Y column: {y_column}\n"
        f"Chart saved to: {saved_path}"
    )


@tool
def create_scatter_plot(
    file_path: str,
    x_column: str,
    y_column: str,
    output_file: str = "scatter_plot.png",
) -> str:
    """
    Create a scatter plot showing the relationship between two numeric columns.

    Use this tool only when the user explicitly requests a scatter plot
    or a visualisation of the relationship between two numeric variables.

    Args:
        file_path: Path to the CSV dataset.
        x_column: Numeric column used on the x-axis.
        y_column: Numeric column used on the y-axis.
        output_file: Optional chart filename. The chart is always saved
            inside the output/ directory.

    Returns:
        A message confirming success and the exact saved chart path.
    """
    df, error = _load_csv(file_path)

    if error:
        return error

    if x_column not in df.columns:
        return _missing_column_message(df, x_column)

    if y_column not in df.columns:
        return _missing_column_message(df, y_column)

    if not pd.api.types.is_numeric_dtype(df[x_column]):
        return f"Column '{x_column}' must be numeric to create a scatter plot."

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"Column '{y_column}' must be numeric to create a scatter plot."

    chart_data = df[[x_column, y_column]].dropna()

    if chart_data.empty:
        return (
            f"No valid non-missing data is available for "
            f"'{x_column}' and '{y_column}'."
        )

    saved_path = _prepare_output_path(
        output_file,
        "scatter_plot.png",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        ax.scatter(
            chart_data[x_column],
            chart_data[y_column],
        )

        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

        fig.tight_layout()
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return (
        f"Scatter plot created successfully.\n"
        f"X column: {x_column}\n"
        f"Y column: {y_column}\n"
        f"Chart saved to: {saved_path}"
    )
