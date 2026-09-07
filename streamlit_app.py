from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import ollama
import pandas as pd
import streamlit as st

import agent
from tools.csv_utils import read_csv_safely


APP_TITLE = "Local AI Data Analyst Agent"
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("output")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)


st.title("📊 Local AI Data Analyst Agent")
st.caption(
    "Upload a CSV file and ask natural-language questions using "
    "Qwen3, Ollama, Strands Agents, Pandas and Matplotlib."
)


# ============================================================
# SIDEBAR - DATASET UPLOAD
# ============================================================

with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="The uploaded CSV is saved locally inside data/uploads/.",
    )

    if uploaded_file is not None:
        safe_name = Path(uploaded_file.name).name
        dataset_path = UPLOAD_DIR / safe_name

        dataset_path.write_bytes(
            uploaded_file.getbuffer()
        )

        st.success(f"Loaded: {safe_name}")
        st.code(str(dataset_path), language=None)

    else:
        dataset_path = None

        st.info(
            "Upload a CSV file to begin analysis."
        )

    st.divider()

    st.markdown("### Example questions")
    st.markdown(
        """
- Inspect the dataset
- Check the data quality
- Calculate an average by a category
- Find the highest numeric value
- Find the lowest numeric value
- Calculate correlation between two numeric columns
- Create a pie chart by a category
- Create a scatter plot between two numeric columns
        """
    )


# ============================================================
# DATASET HELPERS
# ============================================================

def load_dataset_for_ui(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load the selected dataset using the same safe CSV reader
    used by the analysis tools.
    """
    return read_csv_safely(str(file_path))


def build_column_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a column-level profile table.
    """
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Missing values": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Missing %": [
                round(
                    df[column].isna().mean() * 100,
                    1,
                )
                for column in df.columns
            ],
            "Unique values": [
                int(
                    df[column].nunique(
                        dropna=True
                    )
                )
                for column in df.columns
            ],
        }
    )


def build_missing_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only columns containing missing values.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return pd.DataFrame(
            columns=[
                "Column",
                "Missing values",
                "Missing %",
            ]
        )

    summary = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing values": (
                missing.values.astype(int)
            ),
            "Missing %": [
                round(
                    (count / len(df)) * 100,
                    1,
                )
                if len(df) > 0
                else 0.0
                for count in missing.values
            ],
        }
    )

    return summary.sort_values(
        by=[
            "Missing %",
            "Missing values",
        ],
        ascending=False,
    ).reset_index(drop=True)


def get_quality_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate deterministic quality metrics.
    """
    missing_counts = df.isna().sum()

    if len(df) > 0:
        completely_missing = int(
            (
                missing_counts
                == len(df)
            ).sum()
        )

        partially_missing = int(
            (
                (missing_counts > 0)
                & (
                    missing_counts
                    < len(df)
                )
            ).sum()
        )

    else:
        completely_missing = 0
        partially_missing = 0

    complete_columns = int(
        (missing_counts == 0).sum()
    )

    return {
        "duplicates": int(
            df.duplicated().sum()
        ),
        "complete_columns": (
            complete_columns
        ),
        "partially_missing_columns": (
            partially_missing
        ),
        "completely_missing_columns": (
            completely_missing
        ),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
    }


# ============================================================
# CHART HELPERS
# ============================================================

def get_chart_snapshot() -> dict[str, int]:
    """
    Record chart modification times before a request.
    """
    snapshot = {}

    for chart_path in OUTPUT_DIR.glob("*.png"):
        try:
            snapshot[str(chart_path)] = (
                chart_path.stat().st_mtime_ns
            )

        except FileNotFoundError:
            pass

    return snapshot


def get_changed_charts(
    before: dict[str, int],
) -> list[Path]:
    """
    Return only charts created or modified
    by the current request.
    """
    changed = []

    for chart_path in OUTPUT_DIR.glob("*.png"):
        try:
            current_mtime = (
                chart_path.stat().st_mtime_ns
            )

        except FileNotFoundError:
            continue

        previous_mtime = before.get(
            str(chart_path)
        )

        if (
            previous_mtime is None
            or current_mtime
            > previous_mtime
        ):
            changed.append(chart_path)

    return sorted(
        changed,
        key=lambda path: (
            path.stat().st_mtime_ns
        ),
        reverse=True,
    )


# ============================================================
# AGENT EXECUTION
# ============================================================

def run_agent(
    user_question: str,
    file_path: Path,
) -> str:
    """
    Run the agent using the uploaded dataset path
    as the authoritative file path.
    """
    buffer = StringIO()

    with redirect_stdout(buffer):
        result = agent.process_request(
            user_question.strip(),
            dataset_path=str(file_path),
        )

    if (
        isinstance(result, str)
        and result.strip()
    ):
        return result.strip()

    printed_output = (
        buffer.getvalue().strip()
    )

    if "Agent:" in printed_output:
        printed_output = (
            printed_output.rsplit(
                "Agent:",
                1,
            )[-1].strip()
        )

    return (
        printed_output
        or (
            "The agent completed without "
            "returning visible output."
        )
    )


# ============================================================
# AI INSIGHT LAYER
# ============================================================

def generate_ai_insight(
    question: str,
    verified_result: str,
) -> str:
    """
    Generate a short interpretation using only
    the verified tool result.
    """
    prompt = f"""
You are the interpretation layer of a Local Data Analyst AI Agent.

USER QUESTION:
{question}

VERIFIED TOOL RESULT:
{verified_result}

Write a concise interpretation in plain professional language.

STRICT RULES:
- Use only information contained in VERIFIED TOOL RESULT.
- Do not invent numbers, causes, trends, business facts or recommendations.
- Do not perform new calculations.
- If the result is only a chart path or technical confirmation, explain
  that the requested output was created without inventing chart findings.
- If the result is an error, explain the error clearly.
- Use 2 to 5 short bullet points.
- Focus on what the result means, not how the code works.
"""

    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            think=False,
        )

        content = (
            response["message"]["content"]
            .strip()
        )

        if content:
            return content

    except Exception:
        pass

    return (
        "- The verified analysis result "
        "is shown above.\n"
        "- No additional interpretation "
        "was generated."
    )


# ============================================================
# EMPTY STATE
# ============================================================

if dataset_path is None:
    st.info(
        "📂 No dataset loaded. "
        "Upload a CSV file from the sidebar "
        "to begin analysis."
    )

    st.markdown(
        """
### How to use the agent

1. Upload a CSV dataset from the sidebar.
2. Review the dataset overview and preview.
3. Ask a natural-language analysis question.
4. Run the analysis.
5. Review the verified result, generated charts and AI insights.
        """
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

else:
    try:
        ui_df = load_dataset_for_ui(
            dataset_path
        )

        st.subheader("Dataset Overview")

        metric_col1, metric_col2, metric_col3 = (
            st.columns(3)
        )

        with metric_col1:
            st.metric(
                "Rows",
                f"{len(ui_df):,}",
            )

        with metric_col2:
            st.metric(
                "Columns",
                len(ui_df.columns),
            )

        with metric_col3:
            st.metric(
                "Missing Cells",
                f"{int(ui_df.isna().sum().sum()):,}",
            )

        with st.expander(
            "Column Information",
            expanded=False,
        ):
            st.dataframe(
                build_column_profile(
                    ui_df
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Data Preview")

        st.dataframe(
            ui_df.head(5),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # DATA QUALITY SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "### Data Quality Summary"
        )

        quality = get_quality_metrics(
            ui_df
        )

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            st.metric(
                "Duplicate Rows",
                f"{quality['duplicates']:,}",
            )

        with q2:
            st.metric(
                "Complete Columns",
                f"{quality['complete_columns']:,}",
            )

        with q3:
            st.metric(
                "Partially Missing",
                (
                    f"{quality['partially_missing_columns']:,}"
                ),
            )

        with q4:
            st.metric(
                "Completely Missing",
                (
                    f"{quality['completely_missing_columns']:,}"
                ),
            )

        missing_summary = (
            build_missing_summary(
                ui_df
            )
        )

        if missing_summary.empty:
            st.success(
                "No missing values "
                "were detected."
            )

        else:
            with st.expander(
                "Missing Value Breakdown",
                expanded=False,
            ):
                st.dataframe(
                    missing_summary,
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as exc:
        ui_df = None

        st.error(
            f"Could not preview dataset: {exc}"
        )


# ============================================================
# ANALYSIS SECTION
# ============================================================

if dataset_path is not None:
    st.divider()

    question = st.text_area(
        "Ask a question about the dataset",
        placeholder=(
            "Example: Check the data quality"
        ),
        height=120,
    )

    if st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
    ):
        if not question.strip():
            st.warning(
                "Enter a data-analysis "
                "question first."
            )

        elif not dataset_path.exists():
            st.error(
                f"Dataset not found: "
                f"{dataset_path}"
            )

        else:
            chart_snapshot = (
                get_chart_snapshot()
            )

            with st.spinner(
                "Running local AI analysis..."
            ):
                try:
                    response = run_agent(
                        user_question=question,
                        file_path=dataset_path,
                    )

                    st.subheader(
                        "Agent Result"
                    )

                    inspection_only = (
                        "inspect"
                        in question.lower()
                        and not any(
                            phrase
                            in question.lower()
                            for phrase in (
                                "summary",
                                "quality",
                                "average",
                                "mean",
                                "highest",
                                "maximum",
                                "lowest",
                                "minimum",
                                "correlation",
                                "chart",
                                "plot",
                                "graph",
                            )
                        )
                    )

                    if inspection_only:
                        st.success(
                            "Dataset inspection "
                            "completed. The exact "
                            "structure, column profile "
                            "and first five rows are "
                            "shown above."
                        )

                        with st.expander(
                            "Detailed tool result"
                        ):
                            st.text(response)

                    else:
                        st.markdown(
                            response
                        )

                    changed_charts = (
                        get_changed_charts(
                            chart_snapshot
                        )
                    )

                    if changed_charts:
                        st.subheader(
                            "Generated Chart"
                        )

                        for chart_path in (
                            changed_charts
                        ):
                            st.image(
                                str(chart_path),
                                caption=(
                                    chart_path.name
                                ),
                                use_container_width=True,
                            )

                    st.subheader(
                        "AI Insights"
                    )

                    with st.spinner(
                        "Interpreting the "
                        "verified result..."
                    ):
                        insight = (
                            generate_ai_insight(
                                question=question,
                                verified_result=(
                                    response
                                ),
                            )
                        )

                    st.markdown(
                        insight
                    )

                except Exception as exc:
                    st.error(
                        f"Analysis failed: {exc}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Runs locally. No dataset is loaded until the user uploads a CSV file. "
    "The selected dataset path is enforced at tool-call time, verified "
    "Python tool results are displayed directly, and AI insights are "
    "grounded in those verified results."
)
