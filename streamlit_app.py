from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

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
    "Ask natural-language questions about a CSV file using "
    "Qwen3, Ollama, Strands Agents, Pandas and Matplotlib."
)


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
        dataset_path.write_bytes(uploaded_file.getbuffer())

        st.success(f"Loaded: {safe_name}")
        st.code(str(dataset_path), language=None)
    else:
        dataset_path = Path("data/Customers.csv")
        st.info("Using the sample dataset.")
        st.code(str(dataset_path), language=None)

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


def load_dataset_for_ui(file_path: Path) -> pd.DataFrame:
    return read_csv_safely(str(file_path))


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": [str(df[column].dtype) for column in df.columns],
            "Missing values": [int(df[column].isna().sum()) for column in df.columns],
            "Unique values": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )


def get_chart_snapshot() -> dict[str, int]:
    snapshot = {}

    for chart_path in OUTPUT_DIR.glob("*.png"):
        try:
            snapshot[str(chart_path)] = chart_path.stat().st_mtime_ns
        except FileNotFoundError:
            pass

    return snapshot


def get_changed_charts(before: dict[str, int]) -> list[Path]:
    changed = []

    for chart_path in OUTPUT_DIR.glob("*.png"):
        try:
            current_mtime = chart_path.stat().st_mtime_ns
        except FileNotFoundError:
            continue

        previous_mtime = before.get(str(chart_path))

        if previous_mtime is None or current_mtime > previous_mtime:
            changed.append(chart_path)

    return sorted(
        changed,
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def run_agent(user_question: str, file_path: Path) -> str:
    buffer = StringIO()

    with redirect_stdout(buffer):
        result = agent.process_request(
            user_question.strip(),
            dataset_path=str(file_path),
        )

    if isinstance(result, str) and result.strip():
        return result.strip()

    printed_output = buffer.getvalue().strip()

    if "Agent:" in printed_output:
        printed_output = printed_output.rsplit("Agent:", 1)[-1].strip()

    return printed_output or "The agent completed without returning visible output."


try:
    ui_df = load_dataset_for_ui(dataset_path)

    st.subheader("Dataset Overview")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Rows", f"{len(ui_df):,}")

    with metric_col2:
        st.metric("Columns", len(ui_df.columns))

    with metric_col3:
        st.metric(
            "Missing Cells",
            f"{int(ui_df.isna().sum().sum()):,}",
        )

    with st.expander("Column Information", expanded=False):
        st.dataframe(
            build_column_profile(ui_df),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Data Preview")
    st.dataframe(
        ui_df.head(5),
        use_container_width=True,
        hide_index=True,
    )

except Exception as exc:
    ui_df = None
    st.error(f"Could not preview dataset: {exc}")


st.divider()


question = st.text_area(
    "Ask a question about the dataset",
    placeholder="Example: Calculate the average Value by Age",
    height=120,
)


if st.button("Run Analysis", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a data-analysis question first.")

    elif not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")

    else:
        chart_snapshot = get_chart_snapshot()

        with st.spinner("Running local AI analysis..."):
            try:
                response = run_agent(
                    user_question=question,
                    file_path=dataset_path,
                )

                st.subheader("Agent Result")

                inspection_only = (
                    "inspect" in question.lower()
                    and not any(
                        phrase in question.lower()
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
                        "Dataset inspection completed. "
                        "The exact structure, column profile and first five rows "
                        "are shown above."
                    )

                    with st.expander("Detailed tool result"):
                        st.text(response)
                else:
                    st.markdown(response)

                changed_charts = get_changed_charts(chart_snapshot)

                if changed_charts:
                    st.subheader("Generated Chart")

                    for chart_path in changed_charts:
                        st.image(
                            str(chart_path),
                            caption=chart_path.name,
                            use_container_width=True,
                        )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


st.divider()
st.caption(
    "Runs locally. The selected dataset path is enforced at tool-call time, "
    "and verified Python tool results are displayed directly."
)
