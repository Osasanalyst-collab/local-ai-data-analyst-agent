Local AI Data Analyst Agent

[![Python Tests](https://github.com/Osasanalyst-collab/local-ai-data-analyst-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/Osasanalyst-collab/local-ai-data-analyst-agent/actions/workflows/tests.yml)

A local, tool-using AI data analysis agent built with Python, Ollama, Qwen3, Strands Agents, Pandas, and Matplotlib.

The project converts natural-language business questions into a minimal set of verified data-analysis operations, executes only the required tools, and returns deterministic results. It is designed as a portfolio project demonstrating practical skills in AI agents, data analysis, tool orchestration, testing, logging, and local LLM integration.

Project Highlights

Runs locally with Ollama

Uses Qwen3 8B as the planner model

Uses a structured JSON planner

Applies deterministic plan minimisation

Uses single-tool Strands executors

Prevents repeated tool execution

Supports CSV inspection, statistics, data quality, grouped analysis, correlation, and charts

Produces deterministic final responses

Includes application logging

Includes 72 automated tests

Architecture

flowchart LR
    A[User Request] --> B[Qwen3 Structured Planner]
    B --> C[Deterministic Plan Minimiser]
    C --> D{Selected Operations}
    D --> E[Single-Tool Strands Executor]
    E --> F[Verified Python Tool]
    F --> G[Deterministic Result Composer]
    G --> H[Final Response]
    F --> I[Logs]
    F --> J[Charts / Output]

The planner does not directly analyse the dataset. It selects operations from a fixed tool registry. The deterministic minimiser removes unnecessary operations before execution.

For more detail, see docs/architecture.md.

Supported Operations

Operation

Purpose

inspect_csv

Inspect CSV structure and preview dataset information

summary_statistics

Produce descriptive statistics for numeric columns

check_data_quality

Check missing values, nulls, and duplicate rows

group_average

Calculate an average grouped by a category

value_counts

Count category frequencies

find_maximum

Return the row with the maximum value

find_minimum

Return the row with the minimum value

calculate_correlation

Calculate Pearson correlation between numeric columns

create_grouped_bar_chart

Create a grouped-average bar chart

create_pie_chart

Create a categorical pie chart

create_line_chart

Create a line chart

create_scatter_plot

Create a scatter plot

Example Questions

Inspect data/Customers.csv.

Give me summary statistics for data/Customers.csv.

Check the data quality of data/Customers.csv.

Count the number of customers in each Country.

Calculate the average Score by Country.

Find the customer with the highest Score.

Find the customer with the lowest Score.

Calculate the correlation between CustomerID and Score.

Create a pie chart showing customers by Country.

Create a line chart showing Score by CustomerID.

Create a scatter plot showing CustomerID against Score.

Calculate the average Score by Country and create a bar chart.

Example Dataset

The repository includes a small sample dataset:

data/Customers.csv

Columns:

CustomerID
FirstName
LastName
Country
Score

The sample data is intentionally small so the behaviour of each tool can be inspected and tested easily.

Project Structure

strands-ai-agent/
│
├── agent.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Customers.csv
│
├── tools/
│   ├── data_tools.py
│   ├── analysis_tools.py
│   └── chart_tools.py
│
├── tests/
│   ├── test_data_tools.py
│   ├── test_analysis_tools.py
│   ├── test_chart_tools.py
│   ├── test_planner.py
│   └── test_agent_workflows.py
│
├── output/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
└── docs/
    └── architecture.md

Technology Stack

Python 3.12

Pandas

Matplotlib

Ollama

Qwen3 8B

Strands Agents

Pytest

Git / GitHub

Prerequisites

Install:

Python 3.12

Git

Ollama

A local Qwen3 model

Pull the model with:

ollama pull qwen3:8b

Confirm Ollama is available:

ollama list

Setup

Clone the repository:

git clone <YOUR-GITHUB-REPOSITORY-URL>
cd strands-ai-agent

Create or activate a Python environment.

Example with Conda:

conda create -n strands_agent python=3.12 -y
conda activate strands_agent

Install dependencies:

python -m pip install -r requirements.txt

Run the Agent

Make sure Ollama is running and qwen3:8b is available.

Then run:

python agent.py

Enter a data-analysis request at the You: prompt.

Example:

You: Calculate the average Score by Country in data/Customers.csv.

Planner Design

The planner uses Qwen3 to produce structured JSON containing only valid operation names.

Example conceptually:

{
  "operations": [
    "group_average"
  ]
}

The planner output is then passed through deterministic minimisation rules.

This prevents unnecessary execution. For example:

User:
Create a pie chart showing customers by Country.

Possible over-selected LLM plan:
inspect_csv
value_counts
create_pie_chart

Final minimised plan:
create_pie_chart

This design reduces tool misuse and makes the agent easier to test.

Tool Execution Safety

Each selected operation is executed by a fresh Strands agent with access to exactly one tool.

The executor is also protected by a hook that limits the selected tool to one call per invocation.

This avoids:

repeated tool calls

unnecessary retries

accidental use of unrelated tools

tool-selection instability from exposing all tools at once

Deterministic Final Response

The final response is composed from verified tool outputs rather than generated again by the LLM.

This reduces:

hallucinated analysis

missing tool results

duplicated operations

leakage from earlier model output

Logging

Runtime events are written to:

logs/agent.log

The log records:

agent startup

user request

selected plan

operation execution

operation result

execution time

exceptions

Generated logs are ignored by Git.

Chart Output

Charts are written to:

output/

Supported visualisations include:

grouped bar charts

pie charts

line charts

scatter plots

Generated chart files are ignored by Git.

Automated Testing

The project contains 72 automated tests.

Test Suite

Tests

Data tools

9

Analysis tools

25

Chart tools

16

Planner

15

Workflow orchestration

7

Total

72

Run the complete verified suite with:

python -m pytest tests/test_data_tools.py tests/test_analysis_tools.py tests/test_chart_tools.py tests/test_planner.py tests/test_agent_workflows.py -v

Expected result:

72 passed

The workflow tests mock the live LLM/executor layer so the orchestration tests remain deterministic and fast.

Key Engineering Decisions

1. Structured planner instead of free-text planning

The planner produces a constrained JSON operation list. This is more reliable than parsing free-text instructions from the model.

2. Deterministic plan minimisation

A second layer removes operations the user did not explicitly request.

3. One-tool executor

The executor receives only the selected tool. This reduces tool-selection errors.

4. One-call execution guard

A Strands hook prevents repeated calls to the same selected tool in one executor invocation.

5. Deterministic response composition

Verified tool outputs are composed directly instead of asking the LLM to regenerate the final analytical answer.

6. Local LLM execution

The agent runs through Ollama, making the project suitable for local experimentation without requiring a hosted LLM API.

Current Limitations

The current data source is CSV-based.

Column selection still depends on the executor understanding the user request correctly.

The project currently uses a command-line interface.

The sample dataset is intentionally small.

It does not yet include authentication, a web UI, or persistent conversation memory.

Future Improvements

Potential next steps:

Streamlit or FastAPI interface

upload-your-own CSV workflow

SQL database integration

automated schema detection

richer business-question templates

larger benchmark datasets

Docker packaging

GitHub Actions CI

additional statistical tools

data-cleaning actions

report export

multi-dataset analysis

Portfolio Skills Demonstrated

This project demonstrates practical experience with:

Python

Pandas

data analysis

data quality checks

data visualisation

AI agents

local LLMs

Ollama

tool calling

Strands Agents

structured outputs

deterministic orchestration

defensive programming

logging

unit testing

workflow testing

Git and GitHub

Author

Osarinmwian Nelson Oghomwenrhiere

GitHub: Osasanalyst-collab

License

This project is intended as a learning and portfolio project. Add a licence file before reuse or redistribution if you want to publish it under a specific open-source licence.