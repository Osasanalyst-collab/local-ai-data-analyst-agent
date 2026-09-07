from types import SimpleNamespace

import agent


def test_uploaded_dataset_path_overrides_sample_dataset():
    """
    Regression test for the Streamlit upload bug.
    """
    event = SimpleNamespace(
        tool_use={
            "name": "inspect_csv",
            "input": {
                "file_path": "data/Customers.csv",
            },
        }
    )

    guard = agent.ForceDatasetPath(
        "data/uploads/England.csv"
    )

    guard.force_dataset_path(event)

    assert (
        event.tool_use["input"]["file_path"]
        == "data/uploads/England.csv"
    )


def test_process_request_forwards_authoritative_dataset(
    monkeypatch,
):
    """
    Verify that process_request forwards the selected dataset path.
    """
    captured = {}

    monkeypatch.setattr(
        agent,
        "create_plan",
        lambda request: ["inspect_csv"],
    )

    def fake_execute_plan(
        user_request,
        operations,
        dataset_path=None,
    ):
        captured["user_request"] = user_request
        captured["operations"] = operations
        captured["dataset_path"] = dataset_path

        return [
            {
                "operation": "inspect_csv",
                "result": "Verified England dataset result",
            }
        ]

    monkeypatch.setattr(
        agent,
        "execute_plan",
        fake_execute_plan,
    )

    result = agent.process_request(
        "Inspect the dataset",
        dataset_path="data/uploads/England.csv",
    )

    assert captured["operations"] == ["inspect_csv"]
    assert (
        captured["dataset_path"]
        == "data/uploads/England.csv"
    )
    assert "Verified England dataset result" in result
