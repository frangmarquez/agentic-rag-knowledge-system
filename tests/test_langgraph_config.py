import json
from pathlib import Path


def test_langgraph_config_points_to_dev_graph() -> None:
    config = json.loads(Path("langgraph.json").read_text(encoding="utf-8"))

    assert config["graphs"]["agentic_rag"].endswith("/workflow/graph.py:graph")
    assert config["dependencies"] == ["."]
    assert config["env"] == "./.env"
