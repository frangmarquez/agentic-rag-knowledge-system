from importlib import import_module


def test_package_imports() -> None:
    package = import_module("agentic_rag_knowledge_system")

    assert package.__version__
