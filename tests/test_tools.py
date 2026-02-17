from pathlib import Path

from explainable_agent.agent import lexical_jaccard_similarity
from explainable_agent.tools import calculate_math, list_workspace_files, read_text_file


def test_calculate_math_success() -> None:
    result = calculate_math("(2+3)*4", Path("."))
    assert result == "20.0"


def test_calculate_math_rejects_invalid() -> None:
    result = calculate_math("__import__('os').system('whoami')", Path("."))
    assert result.startswith("ERROR:")


def test_read_text_file_blocks_path_escape(tmp_path: Path) -> None:
    out = read_text_file("../outside.txt", tmp_path)
    assert out.startswith("ERROR:")


def test_list_workspace_files(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "dir" / "b.txt"
    venv_file = tmp_path / ".venv" / "skip.py"
    file_b.parent.mkdir(parents=True)
    venv_file.parent.mkdir(parents=True)
    file_a.write_text("A", encoding="utf-8")
    file_b.write_text("B", encoding="utf-8")
    venv_file.write_text("print('x')", encoding="utf-8")

    listing = list_workspace_files(".", tmp_path)
    assert "a.txt" in listing
    assert "dir/b.txt" in listing
    assert ".venv/skip.py" not in listing


def test_list_workspace_files_with_glob(tmp_path: Path) -> None:
    py_file = tmp_path / "main.py"
    txt_file = tmp_path / "notes.txt"
    py_file.write_text("print(1)", encoding="utf-8")
    txt_file.write_text("hello", encoding="utf-8")

    listing = list_workspace_files(".|*.py", tmp_path)
    assert "main.py" in listing
    assert "notes.txt" not in listing


def test_lexical_jaccard_similarity() -> None:
    sim = lexical_jaccard_similarity("alpha beta gamma", "alpha beta")
    assert 0 < sim < 1

