from pathlib import Path

from explainable_agent.tools import (
    sqlite_describe_table,
    sqlite_execute,
    sqlite_init_demo,
    sqlite_list_tables,
    sqlite_query,
)


def test_sqlite_demo_lifecycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/demo.db")

    init_out = sqlite_init_demo("", tmp_path)
    assert init_out.startswith("OK:")

    tables = sqlite_list_tables("", tmp_path)
    assert "customers" in tables
    assert "orders" in tables

    describe = sqlite_describe_table("customers", tmp_path)
    assert "COLUMNS:" in describe
    assert "name" in describe

    query_out = sqlite_query(
        "SELECT name, city FROM customers ORDER BY id;",
        tmp_path,
    )
    assert "Acme Corp" in query_out
    assert "ROW_COUNT:" in query_out

    execute_out = sqlite_execute(
        "INSERT INTO customers (id, name, city) VALUES (10, 'Delta', 'Bursa');",
        tmp_path,
    )
    assert execute_out.startswith("OK:")

    count_out = sqlite_query("SELECT COUNT(*) FROM customers;", tmp_path)
    assert "4" in count_out


def test_sqlite_read_write_guards(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SQLITE_DB", "data/demo.db")
    sqlite_init_demo("", tmp_path)

    read_guard = sqlite_query("DELETE FROM customers;", tmp_path)
    assert read_guard.startswith("ERROR:")

    write_guard = sqlite_execute("SELECT * FROM customers;", tmp_path)
    assert write_guard.startswith("ERROR:")
