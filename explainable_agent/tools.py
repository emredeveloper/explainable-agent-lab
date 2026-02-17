from __future__ import annotations

import ast
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ToolFn = Callable[[str, Path], str]
TOOL_SCHEMA_VERSION = "tool-spec-v1"
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "runs",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    usage_hint: str
    fn: ToolFn
    requires_input: bool = True


def _safe_resolve_path(workspace_root: Path, input_path: str) -> Path:
    candidate = (workspace_root / input_path).resolve()
    workspace = workspace_root.resolve()
    if candidate == workspace:
        return candidate
    if workspace not in candidate.parents:
        raise ValueError("Yol, calisma klasoru disina cikiyor.")
    return candidate


def calculate_math(expression: str, _: Path) -> str:
    expression = expression.strip()
    if not expression:
        return "ERROR: bos ifade"

    allowed_bin_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
    }
    allowed_unary_ops = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            return allowed_unary_ops[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin_ops:
            return allowed_bin_ops[type(node.op)](
                eval_node(node.left), eval_node(node.right)
            )
        raise ValueError("Yalnizca temel aritmetik islemler desteklenir.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    return str(result)


def read_text_file(input_path: str, workspace_root: Path) -> str:
    try:
        path = _safe_resolve_path(workspace_root, input_path.strip())
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not path.exists():
        return f"ERROR: dosya bulunamadi: {input_path}"
    if path.is_dir():
        return f"ERROR: yol bir klasor: {input_path}"

    content = path.read_text(encoding="utf-8", errors="replace")
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]..."
    return content


def list_workspace_files(input_path: str, workspace_root: Path) -> str:
    rel, pattern = _parse_list_input(input_path)
    try:
        path = _safe_resolve_path(workspace_root, rel)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not path.exists():
        return f"ERROR: path not found: {rel}"
    if not path.is_dir():
        return f"ERROR: yol bir klasor degil: {rel}"

    files: list[str] = []
    for candidate in path.rglob(pattern):
        if not candidate.is_file():
            continue
        rel_parts = candidate.relative_to(workspace_root).parts
        if any(part in IGNORED_DIRS for part in rel_parts):
            continue
        files.append(candidate.relative_to(workspace_root).as_posix())
    files = sorted(files)
    max_items = 100
    if len(files) > max_items:
        shown = files[:max_items] + [f"...[truncated {len(files)-max_items} files]..."]
    else:
        shown = files
    return "\n".join(shown) if shown else "(bos)"


def _parse_list_input(raw_input: str) -> tuple[str, str]:
    # Format: "<path>|<glob_pattern>", examples: "." or ".|*.py"
    text = raw_input.strip()
    if not text:
        return ".", "*"
    if "|" not in text:
        return text, "*"
    rel, pattern = text.split("|", 1)
    rel = rel.strip() or "."
    pattern = pattern.strip() or "*"
    return rel, pattern


def now_utc(_: str, __: Path) -> str:
    return datetime.now(timezone.utc).isoformat()




def _resolve_sqlite_db_path(workspace_root: Path) -> Path:
    raw = os.getenv("AGENT_SQLITE_DB", "data/agent.db").strip() or "data/agent.db"
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
        workspace = workspace_root.resolve()
        if workspace not in candidate.parents and candidate != workspace:
            raise ValueError("SQLite DB path must be inside workspace.")
        return candidate
    return _safe_resolve_path(workspace_root, raw)


def sqlite_init_demo(_: str, workspace_root: Path) -> str:
    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    city TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                );

                DELETE FROM orders;
                DELETE FROM customers;

                INSERT INTO customers (id, name, city) VALUES
                    (1, 'Acme Corp', 'Istanbul'),
                    (2, 'Northwind', 'Ankara'),
                    (3, 'Blue Ocean', 'Izmir');

                INSERT INTO orders (id, customer_id, amount, status, created_at) VALUES
                    (101, 1, 1200.50, 'paid', '2026-02-01'),
                    (102, 2, 340.00, 'pending', '2026-02-05'),
                    (103, 1, 89.99, 'paid', '2026-02-07'),
                    (104, 3, 560.10, 'cancelled', '2026-02-08');
                """
            )
            conn.commit()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_init_demo failed: {exc}"

    return f"OK: demo sqlite veritabani olusturuldu: {db_path.as_posix()}"


def sqlite_list_tables(_: str, workspace_root: Path) -> str:
    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite veritabani bulunamadi: {db_path.as_posix()}. "
            "Once sqlite_init_demo calistirin."
        )
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_list_tables basarisiz: {exc}"

    names = [row[0] for row in rows if row and row[0] != "sqlite_sequence"]
    if not names:
        return "TABLOLAR: (yok)"
    return "TABLOLAR:\n" + "\n".join(names)


def sqlite_describe_table(table_name: str, workspace_root: Path) -> str:
    name = table_name.strip()
    if not name:
        return "ERROR: tablo adi bos."
    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite veritabani bulunamadi: {db_path.as_posix()}. "
            "Once sqlite_init_demo calistirin."
        )
    if not _is_safe_sql_identifier(name):
        return "ERROR: gecersiz tablo adi."
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({name})")
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_describe_table basarisiz: {exc}"
    if not rows:
        return f"ERROR: tablo bulunamadi: {name}"

    lines = ["KOLONLAR: cid | name | type | notnull | default | pk"]
    for row in rows:
        lines.append(" | ".join(str(col) for col in row))
    return "\n".join(lines)


def sqlite_query(query: str, workspace_root: Path) -> str:
    sql = query.strip()
    if not sql:
        return "ERROR: SQL sorgusu bos."
    if not _is_read_only_sql(sql):
        return "ERROR: sqlite_query yalnizca salt-okuma sorgularini destekler (SELECT/PRAGMA/WITH/EXPLAIN)."

    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite veritabani bulunamadi: {db_path.as_posix()}. "
            "Once sqlite_init_demo calistirin."
        )

    max_rows = 50
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            columns = [col[0] for col in (cursor.description or [])]
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_query basarisiz: {exc}"

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    lines: list[str] = []
    lines.append(f"KOLONLAR: {' | '.join(columns) if columns else '(yok)'}")
    lines.append("SATIRLAR:")
    if not rows:
        lines.append("(bos)")
    else:
        for row in rows:
            lines.append(" | ".join(_format_sql_cell(cell) for cell in row))
    lines.append(f"SATIR_SAYISI: {len(rows)}")
    if truncated:
        lines.append(f"NOT: ilk {max_rows} satir gosterildi.")
    return "\n".join(lines)


def sqlite_execute(sql_script: str, workspace_root: Path) -> str:
    sql = sql_script.strip()
    if not sql:
        return "ERROR: SQL ifadesi bos."
    if _is_read_only_sql(sql):
        return "ERROR: Salt-okuma SQL icin sqlite_query kullanin."
    first_token = _first_sql_token(sql)
    if first_token in {"attach", "detach"}:
        return "ERROR: ATTACH/DETACH desteklenmiyor."

    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(sql)
            conn.commit()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_execute basarisiz: {exc}"
    return f"OK: sqlite_execute uygulandi: {db_path.as_posix()}"


def _first_sql_token(sql: str) -> str:
    tokens = sql.strip().split()
    return tokens[0].lower() if tokens else ""


def _is_read_only_sql(sql: str) -> bool:
    return _first_sql_token(sql) in {"select", "pragma", "with", "explain"}


def _is_safe_sql_identifier(name: str) -> bool:
    return name.replace("_", "").isalnum()


def _format_sql_cell(cell: object) -> str:
    text = str(cell)
    return text.replace("\n", " ").replace("\r", " ")


AVAILABLE_TOOLS: dict[str, ToolSpec] = {
    "calculate_math": ToolSpec(
        name="calculate_math",
        description="Temel aritmetik ifadeyi hesaplar.",
        usage_hint="Girdi bir ifade metnidir, ornek: (12.5*4)-7",
        fn=calculate_math,
    ),
    "read_text_file": ToolSpec(
        name="read_text_file",
        description="Calisma klasorundeki UTF-8 metin dosyasini okur.",
        usage_hint="Girdi goreli dosya yoludur, ornek: docs/notes.txt",
        fn=read_text_file,
    ),
    "list_workspace_files": ToolSpec(
        name="list_workspace_files",
        description="Calisma klasorunde dosyalari listeler.",
        usage_hint="Girdi '<yol>|<glob>' (glob opsiyonel), ornek: '.' veya '.|*.py'",
        fn=list_workspace_files,
    ),
    "now_utc": ToolSpec(
        name="now_utc",
        description="Guncel UTC zamanini ISO formatinda dondurur.",
        usage_hint="Girdi bos olabilir.",
        fn=now_utc,
        requires_input=False,
    ),
    "sqlite_init_demo": ToolSpec(
        name="sqlite_init_demo",
        description="Ornek customers/orders tablolariyla demo SQLite DB olusturur.",
        usage_hint="Girdi bos olabilir.",
        fn=sqlite_init_demo,
        requires_input=False,
    ),
    "sqlite_list_tables": ToolSpec(
        name="sqlite_list_tables",
        description="Ayarlanan DB dosyasindaki SQLite tablolarini listeler.",
        usage_hint="Girdi bos olabilir.",
        fn=sqlite_list_tables,
        requires_input=False,
    ),
    "sqlite_describe_table": ToolSpec(
        name="sqlite_describe_table",
        description="SQLite tablo semasini gosterir.",
        usage_hint="Girdi tablo adidir, ornek: customers",
        fn=sqlite_describe_table,
    ),
    "sqlite_query": ToolSpec(
        name="sqlite_query",
        description="Salt-okuma SQLite sorgusu calistirir (SELECT/PRAGMA/WITH/EXPLAIN).",
        usage_hint="Girdi SQL sorgu metnidir.",
        fn=sqlite_query,
    ),
    "sqlite_execute": ToolSpec(
        name="sqlite_execute",
        description="SQLite yazma SQL'lerini calistirir (CREATE/INSERT/UPDATE/DELETE).",
        usage_hint="Girdi SQL script metnidir.",
        fn=sqlite_execute,
    ),
}


def tool_catalog_text() -> str:
    lines: list[str] = []
    lines.append(f"[schema_version={TOOL_SCHEMA_VERSION}]")
    for spec in AVAILABLE_TOOLS.values():
        lines.append(f"- {spec.name}: {spec.description} ({spec.usage_hint})")
    return "\n".join(lines)


def tool_catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "usage_hint": spec.usage_hint,
                "requires_input": spec.requires_input,
            }
            for spec in AVAILABLE_TOOLS.values()
        ],
    }


def available_tool_names() -> set[str]:
    return set(AVAILABLE_TOOLS.keys())


def tools_without_input() -> set[str]:
    return {name for name, spec in AVAILABLE_TOOLS.items() if not spec.requires_input}


def run_tool(tool_name: str, tool_input: str, workspace_root: Path) -> str:
    spec = AVAILABLE_TOOLS.get(tool_name)
    if not spec:
        return f"ERROR: bilinmeyen arac '{tool_name}'."
    return spec.fn(tool_input or "", workspace_root)
