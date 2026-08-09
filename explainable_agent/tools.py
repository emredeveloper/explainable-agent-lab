from __future__ import annotations

import ast
import os
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ToolFn = Callable[[str, Path], str]
TOOL_SCHEMA_VERSION = "tool-spec-v1"
SEARCH_TIMEOUT_SECONDS = 10
SEARCH_MAX_RESULTS = 5
READ_ONLY_SQL_STATEMENTS = {"select", "with", "explain"}
ALLOWED_WRITE_STATEMENTS = {"create", "insert", "update", "delete"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "runs",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    usage_hint: str
    fn: ToolFn
    requires_input: bool = True


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    @classmethod
    def from_global(cls) -> ToolRegistry:
        return cls(specs=dict(AVAILABLE_TOOLS))

    def define_tool(
        self, name: str, description: str, usage_hint: str, requires_input: bool = True
    ) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self.specs[name] = ToolSpec(
                name=name,
                description=description,
                usage_hint=usage_hint,
                fn=fn,
                requires_input=requires_input,
            )
            return fn

        return decorator

    def run(self, tool_name: str, tool_input: str, workspace_root: Path) -> str:
        return run_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            workspace_root=workspace_root,
            registry=self.specs,
        )

    def names(self) -> set[str]:
        return available_tool_names(self.specs)

    def without_input(self) -> set[str]:
        return tools_without_input(self.specs)

    def catalog_text(self) -> str:
        return tool_catalog_text(self.specs)

    def catalog_payload(self) -> dict[str, Any]:
        return tool_catalog_payload(self.specs)


AVAILABLE_TOOLS: dict[str, ToolSpec] = {}


def define_tool(
    name: str, description: str, usage_hint: str, requires_input: bool = True
) -> Callable[[ToolFn], ToolFn]:
    """Decorator to register a tool in the AVAILABLE_TOOLS catalog."""

    def decorator(fn: ToolFn) -> ToolFn:
        AVAILABLE_TOOLS[name] = ToolSpec(
            name=name,
            description=description,
            usage_hint=usage_hint,
            fn=fn,
            requires_input=requires_input,
        )
        return fn

    return decorator


def _safe_resolve_path(workspace_root: Path, input_path: str) -> Path:
    candidate = (workspace_root / input_path).resolve()
    workspace = workspace_root.resolve()
    if candidate == workspace:
        return candidate
    if workspace not in candidate.parents:
        raise ValueError("Path escapes workspace directory.")
    return candidate


@define_tool(
    name="calculate_math",
    description="Calculates a basic arithmetic expression.",
    usage_hint="Input is an expression string, e.g., (12.5*4)-7",
)
def calculate_math(expression: str, _: Path) -> str:
    expression = expression.strip()
    if not expression:
        return "ERROR: empty expression"

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
        raise ValueError("Only basic arithmetic operations are supported.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    return str(result)


@define_tool(
    name="read_text_file",
    description="Reads a UTF-8 text file from the workspace.",
    usage_hint="Input is a relative file path, e.g., docs/notes.txt",
)
def read_text_file(input_path: str, workspace_root: Path) -> str:
    try:
        path = _safe_resolve_path(workspace_root, input_path.strip())
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not path.exists():
        return f"ERROR: file not found: {input_path}"
    if path.is_dir():
        return f"ERROR: path is a directory: {input_path}"

    content = path.read_text(encoding="utf-8", errors="replace")
    max_chars = 4000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncated]..."
    return content


@define_tool(
    name="list_workspace_files",
    description="Lists files in the workspace directory.",
    usage_hint="Input is '<path>|<glob>' (glob is optional), e.g., '.' or '.|*.py'",
)
def list_workspace_files(input_path: str, workspace_root: Path) -> str:
    rel, pattern = _parse_list_input(input_path)
    try:
        path = _safe_resolve_path(workspace_root, rel)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not path.exists():
        return f"ERROR: path not found: {rel}"
    if not path.is_dir():
        return f"ERROR: path is not a directory: {rel}"

    resolved_root = workspace_root.resolve()
    files: list[str] = []
    for candidate in path.rglob(pattern):
        if not candidate.is_file():
            continue
        resolved_candidate = candidate.resolve()
        rel_parts = resolved_candidate.relative_to(resolved_root).parts
        if any(part in IGNORED_DIRS for part in rel_parts):
            continue
        files.append(resolved_candidate.relative_to(resolved_root).as_posix())
    files = sorted(files)
    max_items = 100
    if len(files) > max_items:
        shown = files[:max_items] + [
            f"...[truncated {len(files) - max_items} files]..."
        ]
    else:
        shown = files
    return "\n".join(shown) if shown else "(empty)"


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


@define_tool(
    name="now_utc",
    description="Returns current UTC time in ISO format.",
    usage_hint="Input can be empty.",
    requires_input=False,
)
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


@define_tool(
    name="sqlite_init_demo",
    description="Creates a demo SQLite DB with sample customers/orders tables.",
    usage_hint="Input can be empty.",
    requires_input=False,
)
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

    return f"OK: demo sqlite database created: {db_path.as_posix()}"


@define_tool(
    name="sqlite_list_tables",
    description="Lists SQLite tables in the configured DB file.",
    usage_hint="Input can be empty.",
    requires_input=False,
)
def sqlite_list_tables(_: str, workspace_root: Path) -> str:
    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite database not found: {db_path.as_posix()}. "
            "Run sqlite_init_demo first."
        )
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_list_tables failed: {exc}"

    names = [row[0] for row in rows if row and row[0] != "sqlite_sequence"]
    if not names:
        return "TABLES: (none)"
    return "TABLES:\n" + "\n".join(names)


@define_tool(
    name="sqlite_describe_table",
    description="Shows the schema of an SQLite table.",
    usage_hint="Input is the table name, e.g., customers",
)
def sqlite_describe_table(table_name: str, workspace_root: Path) -> str:
    name = table_name.strip()
    if not name:
        return "ERROR: table name is empty."
    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite database not found: {db_path.as_posix()}. "
            "Run sqlite_init_demo first."
        )
    if not _is_safe_sql_identifier(name):
        return "ERROR: invalid table name."
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({name})")
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_describe_table failed: {exc}"
    if not rows:
        return f"ERROR: table not found: {name}"

    lines = ["COLUMNS: cid | name | type | notnull | default | pk"]
    for row in rows:
        lines.append(" | ".join(str(col) for col in row))
    return "\n".join(lines)


@define_tool(
    name="sqlite_query",
    description="Executes a read-only SQLite query (SELECT/WITH/EXPLAIN).",
    usage_hint="Input is an SQL query string.",
)
def sqlite_query(query: str, workspace_root: Path) -> str:
    sql = query.strip()
    if not sql:
        return "ERROR: SQL query is empty."
    if not _is_read_only_sql(sql):
        return (
            "ERROR: sqlite_query only supports read-only queries (SELECT/WITH/EXPLAIN)."
        )

    try:
        db_path = _resolve_sqlite_db_path(workspace_root)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not db_path.exists():
        return (
            f"ERROR: sqlite database not found: {db_path.as_posix()}. "
            "Run sqlite_init_demo first."
        )

    max_rows = 50
    try:
        with _connect_sqlite_read_only(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            columns = [col[0] for col in (cursor.description or [])]
    except sqlite3.Error as exc:
        return f"ERROR: sqlite_query failed: {exc}"

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    lines: list[str] = []
    lines.append(f"COLUMNS: {' | '.join(columns) if columns else '(none)'}")
    lines.append("ROWS:")
    if not rows:
        lines.append("(empty)")
    else:
        for row in rows:
            lines.append(" | ".join(_format_sql_cell(cell) for cell in row))
    lines.append(f"ROW_COUNT: {len(rows)}")
    if truncated:
        lines.append(f"NOTE: only the first {max_rows} rows are shown.")
    return "\n".join(lines)


@define_tool(
    name="sqlite_execute",
    description="Executes SQLite write statements (CREATE/INSERT/UPDATE/DELETE).",
    usage_hint="Input is an SQL script string.",
)
def sqlite_execute(sql_script: str, workspace_root: Path) -> str:
    sql = sql_script.strip()
    if not sql:
        return "ERROR: SQL statement is empty."
    if _is_read_only_sql(sql):
        return "ERROR: Use sqlite_query for read-only SQL."

    # The script is run via executescript(), so every statement must be checked.
    # Validating only the leading token would let "INSERT ...; DROP TABLE x;"
    # smuggle a destructive statement past the guard.
    statements = _split_sql_statements(sql)
    if not statements:
        return "ERROR: SQL statement is empty."
    for statement in statements:
        token = _first_sql_token(statement)
        if token not in ALLOWED_WRITE_STATEMENTS:
            return (
                "ERROR: sqlite_execute only supports CREATE/INSERT/UPDATE/DELETE "
                f"statements (rejected: '{token or statement[:20]}')."
            )

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
        return f"ERROR: sqlite_execute failed: {exc}"
    return f"OK: sqlite_execute applied: {db_path.as_posix()}"


def _first_sql_token(sql: str) -> str:
    tokens = sql.strip().split()
    return tokens[0].lower() if tokens else ""


def _strip_sql_comments(sql: str) -> str:
    """Remove -- line comments and /* */ block comments outside string literals."""
    out: list[str] = []
    i = 0
    length = len(sql)
    quote: str | None = None
    while i < length:
        ch = sql[i]
        if quote:
            out.append(ch)
            if ch == quote:
                # Doubled quote is an escaped quote, not a terminator.
                if i + 1 < length and sql[i + 1] == quote:
                    out.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < length and sql[i + 1] == "-":
            while i < length and sql[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < length and sql[i + 1] == "*":
            i += 2
            while i + 1 < length and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script on semicolons that sit outside string literals."""
    cleaned = _strip_sql_comments(sql)
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    length = len(cleaned)
    while i < length:
        ch = cleaned[i]
        if quote:
            current.append(ch)
            if ch == quote:
                if i + 1 < length and cleaned[i + 1] == quote:
                    current.append(cleaned[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
            current.append(ch)
            i += 1
            continue
        if ch == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def _is_read_only_sql(sql: str) -> bool:
    return _first_sql_token(sql) in READ_ONLY_SQL_STATEMENTS


def _connect_sqlite_read_only(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }

    def _authorizer(action: int, *_args: object) -> int:
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(_authorizer)
    return conn


def _is_safe_sql_identifier(name: str) -> bool:
    return name.replace("_", "").isalnum()


def _format_sql_cell(cell: object) -> str:
    text = str(cell)
    return text.replace("\n", " ").replace("\r", " ")


@define_tool(
    name="duckduckgo_search",
    description="Performs real-time web search via DuckDuckGo.",
    usage_hint="Input is the search query, e.g., Python 3.12 features",
)
def duckduckgo_search(query: str, _: Path) -> str:
    query = query.strip()
    if not query:
        return "ERROR: Search query is empty."
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        # Bound the network call so a stalled search cannot hang the agent.
        # Older duckduckgo-search builds may not accept the keyword.
        try:
            ddgs = DDGS(timeout=SEARCH_TIMEOUT_SECONDS)
        except TypeError:
            ddgs = DDGS()
        results = list(ddgs.text(query, max_results=SEARCH_MAX_RESULTS))
    except ImportError:
        return "ERROR: ddgs or duckduckgo-search package is not installed. Run: pip install ddgs"
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"

    if not results:
        return "Not found."

    lines = [f"Search Results ('{query}'):", ""]
    for i, res in enumerate(results, 1):
        lines.append(f"{i}. {res.get('title', 'No Title')}")
        lines.append(f"   URL: {res.get('href', 'No URL')}")
        snippet = res.get("body", "")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        lines.append(f"   Summary: {snippet}")
        lines.append("")

    return "\n".join(lines)


def openai_tool_definitions(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool definitions for native function calling."""
    defs: list[dict[str, Any]] = []
    for spec in _coerce_registry(registry).values():
        params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        if spec.requires_input:
            params["properties"]["input"] = {
                "type": "string",
                "description": spec.usage_hint,
            }
            params["required"] = ["input"]
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": params,
                },
            }
        )
    return defs


def _coerce_registry(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> Mapping[str, ToolSpec]:
    if isinstance(registry, ToolRegistry):
        return registry.specs
    return registry or AVAILABLE_TOOLS


def tool_catalog_text(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> str:
    specs = _coerce_registry(registry)
    lines: list[str] = []
    lines.append(f"[schema_version={TOOL_SCHEMA_VERSION}]")
    for spec in specs.values():
        lines.append(f"- {spec.name}: {spec.description} ({spec.usage_hint})")
    return "\n".join(lines)


def tool_catalog_payload(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> dict[str, Any]:
    specs = _coerce_registry(registry)
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "usage_hint": spec.usage_hint,
                "requires_input": spec.requires_input,
            }
            for spec in specs.values()
        ],
    }


def available_tool_names(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> set[str]:
    return set(_coerce_registry(registry).keys())


def tools_without_input(
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> set[str]:
    return {
        name
        for name, spec in _coerce_registry(registry).items()
        if not spec.requires_input
    }


def run_tool(
    tool_name: str,
    tool_input: str,
    workspace_root: Path,
    registry: Mapping[str, ToolSpec] | ToolRegistry | None = None,
) -> str:
    spec = _coerce_registry(registry).get(tool_name)
    if not spec:
        return f"ERROR: unknown tool '{tool_name}'."
    return spec.fn(tool_input or "", workspace_root)
