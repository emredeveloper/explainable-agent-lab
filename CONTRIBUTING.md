# Contributing

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Validation

Run the same checks CI runs, in order:

```bash
python scripts\prepublish_check.py
ruff check .
ruff format --check .
mkdir .tmp
pytest -q --basetemp=.tmp/pytest
python -m build
```

`pytest` writes its temporary files under `.tmp/`, so that directory must exist
before the run.

## Project conventions

- Keep tools deterministic and guarded.
- Keep traces compact and human-readable.
- Prefer lightweight validation steps and runnable examples for behavior changes.
- Write code, comments, prompts and user-facing output in English.
- Guard every statement a tool executes, not just the first one — see
  `sqlite_execute` for why partial validation is not enough.
