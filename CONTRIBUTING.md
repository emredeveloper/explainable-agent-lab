# Contributing

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Run tests

```bash
python -m unittest discover
```

If your Windows temp directory is restricted, run with a local base temp:

```bash
python -m unittest discover
```

## Project conventions

- Keep tools deterministic and guarded.
- Keep traces compact and human-readable.
- Prefer adding tests for behavior changes.
- Use Turkish output in agent responses unless a benchmark requires English prompts.
