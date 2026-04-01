# Contributing

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Validation

```bash
python -m py_compile explainable_agent\__init__.py explainable_agent\config.py explainable_agent\cli.py explainable_agent\agent.py explainable_agent\tools.py explainable_agent\openai_client.py
```

For a broader prepublish pass, run:

```bash
python scripts\prepublish_check.py
```

## Project conventions

- Keep tools deterministic and guarded.
- Keep traces compact and human-readable.
- Prefer lightweight validation steps and runnable examples for behavior changes.
- Use Turkish output in agent responses unless a benchmark requires English prompts.
