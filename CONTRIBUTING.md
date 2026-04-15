# Contributing

Contributions are welcome. Please follow these guidelines:

## Setup

```bash
git clone https://github.com/seetrex-ai/monolith.git
cd monolith
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

## Before submitting

1. All fast tests must pass: `pytest tests/ -v -m "not slow"`
2. Code should follow existing style (no docstrings on every function, keep it concise)
3. New features need tests

## Pull requests

- One logical change per PR
- Descriptive commit messages
- Reference any relevant issues
