# Repository Guidelines

## Project Structure & Module Organization
- `player.py` is the primary bot entry point and strategy driver.
- `skeleton/` contains the game engine interfaces (actions, states, runner, bot base class).
- `stats*.py`, `strategy.py`, and `rough_rank.py` hold supporting poker math and heuristics.
- `gamelog.txt` captures sample run output; `notes` contains research ideas.
- `pyproject.toml` defines Python 3.11+ dependencies for local setup.

## Build, Test, and Development Commands
- `python3 player.py` runs the bot locally using the skeleton runner.
- No dedicated build command is defined; install dependencies from `pyproject.toml` with your preferred Python tool (e.g., `uv`, `pip`, or `poetry`).

## Coding Style & Naming Conventions
- Use 4-space indentation and standard Python conventions (PEP 8 style, snake_case for functions and variables, CapWords for classes).
- Keep logic in `player.py` focused on decisions; place reusable helpers in `stats*.py` or `strategy.py`.
- There is no enforced formatter or linter in this repo; keep edits minimal and consistent with existing code.

## Testing Guidelines
- There is no formal test suite currently. Validate changes by running `python3 player.py` and reviewing `gamelog.txt`.
- If you add tests, keep them lightweight and document the command to run them here.

## Commit & Pull Request Guidelines
- Git history uses short, informal messages (e.g., "notes written"). Prefer clear, concise summaries such as `tune discard equity` or `refactor action selection`.
- PRs should include a brief description of the behavior change, any relevant logs, and how to run the bot locally.

## Configuration Tips
- Python 3.11+ is required per `pyproject.toml`.
- Keep third-party libraries updated in `pyproject.toml` and avoid adding heavyweight dependencies unless necessary.
- The submission archive must be <1GB (ideally <100MB compressed). Build PokerStove with SWIG, run `pipx run build` (or `python -m build`) in `documentation/pokerstove-src`, and include the resulting wheel in the package so the scrimmage server can install it without compiling.
- Set `NEUROPOKER_EVAL_BACKEND=pokerstove` and point `PYTHONPATH` at the wheel if you want the scipy backend in production; `stats.py` auto-detects PokerStove when available.
