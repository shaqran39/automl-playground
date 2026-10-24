---
description: Plan, test-first build, verify and commit a feature
---

Feature to build: $ARGUMENTS

1. Re-read CLAUDE.md. Plan the feature and show me the plan before writing code.
2. Write the tests first (use the test-writer agent).
3. Implement, using the relevant specialist agent.
4. Run `ruff check . --fix`, `ruff format .` and `pytest`. Fix failures.
5. Run the reviewer agent on the diff and fix anything high severity.
6. Tick the phase in CLAUDE.md if it's finished.
7. Commit with a clear message and push to main.
