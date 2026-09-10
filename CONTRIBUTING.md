# Contributing to SageMaker Capacity-Resilient Job Launcher

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@ssh.gitlab.aws.dev:supcasa/Sagemaker_Batch_Resilient_Launcher.git
   cd Sagemaker_Batch_Resilient_Launcher
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the unit tests:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Write your code and tests.

3. Run the full test suite:
   ```bash
   pytest tests/ -v
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add support for X"
   ```

5. Push and create a merge request.

## Code Style

- Python 3.10+ with type hints.
- Follow existing code patterns and naming conventions.
- Use `_` prefix for internal modules (e.g., `_launcher.py`).
- All public functions must have docstrings.

## Testing

- **Unit tests** (`tests/test_*.py`): Use mocked boto3 clients. Zero real AWS calls. Must pass in CI.
- **Integration tests** (`tests/integration/test_real_aws.py`): Run against real AWS. Require environment variables. See the test file docstring for setup.

All new features must include unit tests. Integration tests are encouraged for cross-region or API-dependent changes.

## Architecture

The SDK uses a layered architecture:

- **Entry point** (`_entry_point.py`): Public API, parameter separation, behavior matrix routing.
- **Launcher** (`_launcher.py`): Job-type-agnostic orchestrator.
- **Strategy engines** (`_sequential_engine.py`, `_race_engine.py`): Execution strategies.
- **JobAdapter ABC** (`_job_adapter.py`): Abstract interface for job types (Batch Transform implemented, Training Jobs extensible).
- **CrossRegionManager** (`_cross_region_manager.py`): Multi-region fallback with pre-flight checks.

When adding a new job type (e.g., Training Jobs), implement `JobAdapter` — no changes needed to the strategy engines or Launcher.

## Commit Message Convention

Use conventional commit prefixes:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `refactor:` — code restructuring without behavior change

## Questions?

Open an issue in the GitLab repository or reach out to the maintainers.
