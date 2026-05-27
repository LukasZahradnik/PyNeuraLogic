# Contributing to PyNeuraLogic

Thank you for considering contributing to the PyNeuraLogic project. Every contribution is welcome.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating — we expect everyone to follow it.

You can contribute in many ways: with code, [reporting bugs](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%5B%F0%9F%90%9B+Bug+Report%5D%3A+),
[proposing new features](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=enhancement&template=feature_request.yaml&title=%5B%E2%9C%A8+Feature+Request%5D%3A+),
participating in [discussions](https://github.com/LukasZahradnik/PyNeuraLogic/discussions), improving documentation, or writing tests.

---

## How can I contribute

### 🐛 I came across a bug

If you encounter a bug, first make sure you are using the latest version of PyNeuraLogic. Check the [existing issues](https://github.com/LukasZahradnik/PyNeuraLogic/issues) to see if it's already known. Then you can either:

- [Report the bug](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%5B%F0%9F%90%9B+Bug+Report%5D%3A+) — we'll take a look.
- Fix it yourself — check the issue list first so nobody else is already working on it.

For security vulnerabilities, please **do not** open a public issue. See [SECURITY.md](SECURITY.md) for responsible disclosure.

### ✨ I am missing some features

Open a new issue with a description of the feature you'd like. If you're unsure whether it fits the scope of PyNeuraLogic, start a [discussion](https://github.com/LukasZahradnik/PyNeuraLogic/discussions) instead.

Once the issue is created and the feature is relevant, someone can be assigned to implement it — and you're welcome to contribute the implementation yourself.

### 📖 I want to contribute to the documentation

Documentation improvements are always appreciated — you don't need to open an issue for small fixes or typos.

For larger changes (e.g., a new tutorial or section), please open an issue or start a [discussion](https://github.com/LukasZahradnik/PyNeuraLogic/discussions) first so we can coordinate.

The documentation source lives in the `docs/` directory and uses Sphinx with reStructuredText. See [Building documentation](#building-documentation) below.

### 🤔 Something else

If your contribution doesn't fit the categories above or you're unsure, open a [blank issue](https://github.com/LukasZahradnik/PyNeuraLogic/issues/new) or ask us in [discussions](https://github.com/LukasZahradnik/PyNeuraLogic/discussions).

---

## Development workflow

### 1. Set up your environment

```bash
# Fork & clone the repository
git clone git@github.com:<Your_Username>/PyNeuraLogic.git
cd PyNeuraLogic

# Install dependencies (pick one)
uv sync --dev                               # using uv (recommended)
pip install -e ".[dev]"                     # using pip (alternative)

# Install pre-commit hooks (runs ruff on every commit)
pre-commit install
```

The `dev` extra installs `pytest`, `ruff`, `pyright`, and `pre-commit`.

### 2. Make your changes

Write your code, add or update tests, and commit:

```bash
git checkout -b my-feature-branch
# ... make changes ...
git add .
git commit -m "A meaningful commit message"
```

Pre-commit hooks will automatically format your code with **ruff** on each commit. If formatting fails, the commit is blocked — just run `git commit` again to retry with fixes applied.

### 3. Run linters and type checks

Before pushing, run the same checks as CI:

```bash
ruff format --check neuralogic    # formatting
ruff check neuralogic             # linting
pyright neuralogic                # type checking
```

Fix any issues. You can auto-fix most lint and formatting issues with:

```bash
ruff format neuralogic
ruff check --fix neuralogic
```

### 4. Run tests

```bash
pytest
```

If you added new functionality, please add corresponding tests in the `tests/` directory. Make sure the full suite passes.

### 5. Update the changelog

Add an entry under the `[Unreleased]` section in [CHANGELOG.md](CHANGELOG.md) describing your change. Follow the existing format (one line per change, grouped by **Added** / **Changed** / **Fixed** / **Removed**).

### 6. Build the documentation (if applicable)

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` in your browser to verify the output.

### 7. Submit a pull request

Push your branch and open a pull request. The [PR template](.github/PULL_REQUEST_TEMPLATE.md) will guide you through the description and checklist.

GitHub Actions will run tests, linting, and type checks automatically on your PR. Ensure all checks pass before requesting a review.

---

## Project structure

```
PyNeuraLogic/
├── neuralogic/          # Library source code
│   ├── core/            # Core constructs, builder, settings
│   ├── nn/              # Neural network modules & training
│   ├── dataset/         # Dataset loaders (CSV, PDDL, logic, etc.)
│   ├── optim/           # Optimizers & LR schedulers
│   ├── logging/         # Logging utilities
│   └── utils/           # Data utilities & visualization
├── tests/               # Test suite (pytest)
├── examples/            # Jupyter notebooks & scripts
├── benchmarks/          # Performance benchmarks
└── docs/                # Sphinx documentation
```
