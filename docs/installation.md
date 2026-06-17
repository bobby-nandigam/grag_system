# Installation

## Requirements

Python 3.10+

---

## Basic Installation

```bash
pip install grag-system
```

---

## Full Installation

```bash
pip install "grag-system[all]"
```

---

## Development Setup

Clone the repository.

```bash
git clone https://github.com/bobby-nandigam/grag_system.git

cd grag_system
```

Create environment.

```bash
python -m venv .venv

source .venv/bin/activate
```

Install package.

```bash
pip install -e ".[dev]"
```

---

## Verify Installation

```python
import grag

print("GRAG loaded")
```