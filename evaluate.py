"""Feynman Symbolic Regression Benchmark — evaluation & data loading.

This file is safe for agents to read. It contains NO ground-truth formulas.
Agents import from here (not from prepare.py).

Usage from train.py:
    from evaluate import EQUATIONS, load_equation_data, r_squared, is_exact
"""

import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Equation metadata (loaded from generated metadata.json — no formulas)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "equations")
_METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

EQUATIONS: dict[str, dict] = {}


def _load_metadata():
    """Load equation metadata from the generated metadata.json file."""
    global EQUATIONS
    if EQUATIONS:
        return
    if not os.path.exists(_METADATA_PATH):
        print(
            "ERROR: equations/metadata.json not found.\n"
            "Run 'python prepare.py' first to generate equation data.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(_METADATA_PATH) as f:
        EQUATIONS.update(json.load(f))


# Auto-load on import
_load_metadata()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_equation_data(eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (X_train, y_train, X_test, y_test) for an equation."""
    path = os.path.join(DATA_DIR, f"{eid}.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 'python prepare.py' first.", file=sys.stderr)
        sys.exit(1)
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def list_equations():
    """Print equation catalogue (no formulas)."""
    _load_metadata()
    print(f"{'ID':<8} {'Tier':<6} {'Vars':<6} {'Name':<40} {'Variables'}")
    print("-" * 95)
    for eid in sorted(EQUATIONS):
        eq = EQUATIONS[eid]
        nvars = len(eq["variables"])
        varlist = ", ".join(eq["variables"])
        print(f"{eid:<8} {eq['tier']:<6} {nvars:<6} {eq['name']:<40} {varlist}")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def is_exact(y_true: np.ndarray, y_pred: np.ndarray, rtol: float = 1e-4) -> bool:
    """Check if prediction matches ground truth within relative tolerance."""
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    return bool(np.allclose(y_true, y_pred, rtol=rtol, atol=1e-8))
