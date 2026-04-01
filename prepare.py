"""Feynman Symbolic Regression Benchmark — data preparation & evaluation.

DO NOT MODIFY THIS FILE. Agents should only modify train.py.

Usage:
    python prepare.py                  # generate all equation datasets
    python prepare.py --list           # list available equations
    python prepare.py --equation eq01  # generate one equation dataset
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Equation catalogue
# Each entry: (name, source, formula_str, variable_names, variable_ranges, func)
# variable_ranges: list of (lo, hi) tuples — uniform sampling
# ---------------------------------------------------------------------------

EQUATIONS: dict[str, dict] = {}


def _reg(eid, name, source, formula_str, variables, ranges, func, *, tier):
    EQUATIONS[eid] = dict(
        id=eid,
        name=name,
        source=source,
        formula=formula_str,
        variables=variables,
        ranges=ranges,
        func=func,
        tier=tier,
    )


# === Tier 1 — Very Easy (2–3 vars, basic arithmetic) ========================

_reg("eq01", "Kinetic Energy", "Feynman I.15.1",
     "0.5 * m * v**2", ["m", "v"], [(1, 5), (1, 5)],
     lambda m, v: 0.5 * m * v ** 2, tier=1)

_reg("eq02", "Ohm's Law (current)", "Ohm's Law",
     "V / R", ["V", "R"], [(1, 10), (0.5, 10)],
     lambda V, R: V / R, tier=1)

_reg("eq03", "Gravitational PE", "Feynman I.12.11",
     "m * g * h", ["m", "g", "h"], [(1, 5), (9, 11), (1, 10)],
     lambda m, g, h: m * g * h, tier=1)

_reg("eq04", "Ideal Gas (P)", "Ideal Gas Law",
     "n * T / V", ["n", "T", "V"], [(1, 5), (200, 400), (1, 10)],
     lambda n, T, V: n * T / V, tier=1)

_reg("eq05", "Distance (uniform accel)", "Kinematics",
     "v * t + 0.5 * a * t**2", ["v", "t", "a"], [(0, 10), (0.1, 5), (0.1, 5)],
     lambda v, t, a: v * t + 0.5 * a * t ** 2, tier=1)

_reg("eq06", "Momentum", "Feynman I.15.3t",
     "m * v", ["m", "v"], [(1, 10), (1, 10)],
     lambda m, v: m * v, tier=1)

_reg("eq07", "Power (F*v)", "Feynman I.13.12",
     "F * v", ["F", "v"], [(1, 10), (1, 10)],
     lambda F, v: F * v, tier=1)

_reg("eq08", "Spring PE", "Feynman I.15.1b",
     "0.5 * k * x**2", ["k", "x"], [(1, 10), (-5, 5)],
     lambda k, x: 0.5 * k * x ** 2, tier=1)

_reg("eq09", "Work-energy", "Mechanics",
     "F * d * cos_theta", ["F", "d", "cos_theta"], [(1, 10), (1, 10), (0, 1)],
     lambda F, d, cos_theta: F * d * cos_theta, tier=1)

_reg("eq10", "Pressure (F/A)", "Feynman I.14.3",
     "F / A", ["F", "A"], [(1, 100), (0.5, 10)],
     lambda F, A: F / A, tier=1)

# === Tier 2 — Easy (2–4 vars, division, powers, compound) ===================

_reg("eq11", "Coulomb's Law", "Feynman I.12.1",
     "q1 * q2 / r**2", ["q1", "q2", "r"], [(1, 5), (1, 5), (0.5, 5)],
     lambda q1, q2, r: q1 * q2 / r ** 2, tier=2)

_reg("eq12", "Newton Gravitation", "Feynman I.12.2",
     "m1 * m2 / r**2", ["m1", "m2", "r"], [(1, 10), (1, 10), (0.5, 10)],
     lambda m1, m2, r: m1 * m2 / r ** 2, tier=2)

_reg("eq13", "Compton Scattering", "Feynman I.34.14",
     "1 / (1 - v / c) - 1", ["v", "c"], [(0.1, 0.9), (1, 1)],
     lambda v, c: 1.0 / (1.0 - v / c) - 1.0, tier=2)

_reg("eq14", "Centripetal Acceleration", "Feynman I.13.4",
     "v**2 / r", ["v", "r"], [(1, 10), (0.5, 10)],
     lambda v, r: v ** 2 / r, tier=2)

_reg("eq15", "Coulomb Potential", "Feynman I.12.4",
     "q / (4 * pi * eps * r)", ["q", "eps", "r"],
     [(1, 10), (0.5, 5), (0.5, 10)],
     lambda q, eps, r: q / (4 * np.pi * eps * r), tier=2)

_reg("eq16", "Gravitational Potential", "Feynman I.12.2b",
     "-G * m / r", ["G", "m", "r"], [(1, 10), (1, 10), (0.5, 10)],
     lambda G, m, r: -G * m / r, tier=2)

_reg("eq17", "Capacitor Energy", "Electrostatics",
     "0.5 * q**2 / C", ["q", "C"], [(1, 10), (0.5, 10)],
     lambda q, C: 0.5 * q ** 2 / C, tier=2)

_reg("eq18", "Doppler (classical)", "Feynman I.47.12",
     "f * (c + v_r) / (c + v_s)", ["f", "c", "v_r", "v_s"],
     [(100, 1000), (300, 350), (-50, 50), (-50, 50)],
     lambda f, c, v_r, v_s: f * (c + v_r) / (c + v_s), tier=2)

_reg("eq19", "Parallel Resistors", "Circuit Theory",
     "R1 * R2 / (R1 + R2)", ["R1", "R2"], [(1, 100), (1, 100)],
     lambda R1, R2: R1 * R2 / (R1 + R2), tier=2)

_reg("eq20", "Lens Equation (1/f)", "Optics",
     "1/d_o + 1/d_i", ["d_o", "d_i"], [(1, 20), (1, 20)],
     lambda d_o, d_i: 1.0 / d_o + 1.0 / d_i, tier=2)

_reg("eq21", "Reduced Mass", "Mechanics",
     "m1 * m2 / (m1 + m2)", ["m1", "m2"], [(1, 20), (1, 20)],
     lambda m1, m2: m1 * m2 / (m1 + m2), tier=2)

_reg("eq22", "Drag Force", "Fluid Mechanics",
     "0.5 * rho * v**2 * C_d * A", ["rho", "v", "C_d", "A"],
     [(0.5, 2), (1, 30), (0.1, 2), (0.1, 5)],
     lambda rho, v, C_d, A: 0.5 * rho * v ** 2 * C_d * A, tier=2)

# === Tier 3 — Medium (sqrt, abs, compound expressions) ======================

_reg("eq23", "Relativistic Mass", "Feynman I.10.7",
     "m / sqrt(1 - v**2)", ["m", "v"], [(1, 10), (0.01, 0.9)],
     lambda m, v: m / np.sqrt(1 - v ** 2), tier=3)

_reg("eq24", "Pendulum Period", "Feynman I.12.11",
     "2 * pi * sqrt(l / g)", ["l", "g"], [(0.1, 10), (5, 15)],
     lambda l, g: 2 * np.pi * np.sqrt(l / g), tier=3)

_reg("eq25", "Escape Velocity", "Orbital Mechanics",
     "sqrt(2 * G * M / r)", ["G", "M", "r"], [(1, 10), (1, 10), (1, 10)],
     lambda G, M, r: np.sqrt(2 * G * M / r), tier=3)

_reg("eq26", "Relativistic Momentum", "Feynman I.15.10",
     "m * v / sqrt(1 - v**2 / c**2)", ["m", "v", "c"],
     [(1, 10), (0.1, 0.9), (1, 1)],
     lambda m, v, c: m * v / np.sqrt(1 - v ** 2 / c ** 2), tier=3)

_reg("eq27", "RMS Speed (gas)", "Feynman I.41.16",
     "sqrt(3 * k_B * T / m)", ["k_B", "T", "m"],
     [(1, 5), (200, 500), (1, 10)],
     lambda k_B, T, m: np.sqrt(3 * k_B * T / m), tier=3)

_reg("eq28", "Orbital Velocity", "Orbital Mechanics",
     "sqrt(G * M / r)", ["G", "M", "r"], [(1, 10), (1, 100), (1, 20)],
     lambda G, M, r: np.sqrt(G * M / r), tier=3)

_reg("eq29", "Schwarzschild Radius", "General Relativity",
     "2 * G * M / c**2", ["G", "M", "c"],
     [(1, 10), (1, 10), (1, 5)],
     lambda G, M, c: 2 * G * M / c ** 2, tier=3)

_reg("eq30", "Relativistic Energy", "Feynman I.15.2t",
     "m * c**2 / sqrt(1 - v**2 / c**2)", ["m", "v", "c"],
     [(1, 5), (0.1, 0.9), (1, 1)],
     lambda m, v, c: m * c ** 2 / np.sqrt(1 - v ** 2 / c ** 2), tier=3)

_reg("eq31", "Time Dilation", "Special Relativity",
     "t / sqrt(1 - v**2 / c**2)", ["t", "v", "c"],
     [(1, 10), (0.01, 0.9), (1, 1)],
     lambda t, v, c: t / np.sqrt(1 - v ** 2 / c ** 2), tier=3)

_reg("eq32", "De Broglie Wavelength", "Quantum Mechanics",
     "h / (m * v)", ["h", "m", "v"], [(1, 10), (1, 10), (1, 10)],
     lambda h, m, v: h / (m * v), tier=3)

# === Tier 4 — Hard (trig, exp, log) =========================================

_reg("eq33", "Simple Harmonic Motion", "Feynman I.47.2",
     "A * sin(omega * t)", ["A", "omega", "t"],
     [(1, 5), (0.5, 5), (0, 6.28)],
     lambda A, omega, t: A * np.sin(omega * t), tier=4)

_reg("eq34", "Gaussian", "Feynman I.6.20b",
     "exp(-x**2 / (2 * sigma**2))", ["x", "sigma"],
     [(-3, 3), (0.5, 3)],
     lambda x, sigma: np.exp(-x ** 2 / (2 * sigma ** 2)), tier=4)

_reg("eq35", "Damped Oscillation", "Feynman I.47.6",
     "A * exp(-b * t) * cos(omega * t)",
     ["A", "b", "omega", "t"],
     [(1, 5), (0.1, 1), (1, 5), (0, 6.28)],
     lambda A, b, omega, t: A * np.exp(-b * t) * np.cos(omega * t), tier=4)

_reg("eq36", "Snell's Law (angle)", "Feynman I.26.2",
     "arcsin(n1 * sin(theta1) / n2)", ["n1", "theta1", "n2"],
     [(1, 1.5), (0.1, 1.0), (1.3, 2.0)],
     lambda n1, theta1, n2: np.arcsin(n1 * np.sin(theta1) / n2), tier=4)

_reg("eq37", "Cosine Rule", "Geometry",
     "sqrt(a**2 + b**2 - 2*a*b*cos(theta))",
     ["a", "b", "theta"], [(1, 10), (1, 10), (0.1, 3.0)],
     lambda a, b, theta: np.sqrt(a**2 + b**2 - 2*a*b*np.cos(theta)), tier=4)

_reg("eq38", "Projectile Range", "Kinematics",
     "v**2 * sin(2 * theta) / g", ["v", "theta", "g"],
     [(5, 30), (0.1, 1.5), (9, 11)],
     lambda v, theta, g: v**2 * np.sin(2 * theta) / g, tier=4)

_reg("eq39", "Radioactive Decay", "Nuclear Physics",
     "N0 * exp(-lambda_d * t)", ["N0", "lambda_d", "t"],
     [(100, 1000), (0.01, 1), (0, 10)],
     lambda N0, lambda_d, t: N0 * np.exp(-lambda_d * t), tier=4)

_reg("eq40", "Planck Radiation (simplified)", "Feynman I.41.2",
     "x**3 / (exp(x) - 1)", ["x"],
     [(0.5, 5)],
     lambda x: x**3 / (np.exp(x) - 1), tier=4)

_reg("eq41", "Wave Superposition", "Feynman I.47.8",
     "2 * A * cos(delta / 2) * sin(omega * t + delta / 2)",
     ["A", "delta", "omega", "t"],
     [(1, 5), (0, 3.14), (1, 5), (0, 6.28)],
     lambda A, delta, omega, t: 2*A*np.cos(delta/2)*np.sin(omega*t + delta/2),
     tier=4)

_reg("eq42", "Magnetic Force", "Feynman II.11.27",
     "q * v * B * sin(theta)", ["q", "v", "B", "theta"],
     [(1, 5), (1, 10), (0.1, 5), (0.1, 3.0)],
     lambda q, v, B, theta: q * v * B * np.sin(theta), tier=4)

_reg("eq43", "RC Circuit Discharge", "Feynman II.2.42",
     "V0 * exp(-t / (R * C))", ["V0", "t", "R", "C"],
     [(1, 10), (0, 10), (1, 10), (0.1, 5)],
     lambda V0, t, R, C: V0 * np.exp(-t / (R * C)), tier=4)

_reg("eq44", "Boltzmann Distribution", "Feynman I.41.8",
     "exp(-E / (k_B * T))", ["E", "k_B", "T"],
     [(0.1, 10), (1, 5), (1, 10)],
     lambda E, k_B, T: np.exp(-E / (k_B * T)), tier=4)

_reg("eq45", "Standing Wave", "Feynman I.47.10",
     "A * sin(n * pi * x / L)", ["A", "n", "x", "L"],
     [(1, 5), (1, 4), (0, 1), (1, 2)],
     lambda A, n, x, L: A * np.sin(n * np.pi * x / L), tier=4)

_reg("eq46", "Lennard-Jones Potential", "Molecular Physics",
     "4 * eps * ((sigma / r)**12 - (sigma / r)**6)",
     ["eps", "sigma", "r"],
     [(0.5, 5), (0.5, 2), (1, 5)],
     lambda eps, sigma, r: 4*eps*((sigma/r)**12 - (sigma/r)**6), tier=4)

_reg("eq47", "Logistic Growth", "Population Dynamics",
     "K / (1 + exp(-r * (t - t0)))", ["K", "r", "t", "t0"],
     [(10, 100), (0.5, 3), (0, 10), (3, 7)],
     lambda K, r, t, t0: K / (1 + np.exp(-r * (t - t0))), tier=4)

_reg("eq48", "Morse Potential", "Molecular Physics",
     "D * (1 - exp(-a * (r - r0)))**2",
     ["D", "a", "r", "r0"],
     [(1, 10), (0.5, 3), (0.5, 5), (1, 3)],
     lambda D, a, r, r0: D * (1 - np.exp(-a * (r - r0)))**2, tier=4)

_reg("eq49", "Pendulum (exact, small angle)", "Feynman I.47.2b",
     "theta0 * cos(sqrt(g / l) * t)",
     ["theta0", "g", "l", "t"],
     [(0.1, 0.5), (9, 11), (0.5, 5), (0, 6.28)],
     lambda theta0, g, l, t: theta0 * np.cos(np.sqrt(g / l) * t), tier=4)

_reg("eq50", "Blackbody Peak (Wien)", "Feynman I.41.3",
     "a / (exp(a / T) - 1)", ["a", "T"],
     [(0.5, 5), (1, 10)],
     lambda a, T: a / (np.exp(a / T) - 1), tier=4)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

SEED = 42
N_TRAIN = 200
N_TEST = 100
DATA_DIR = os.path.join(os.path.dirname(__file__), "equations")


def generate_data(eq: dict, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X.shape == (n, n_vars) and y.shape == (n,)."""
    rng = np.random.default_rng(seed)
    cols = []
    for lo, hi in eq["ranges"]:
        cols.append(rng.uniform(lo, hi, size=n))
    X = np.column_stack(cols)
    y = eq["func"](*[X[:, i] for i in range(X.shape[1])])
    return X, y


def prepare_equation(eid: str) -> None:
    eq = EQUATIONS[eid]
    os.makedirs(DATA_DIR, exist_ok=True)

    X_train, y_train = generate_data(eq, N_TRAIN, SEED)
    X_test, y_test = generate_data(eq, N_TEST, SEED + 1)

    path = os.path.join(DATA_DIR, f"{eid}.npz")
    np.savez(path, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f"  {eid}: {eq['name']} ({X_train.shape[1]} vars, tier {eq['tier']})")


def load_equation_data(eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (X_train, y_train, X_test, y_test) for an equation."""
    path = os.path.join(DATA_DIR, f"{eid}.npz")
    if not os.path.exists(path):
        prepare_equation(eid)
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


# ---------------------------------------------------------------------------
# Evaluation
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_equations():
    print(f"{'ID':<8} {'Tier':<6} {'Vars':<6} {'Name':<35} {'Formula'}")
    print("-" * 90)
    for eid, eq in sorted(EQUATIONS.items()):
        nvars = len(eq["variables"])
        print(f"{eid:<8} {eq['tier']:<6} {nvars:<6} {eq['name']:<35} {eq['formula']}")


def main():
    parser = argparse.ArgumentParser(description="Feynman SR benchmark data preparation")
    parser.add_argument("--list", action="store_true", help="list equations")
    parser.add_argument("--equation", type=str, help="prepare a single equation")
    args = parser.parse_args()

    if args.list:
        list_equations()
        return

    if args.equation:
        if args.equation not in EQUATIONS:
            print(f"Unknown equation: {args.equation}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(EQUATIONS.keys()))}", file=sys.stderr)
            sys.exit(1)
        prepare_equation(args.equation)
        return

    print(f"Preparing {len(EQUATIONS)} equations → {DATA_DIR}/")
    for eid in sorted(EQUATIONS):
        prepare_equation(eid)
    print("Done.")


if __name__ == "__main__":
    main()
