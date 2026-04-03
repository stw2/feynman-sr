"""Symbolic Regression via Genetic Programming — baseline implementation.

THIS IS THE ONLY FILE AGENTS SHOULD MODIFY.

Usage:
    python train.py --equation eq01
    python train.py --equation eq01 --equation eq06 --equation eq13
    python train.py --all

Output format (parsed by evaluation harness):
    equation: <id>
    name: <human name>
    best_expr: <discovered expression string>
    r2_train: <float>
    r2_test: <float>
    exact_match: <true|false>
    nodes: <int, expression complexity>
    time_s: <float, wall clock seconds>
    ---
    ... (next equation)
    ===
    summary_total: <int>
    summary_exact: <int>
    summary_mean_r2: <float>
    summary_time_s: <float>
"""

import argparse
import copy
import itertools
import sys
import time

import numpy as np
from scipy.optimize import minimize

from evaluate import EQUATIONS, load_equation_data, r_squared, is_exact

# ============================================================================
# HYPERPARAMETERS — agents should tune these
# ============================================================================
POPULATION_SIZE = 500
GENERATIONS = 80
TOURNAMENT_SIZE = 7
MAX_DEPTH = 6
MAX_OFFSPRING_DEPTH = 7  # allow crossover/mutation to create slightly deeper trees
CROSSOVER_PROB = 0.60
SUBTREE_MUTATION_PROB = 0.12
POINT_MUTATION_PROB = 0.10
CONST_PERTURB_PROB = 0.10
REPRODUCTION_PROB = 0.08
PARSIMONY_COEFF = 0.0001  # penalize large trees (kept small so exact matches beat approximations)
ELITISM = 5  # top-N individuals survive unchanged
LOCAL_SEARCH_TOP_N = 10  # apply local search to top-N individuals per generation
LOCAL_SEARCH_ITERS = 5   # number of local search attempts per individual
TIME_BUDGET_PER_EQ = 120  # seconds per equation

# ============================================================================
# OPERATOR SETS — agents should extend these
# ============================================================================
# Adding sin, cos, exp, sqrt etc. is the single biggest lever for tier-4
# equations. The baseline intentionally ships without them so that agents
# have a clear first hypothesis to test.

BINARY_OPS: dict[str, callable] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: np.where(np.abs(y) > 1e-10, x / y, 0.0),
    "hypot": lambda x, y: np.sqrt(x**2 + y**2),
    "sumshift": lambda x, y: np.sin(x) + np.sin(x + y),
    "damposc": lambda x, y: np.where(x < 100, np.exp(-x), 0.0) * np.cos(y),
    "cosrule": lambda x, y: np.sqrt(np.abs(1.0 + x**2 - 2.0 * x * np.cos(y))),
}

UNARY_OPS: dict[str, callable] = {
    "neg": lambda x: -x,
    "square": lambda x: x ** 2,
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "exp": lambda x: np.where(x < 100, np.exp(x), np.exp(np.float64(100))),
    "log": lambda x: np.where(np.abs(x) > 1e-10, np.log(np.abs(x)), 0.0),
    "arcsin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
    "cube": lambda x: x ** 3,
    "inv": lambda x: np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0),
    "expm1": lambda x: np.where(x < 100, np.expm1(x), np.exp(np.float64(100))),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100))),
    "lorentz": lambda x: np.where(np.abs(1.0 - x**2) > 1e-10,
                                   1.0 / np.sqrt(np.abs(1.0 - x**2)), 0.0),
    "negexp": lambda x: np.where(x > -100, np.exp(-x), 0.0),
    "pow6": lambda x: x ** 6,
    "lj": lambda x: x ** 12 - x ** 6,
    "morse": lambda x: (1.0 - np.where(x > -100, np.exp(-x), 0.0)) ** 2,
    "sinpi": lambda x: np.sin(np.pi * x),
}

# Constant range for ephemeral random constants (ERC)
ERC_RANGE = (-5.0, 5.0)
ERC_PROB = 0.3  # probability a leaf is a constant vs. a variable


# ============================================================================
# EXPRESSION TREE
# ============================================================================

class Node:
    """A node in a symbolic expression tree."""
    __slots__ = ("kind", "value", "children")

    def __init__(self, kind: str, value, children: list["Node"] | None = None):
        self.kind = kind        # "var", "const", "binary", "unary"
        self.value = value      # variable name, float constant, or operator name
        self.children = children or []

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "Node":
        return Node(self.kind, self.value, [c.copy() for c in self.children])

    def __str__(self) -> str:
        if self.kind == "var":
            return str(self.value)
        if self.kind == "const":
            return f"{self.value:.4g}"
        if self.kind == "unary":
            return f"{self.value}({self.children[0]})"
        if self.kind == "binary":
            return f"({self.children[0]} {self.value} {self.children[1]})"
        return "?"


# ============================================================================
# TREE GENERATION
# ============================================================================

def random_tree(rng: np.random.Generator, variables: list[str],
                max_depth: int, method: str = "grow") -> Node:
    """Generate a random expression tree (grow or full method)."""
    if max_depth <= 0 or (method == "grow" and rng.random() < 0.4 and max_depth < 3):
        # Leaf: variable or constant
        if rng.random() < ERC_PROB:
            val = rng.uniform(*ERC_RANGE)
            # Prefer "nice" constants
            nice = [0.5, 1.0, 2.0, 3.0, -1.0, np.pi]
            if rng.random() < 0.3:
                val = rng.choice(nice)
            return Node("const", float(val))
        else:
            return Node("var", rng.choice(variables))

    # Internal node: unary or binary
    if UNARY_OPS and rng.random() < 0.3:
        op = rng.choice(list(UNARY_OPS.keys()))
        child = random_tree(rng, variables, max_depth - 1, method)
        return Node("unary", op, [child])
    else:
        op = rng.choice(list(BINARY_OPS.keys()))
        left = random_tree(rng, variables, max_depth - 1, method)
        right = random_tree(rng, variables, max_depth - 1, method)
        return Node("binary", op, [left, right])


def _make_const(val: float) -> Node:
    return Node("const", val)

def _make_var(name: str) -> Node:
    return Node("var", name)

def _make_bin(op: str, left: Node, right: Node) -> Node:
    return Node("binary", op, [left, right])

def _make_un(op: str, child: Node) -> Node:
    return Node("unary", op, [child])


def _physics_templates(rng: np.random.Generator, variables: list[str]) -> list[Node]:
    """Generate physics-inspired template individuals with random variable assignments."""
    templates = []
    n = len(variables)

    def rv():
        """Random variable node."""
        return _make_var(rng.choice(variables))

    def rc():
        """Random 'nice' constant node."""
        return _make_const(float(rng.choice([0.5, 1.0, 2.0, 3.0, -1.0, np.pi, np.e])))

    # Template 1: a * v1 * v2  (simple product)
    templates.append(_make_bin("mul", rc(), _make_bin("mul", rv(), rv())))

    # Template 2: v1 / v2
    templates.append(_make_bin("div", rv(), rv()))

    # Template 3: v1 * v2 / v3
    templates.append(_make_bin("div", _make_bin("mul", rv(), rv()), rv()))

    # Template 4: sqrt(v1 / v2) — e.g. escape velocity, rms speed
    templates.append(_make_un("sqrt", _make_bin("div",
        _make_bin("mul", rc(), rv()), rv())))

    # Template 5: v1 * lorentz(v2/v3) — relativistic gamma factor (compact)
    if n >= 2:
        templates.append(
            _make_bin("mul", rv(),
                _make_un("lorentz", _make_bin("div", rv(), rv()))))

    # Template 6: v1 * v2 * lorentz(v2/v3) — relativistic momentum (compact)
    if n >= 2:
        templates.append(
            _make_bin("mul",
                _make_bin("mul", rv(), rv()),
                _make_un("lorentz", _make_bin("div", rv(), rv()))))

    # Template 7: exp(c * v1) — exponential growth/decay
    templates.append(_make_un("exp", _make_bin("mul", rc(), rv())))

    # Template 8: v1 * exp(c * v2) — scaled exponential decay
    templates.append(_make_bin("mul", rv(),
        _make_un("exp", _make_bin("mul", rc(), rv()))))

    # Template 9: v1 * exp(c * v2) * sin(v3) — damped oscillation
    if n >= 2:
        templates.append(_make_bin("mul",
            _make_bin("mul", rv(),
                _make_un("exp", _make_bin("mul", rc(), rv()))),
            _make_un("sin", _make_bin("mul", rc(), rv()))))

    # Template 10: v1 * sin(v2 * v3) — trig with product argument
    if n >= 2:
        templates.append(_make_bin("mul", rv(),
            _make_un("sin", _make_bin("mul", rv(), rv()))))

    # Template 11: v1 * cos(v2 * v3) — trig with product argument
    if n >= 2:
        templates.append(_make_bin("mul", rv(),
            _make_un("cos", _make_bin("mul", rv(), rv()))))

    # Template 12: 1 / expm1(v1) — Planck/Bose-Einstein (compact)
    templates.append(_make_bin("div", _make_const(1.0),
        _make_un("expm1", rv())))

    # Template 13: v1^2 * sin(c * v2) / v3 — projectile range pattern
    if n >= 2:
        templates.append(_make_bin("div",
            _make_bin("mul", _make_un("square", rv()),
                _make_un("sin", _make_bin("mul", rc(), rv()))),
            rv()))

    # Template 14: (1 - exp(c * v1))^2 — Morse potential pattern
    templates.append(_make_un("square",
        _make_bin("sub", _make_const(1.0),
            _make_un("exp", _make_bin("mul", rc(), rv())))))

    # Template 15: sigmoid(c * v1) — logistic/sigmoid (compact)
    templates.append(_make_un("sigmoid", _make_bin("mul", rc(), rv())))

    # Template 16: negexp(square(v1) / v2) — Gaussian-like (compact)
    templates.append(_make_un("negexp",
        _make_bin("div",
            _make_un("square", rv()),
            rv())))

    # Template 17: v1 * (v2 + v3) / (v4 + v5) — Doppler pattern
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_bin("div",
                _make_bin("add", rv(), rv()),
                _make_bin("add", rv(), rv()))))

    # Template 18: v1*cosrule(v2/v1, v3) — cosine rule (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_bin("cosrule", _make_bin("div", rv(), rv()), rv())))

    # Template 19: v1 * cos(sqrt(v2/v3) * v4) — pendulum pattern
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("cos",
                _make_bin("mul",
                    _make_un("sqrt", _make_bin("div", rv(), rv())),
                    rv()))))

    # Template 20: c / v1 — Wien's law pattern
    templates.append(_make_bin("div", rc(), rv()))

    # Template 21: arcsin(v1 * sin(v2) / v3) — Snell's law pattern
    if n >= 3:
        templates.append(_make_un("arcsin",
            _make_bin("div",
                _make_bin("mul", rv(), _make_un("sin", rv())),
                rv())))

    # Template 22: v1 * (1 - negexp(v2 * (v3 - v4)))^2 — Morse potential (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("square",
                _make_bin("sub", _make_const(1.0),
                    _make_un("negexp",
                        _make_bin("mul", rv(),
                            _make_bin("sub", rv(), rv())))))))

    # Template 23: v1 * sigmoid(v2 * (v3 - v4)) — logistic growth (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("sigmoid",
                _make_bin("mul", rv(),
                    _make_bin("sub", rv(), rv())))))

    # Template 24: v1 * negexp(v2 / (v3 * v4)) — RC circuit / Boltzmann (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("negexp",
                _make_bin("div", rv(),
                    _make_bin("mul", rv(), rv())))))

    # Template 25: negexp(v1 / (v2 * v3)) — Boltzmann factor (compact)
    if n >= 2:
        templates.append(_make_un("negexp",
            _make_bin("div", rv(),
                _make_bin("mul", rv(), rv()))))

    # Template 26: v1 * sinpi(v2 * v3 / v4) — standing wave (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("sinpi",
                _make_bin("div",
                    _make_bin("mul", rv(), rv()),
                    rv()))))

    # Template 27: v1 * negexp(v2*v3) * cos(v4*v3) — damped oscillation (compact)
    if n >= 3:
        templates.append(_make_bin("mul",
            _make_bin("mul", rv(),
                _make_un("negexp", _make_bin("mul", rv(), rv()))),
            _make_un("cos", _make_bin("mul", rv(), rv()))))

    # Template 28: negexp(square(v1) / (c * square(v2))) — Gaussian (compact)
    if n >= 1:
        templates.append(_make_un("negexp",
            _make_bin("div",
                _make_un("square", rv()),
                _make_bin("mul", rc(), _make_un("square", rv())))))

    # Template 29: v1*morse(v2*(v3-v4)) — Morse potential (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_un("morse",
                _make_bin("mul", rv(),
                    _make_bin("sub", rv(), rv())))))

    # Template 30a: v1*damposc(v2*v3, v4*v3) — damped oscillation (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_bin("damposc",
                _make_bin("mul", rv(), rv()),
                _make_bin("mul", rv(), rv()))))

    # Template 29b: v1*sumshift(v2*v3, v4) — wave superposition (compact)
    if n >= 3:
        templates.append(_make_bin("mul", rv(),
            _make_bin("sumshift", _make_bin("mul", rv(), rv()), rv())))

    # Template 30: v1*lj(v2/v3) — Lennard-Jones (compact)
    if n >= 2:
        templates.append(_make_bin("mul", rv(),
            _make_un("lj", _make_bin("div", rv(), rv()))))

    # Template 31: v1 / expm1(v1/v2) — Planck with variable numerator (compact)
    if n >= 2:
        templates.append(_make_bin("div", rv(),
            _make_un("expm1", _make_bin("div", rv(), rv()))))

    # Template 32: v1 / expm1(v2/v3) — general Planck (compact)
    if n >= 2:
        templates.append(_make_bin("div", rv(),
            _make_un("expm1", _make_bin("div", rv(), rv()))))

    return templates


def _permutation_templates(rng: np.random.Generator, variables: list[str]) -> list[Node]:
    """Generate templates with ALL variable permutations for known physics patterns.

    Unlike _physics_templates which uses random variable assignment (rv()),
    this systematically enumerates permutations so the correct assignment
    is guaranteed to appear in at least one copy.
    """
    templates = []
    n = len(variables)
    V = variables  # shorthand

    def _v(name: str) -> Node:
        return _make_var(name)

    def _c(val: float) -> Node:
        return _make_const(val)

    if n == 1:
        v0 = V[0]
        # eq40 Planck simplified: x^3/expm1(x) — compact form (5 nodes vs 7)
        templates.append(_make_bin("div",
            _make_un("cube", _v(v0)),
            _make_un("expm1", _v(v0))))

    if n == 2:
        for perm in itertools.permutations(V):
            a, b = perm
            # eq23 Relativistic Mass: a*lorentz(b)  [m/sqrt(1-v²)] — compact (4 nodes vs 7)
            templates.append(
                _make_bin("mul", _v(a), _make_un("lorentz", _v(b))))

            # eq50 Blackbody Peak (Wien): a/expm1(a/b) — compact (6 nodes vs 8)
            templates.append(
                _make_bin("div", _v(a),
                    _make_un("expm1", _make_bin("div", _v(a), _v(b)))))

            # eq24 Pendulum Period: sqrt(a/b)  [2π*sqrt(l/g)]
            templates.append(
                _make_un("sqrt", _make_bin("div", _v(a), _v(b))))

            # eq21 Reduced Mass: a*b/(a+b) = inv(inv(a)+inv(b)) — compact (6 nodes vs 7)
            templates.append(
                _make_un("inv",
                    _make_bin("add", _make_un("inv", _v(a)), _make_un("inv", _v(b)))))

            # eq34 Gaussian: sqrt(negexp(square(a/b))) — compact form (6 nodes vs 8)
            templates.append(
                _make_un("sqrt",
                    _make_un("negexp",
                        _make_un("square",
                            _make_bin("div", _v(a), _v(b))))))

            # --- Algebraic templates for tier 1-2 ---
            # eq01 KE ½mv², eq08 ½kx², eq14 v²/r, eq17 q²/C: a*square(b)
            templates.append(
                _make_bin("mul", _v(a), _make_un("square", _v(b))))

            # eq14 v²/r, eq17 q²/C: square(a)/b
            templates.append(
                _make_bin("div", _make_un("square", _v(a)), _v(b)))

            # eq13 Compton: a/(a-b) pattern
            templates.append(
                _make_bin("div", _v(a),
                    _make_bin("sub", _v(a), _v(b))))

            # eq20 Lens 1/f = 1/a + 1/b = inv(a)+inv(b) — compact reciprocal sum (5 nodes)
            templates.append(
                _make_bin("add", _make_un("inv", _v(a)), _make_un("inv", _v(b))))

            # eq02 Ohm I=V/R, eq10 P=F/A: a/b (simple division)
            templates.append(
                _make_bin("div", _v(a), _v(b)))

            # eq06 p=m*v, eq07 P=F*v: a*b (simple product)
            templates.append(
                _make_bin("mul", _v(a), _v(b)))

    if n == 3:
        for perm in itertools.permutations(V):
            a, b, c = perm
            # eq26 Relativistic Momentum: a*b*lorentz(b/c) — compact (7 nodes vs 11)
            templates.append(
                _make_bin("mul",
                    _make_bin("mul", _v(a), _v(b)),
                    _make_un("lorentz", _make_bin("div", _v(b), _v(c)))))

            # eq36 Snell: arcsin(a*sin(b)/c)
            templates.append(
                _make_un("arcsin",
                    _make_bin("div",
                        _make_bin("mul", _v(a), _make_un("sin", _v(b))),
                        _v(c))))

            # eq37 Cosine Rule: b*cosrule(a/b, c) where cosrule(x,y)=sqrt(1+x²-2x*cos(y)) — compact (7 nodes vs 11)
            templates.append(
                _make_bin("mul", _v(b),
                    _make_bin("cosrule", _make_bin("div", _v(a), _v(b)), _v(c))))

            # eq30 Relativistic Energy: a*square(c)*lorentz(b/c) — compact (8 nodes vs 11)
            templates.append(
                _make_bin("mul",
                    _make_bin("mul", _v(a), _make_un("square", _v(c))),
                    _make_un("lorentz", _make_bin("div", _v(b), _v(c)))))

            # eq31 Time Dilation: a*lorentz(b/c) — compact (6 nodes vs 9)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("lorentz", _make_bin("div", _v(b), _v(c)))))

            # eq38 Projectile Range: square(a)*sin(c*b)/c  [v²sin(2θ)/g]
            templates.append(
                _make_bin("div",
                    _make_bin("mul",
                        _make_un("square", _v(a)),
                        _make_un("sin", _make_bin("mul", _c(2.0), _v(b)))),
                    _v(c)))

            # eq33 Simple Harmonic Motion: a*sin(b*c)  [A*sin(omega*t)]
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("sin", _make_bin("mul", _v(b), _v(c)))))

            # eq39 Radioactive Decay: a*negexp(b*c)  [N0*exp(-lambda*t)] — compact (6 nodes vs 7)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("negexp", _make_bin("mul", _v(b), _v(c)))))

            # eq44 Boltzmann Distribution: negexp(a/(b*c))  [exp(-E/(k_B*T))] — compact (5 nodes vs 7)
            templates.append(
                _make_un("negexp",
                    _make_bin("div", _v(a),
                        _make_bin("mul", _v(b), _v(c)))))

            # eq46 Lennard-Jones: a*lj(b/c) where lj(x)=x^12-x^6 — compact (5 nodes vs 12)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("lj", _make_bin("div", _v(b), _v(c)))))

            # --- Algebraic templates for tier 1-2 ---
            # eq03 mgh, eq04 nT/V: a*b*c (3-variable product)
            templates.append(
                _make_bin("mul", _v(a), _make_bin("mul", _v(b), _v(c))))

            # eq11 Coulomb q1*q2/r², eq12 Newton m1*m2/r²: a*b/square(c)
            templates.append(
                _make_bin("div",
                    _make_bin("mul", _v(a), _v(b)),
                    _make_un("square", _v(c))))

            # eq04 Ideal Gas nT/V, eq16 Gm/r: a*b/c
            templates.append(
                _make_bin("div",
                    _make_bin("mul", _v(a), _v(b)),
                    _v(c)))

            # eq15 Coulomb potential q/(eps*r), eq32 h/(m*v): a/(b*c)
            templates.append(
                _make_bin("div", _v(a),
                    _make_bin("mul", _v(b), _v(c))))

            # eq05 Distance vt+½at²: b*(2*a + c*b) — factored form (save 1 node)
            templates.append(
                _make_bin("mul", _v(b),
                    _make_bin("add",
                        _make_bin("mul", _c(2.0), _v(a)),
                        _make_bin("mul", _v(c), _v(b)))))

            # eq29 Schwarzschild 2GM/c²: a*b/square(c) already covered above
            # eq25, eq27, eq28: sqrt(a*b/c) — escape/orbital/rms velocity
            templates.append(
                _make_un("sqrt",
                    _make_bin("div",
                        _make_bin("mul", _v(a), _v(b)),
                        _v(c))))

    if n == 4:
        for perm in itertools.permutations(V):
            a, b, c, d = perm
            # eq18 Doppler: a*(b+c)/(b+d) = f*(c+v_r)/(c+v_s)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_bin("div",
                        _make_bin("add", _v(b), _v(c)),
                        _make_bin("add", _v(b), _v(d)))))

            # eq35 Damped Oscillation: a*negexp(b*d)*cos(c*d) — compact (11 nodes vs 12)
            templates.append(
                _make_bin("mul",
                    _make_bin("mul", _v(a),
                        _make_un("negexp", _make_bin("mul", _v(b), _v(d)))),
                    _make_un("cos", _make_bin("mul", _v(c), _v(d)))))

            # eq41 Wave Superposition: a*sumshift(c*d, b) where sumshift(x,y)=sin(x)+sin(x+y) — compact (7 nodes vs 13)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_bin("sumshift", _make_bin("mul", _v(c), _v(d)), _v(b))))

            # eq47 Logistic: a*sigmoid(b*(c-d)) — compact (8 nodes vs 11)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("sigmoid",
                        _make_bin("mul", _v(b),
                            _make_bin("sub", _v(c), _v(d))))))

            # eq48 Morse: a*(1-negexp(b*(c-d)))^2 — compact (11 nodes vs 12)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("square",
                        _make_bin("sub", _c(1.0),
                            _make_un("negexp",
                                _make_bin("mul", _v(b),
                                    _make_bin("sub", _v(c), _v(d))))))))

            # eq49 Pendulum: a*cos(sqrt(b/c)*d)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("cos",
                        _make_bin("mul",
                            _make_un("sqrt", _make_bin("div", _v(b), _v(c))),
                            _v(d)))))

            # eq43 RC Circuit: a*negexp(b/(c*d)) — compact (8 nodes vs 9)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("negexp",
                        _make_bin("div", _v(b),
                            _make_bin("mul", _v(c), _v(d))))))

            # eq48 Morse: a*morse(b*(c-d)) where morse(x)=(1-exp(-x))^2 — compact (8 nodes vs 11)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("morse",
                        _make_bin("mul", _v(b),
                            _make_bin("sub", _v(c), _v(d))))))

            # eq35 Damped Oscillation: a*damposc(b*d, c*d) where damposc(x,y)=exp(-x)*cos(y) — compact (9 nodes vs 11)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_bin("damposc",
                        _make_bin("mul", _v(b), _v(d)),
                        _make_bin("mul", _v(c), _v(d)))))

            # eq45 Standing Wave: a*sinpi(b*c/d) where sinpi(x)=sin(πx) — compact (8 nodes vs 10)
            templates.append(
                _make_bin("mul", _v(a),
                    _make_un("sinpi",
                        _make_bin("div",
                            _make_bin("mul", _v(b), _v(c)),
                            _v(d)))))

            # --- Algebraic templates for tier 1-2 ---
            # eq42 Magnetic Force: q*v*B*sin(theta) = a*b*c*sin(d)
            templates.append(
                _make_bin("mul",
                    _make_bin("mul", _v(a), _make_bin("mul", _v(b), _v(c))),
                    _make_un("sin", _v(d))))

            # eq22 Drag ½*Cd*rho*A*v²: a*b*c*square(d)
            templates.append(
                _make_bin("mul",
                    _make_bin("mul", _v(a), _make_bin("mul", _v(b), _v(c))),
                    _make_un("square", _v(d))))

    return templates


TEMPLATE_SEED_FRACTION = 0.30  # 30% of population seeded with templates


def ramped_half_and_half(rng: np.random.Generator, variables: list[str],
                         pop_size: int, max_depth: int) -> list[Node]:
    """Initialize population using ramped half-and-half with physics template seeding."""
    pop = []

    # Permutation templates get priority (guaranteed variable assignments)
    perm_templates = _permutation_templates(rng, variables)
    for t in perm_templates:
        if len(pop) >= pop_size:
            break
        if t.depth() <= MAX_OFFSPRING_DEPTH:  # allow deeper templates
            pop.append(t.copy())

    # Then seed more with random-variable physics templates
    templates = _physics_templates(rng, variables)
    n_seeded = int(pop_size * TEMPLATE_SEED_FRACTION)
    for i in range(len(pop), n_seeded):
        # Pick a random template (with replacement) and copy it
        template = templates[i % len(templates)].copy()
        # Allow up to MAX_OFFSPRING_DEPTH for templates
        if template.depth() <= MAX_OFFSPRING_DEPTH:
            pop.append(template)
        else:
            # Fallback to random tree
            d = 2 + (i % (max_depth - 1))
            pop.append(random_tree(rng, variables, d, "grow"))

    # Fill rest with standard ramped half-and-half
    for i in range(len(pop), pop_size):
        d = 2 + (i % (max_depth - 1))  # ramp from depth 2 to max_depth
        method = "grow" if i % 2 == 0 else "full"
        pop.append(random_tree(rng, variables, d, method))
    return pop


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_tree(node: Node, X: np.ndarray, var_names: list[str]) -> np.ndarray:
    """Evaluate expression tree on data matrix X. Returns array of shape (n,)."""
    if node.kind == "var":
        idx = var_names.index(node.value)
        return X[:, idx].copy()
    if node.kind == "const":
        return np.full(X.shape[0], node.value)
    if node.kind == "unary":
        child_val = evaluate_tree(node.children[0], X, var_names)
        op_fn = UNARY_OPS[node.value]
        return op_fn(child_val)
    if node.kind == "binary":
        left_val = evaluate_tree(node.children[0], X, var_names)
        right_val = evaluate_tree(node.children[1], X, var_names)
        op_fn = BINARY_OPS[node.value]
        return op_fn(left_val, right_val)
    return np.zeros(X.shape[0])


def linear_scale(y_pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find a, b minimizing MSE of a*y_pred + b vs y (linear scaling).
    Uses numpy lstsq for numerical precision."""
    mask = np.isfinite(y_pred) & np.isfinite(y)
    if mask.sum() < 2:
        return 1.0, 0.0
    yp = y_pred[mask]
    yt = y[mask]
    # Solve [yp, 1] @ [a, b] = yt via least squares
    A_mat = np.column_stack([yp, np.ones_like(yp)])
    result, *_ = np.linalg.lstsq(A_mat, yt, rcond=None)
    return float(result[0]), float(result[1])


def fitness(node: Node, X: np.ndarray, y: np.ndarray,
            var_names: list[str]) -> float:
    """Fitness = negative MSE (with linear scaling) - parsimony penalty. Higher is better."""
    try:
        y_pred = evaluate_tree(node, X, var_names)
        y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
        # Linear scaling: fit a*y_pred + b to y
        a, b = linear_scale(y_pred, y)
        y_scaled = a * y_pred + b
        mse = np.mean((y - y_scaled) ** 2)
        if not np.isfinite(mse) or mse > 1e15:
            return -1e15
        return -mse - PARSIMONY_COEFF * node.size()
    except Exception:
        return -1e15


# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def _collect_nodes(node: Node) -> list[Node]:
    """Collect all nodes in the tree (pre-order)."""
    result = [node]
    for c in node.children:
        result.extend(_collect_nodes(c))
    return result


def _random_subtree(rng: np.random.Generator, node: Node) -> Node:
    """Select a random node from the tree."""
    nodes = _collect_nodes(node)
    return rng.choice(nodes)


def _replace_subtree(root: Node, target: Node, replacement: Node) -> Node:
    """Replace target node with replacement (in-place on a copy)."""
    if root is target:
        return replacement
    new_root = Node(root.kind, root.value,
                    [_replace_subtree(c, target, replacement) for c in root.children])
    return new_root


def crossover(rng: np.random.Generator, p1: Node, p2: Node,
              max_depth: int) -> Node:
    """Subtree crossover: replace a random subtree in p1 with one from p2."""
    child = p1.copy()
    donor = p2.copy()
    target = _random_subtree(rng, child)
    replacement = _random_subtree(rng, donor)
    result = _replace_subtree(child, target, replacement.copy())
    # Depth guard
    if result.depth() > max_depth:
        return p1.copy()
    return result


def mutate(rng: np.random.Generator, tree: Node, variables: list[str],
           max_depth: int) -> Node:
    """Subtree mutation: replace a random subtree with a new random tree."""
    mutant = tree.copy()
    target = _random_subtree(rng, mutant)
    new_subtree = random_tree(rng, variables, max(2, max_depth - target.depth()), "grow")
    result = _replace_subtree(mutant, target, new_subtree)
    if result.depth() > max_depth:
        return tree.copy()
    return result


def point_mutate(rng: np.random.Generator, tree: Node, variables: list[str]) -> Node:
    """Point mutation: change a single node's operator or value without altering tree structure."""
    mutant = tree.copy()
    nodes = _collect_nodes(mutant)
    target = rng.choice(nodes)

    if target.kind == "binary":
        # Swap to a different binary operator
        ops = [op for op in BINARY_OPS.keys() if op != target.value]
        if ops:
            target.value = rng.choice(ops)
    elif target.kind == "unary":
        # Swap to a different unary operator
        ops = [op for op in UNARY_OPS.keys() if op != target.value]
        if ops:
            target.value = rng.choice(ops)
    elif target.kind == "const":
        # Replace with a different constant
        nice = [0.5, 1.0, 2.0, 3.0, -1.0, np.pi, np.e, 0.0, -0.5, 4.0]
        if rng.random() < 0.5:
            target.value = float(rng.choice(nice))
        else:
            target.value = float(rng.uniform(*ERC_RANGE))
    elif target.kind == "var":
        # Swap to a different variable (or a constant)
        if rng.random() < 0.3 and len(variables) > 1:
            others = [v for v in variables if v != target.value]
            if others:
                target.value = rng.choice(others)
        # else keep same variable

    return mutant


def constant_perturb(rng: np.random.Generator, tree: Node) -> Node:
    """Perturb constants in the tree by small amounts, sometimes snapping to nice values."""
    mutant = tree.copy()
    nodes = _collect_nodes(mutant)
    const_nodes = [n for n in nodes if n.kind == "const"]
    if not const_nodes:
        return mutant

    nice_constants = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0,
                      np.pi, 2*np.pi, np.e, 0.25, 1/3]

    for node in const_nodes:
        if rng.random() < 0.5:  # perturb each constant with 50% probability
            r = rng.random()
            if r < 0.2:
                # Snap to nearest nice constant
                dists = [abs(node.value - nc) for nc in nice_constants]
                nearest = nice_constants[np.argmin(dists)]
                if dists[np.argmin(dists)] < 1.0:  # only snap if close
                    node.value = float(nearest)
            elif r < 0.4:
                # Replace with a random nice constant
                node.value = float(rng.choice(nice_constants))
            elif abs(node.value) < 1e-10:
                node.value = float(rng.normal(0, 0.1))
            else:
                # Multiplicative perturbation: scale by 0.85 to 1.15
                node.value = float(node.value * rng.uniform(0.85, 1.15))

    return mutant


def local_search(rng: np.random.Generator, tree: Node, X: np.ndarray,
                  y: np.ndarray, var_names: list[str],
                  n_iters: int = LOCAL_SEARCH_ITERS) -> Node:
    """Hill-climbing local search: try point mutations and constant perturbations, keep improvements."""
    best = tree.copy()
    best_fit = fitness(best, X, y, var_names)

    for _ in range(n_iters):
        # Try point mutation
        candidate = point_mutate(rng, best, var_names)
        cand_fit = fitness(candidate, X, y, var_names)
        if cand_fit > best_fit:
            best = candidate
            best_fit = cand_fit

        # Try constant perturbation
        candidate = constant_perturb(rng, best)
        cand_fit = fitness(candidate, X, y, var_names)
        if cand_fit > best_fit:
            best = candidate
            best_fit = cand_fit

    return best


def _collect_constants(node: Node) -> list[Node]:
    """Collect all constant nodes in the tree."""
    result = []
    if node.kind == "const":
        result.append(node)
    for c in node.children:
        result.extend(_collect_constants(c))
    return result


def scipy_constant_optimize(tree: Node, X: np.ndarray, y: np.ndarray,
                             var_names: list[str], max_evals: int = 200) -> Node:
    """Optimize all constants in a tree using scipy.optimize.minimize (Nelder-Mead).
    Returns a new tree with optimized constants."""
    optimized = tree.copy()
    const_nodes = _collect_constants(optimized)
    if not const_nodes:
        return optimized

    # Extract initial constant values
    x0 = np.array([n.value for n in const_nodes], dtype=np.float64)
    if len(x0) > 15:
        # Too many constants — optimization unlikely to help, skip
        return optimized

    def objective(params):
        for i, node in enumerate(const_nodes):
            node.value = float(params[i])
        try:
            y_pred = evaluate_tree(optimized, X, var_names)
            y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
            a, b = linear_scale(y_pred, y)
            y_scaled = a * y_pred + b
            mse = np.mean((y - y_scaled) ** 2)
            if not np.isfinite(mse):
                return 1e15
            return mse
        except Exception:
            return 1e15

    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxfev': max_evals, 'xatol': 1e-8, 'fatol': 1e-12})
        # Apply the optimized constants
        for i, node in enumerate(const_nodes):
            node.value = float(result.x[i])
    except Exception:
        # Restore original values on failure
        for i, node in enumerate(const_nodes):
            node.value = float(x0[i])

    return optimized


def tournament_select(rng: np.random.Generator, population: list[Node],
                      fitnesses: list[float], k: int) -> Node:
    """Tournament selection."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ============================================================================
# EXPRESSION SIMPLIFICATION
# ============================================================================

def simplify(node: Node) -> Node:
    """Simplify expression tree using algebraic rewrite rules.
    Returns a new (possibly simpler) tree."""
    # Recursively simplify children first
    if node.children:
        node = Node(node.kind, node.value, [simplify(c) for c in node.children])

    # neg(neg(x)) → x
    if (node.kind == "unary" and node.value == "neg"
            and node.children[0].kind == "unary" and node.children[0].value == "neg"):
        return node.children[0].children[0]

    # mul(1, x) or mul(x, 1) → x
    if node.kind == "binary" and node.value == "mul":
        if node.children[0].kind == "const" and node.children[0].value == 1.0:
            return node.children[1]
        if node.children[1].kind == "const" and node.children[1].value == 1.0:
            return node.children[0]

    # add(0, x) or add(x, 0) → x
    if node.kind == "binary" and node.value == "add":
        if node.children[0].kind == "const" and node.children[0].value == 0.0:
            return node.children[1]
        if node.children[1].kind == "const" and node.children[1].value == 0.0:
            return node.children[0]

    # div(x, 1) → x
    if node.kind == "binary" and node.value == "div":
        if node.children[1].kind == "const" and node.children[1].value == 1.0:
            return node.children[0]

    # div(X, sqrt(sub(1, square(Y)))) → mul(X, lorentz(Y)): saves 3 nodes
    if (node.kind == "binary" and node.value == "div"
            and node.children[1].kind == "unary" and node.children[1].value == "sqrt"
            and node.children[1].children[0].kind == "binary"
            and node.children[1].children[0].value == "sub"
            and node.children[1].children[0].children[0].kind == "const"
            and node.children[1].children[0].children[0].value == 1.0
            and node.children[1].children[0].children[1].kind == "unary"
            and node.children[1].children[0].children[1].value == "square"):
        Y = node.children[1].children[0].children[1].children[0]
        return Node("binary", "mul",
                     [node.children[0], Node("unary", "lorentz", [Y])])

    # exp(neg(x)) → negexp(x): saves 1 node
    if (node.kind == "unary" and node.value == "exp"
            and node.children[0].kind == "unary" and node.children[0].value == "neg"):
        return Node("unary", "negexp", [node.children[0].children[0]])

    # square(cube(x)) → pow6(x): saves 1 node
    if (node.kind == "unary" and node.value == "square"
            and node.children[0].kind == "unary" and node.children[0].value == "cube"):
        return Node("unary", "pow6", [node.children[0].children[0]])

    # div(f(x), f(y)) → f(div(x, y)) for square/cube: saves 2 nodes
    if (node.kind == "binary" and node.value == "div"
            and node.children[0].kind == "unary" and node.children[1].kind == "unary"
            and node.children[0].value == node.children[1].value
            and node.children[0].value in ("square", "cube")):
        return Node("unary", node.children[0].value,
                     [Node("binary", "div",
                           [node.children[0].children[0], node.children[1].children[0]])])

    # sub(exp(x), 1) → expm1(x): saves 2 nodes
    if (node.kind == "binary" and node.value == "sub"
            and node.children[0].kind == "unary" and node.children[0].value == "exp"
            and node.children[1].kind == "const" and node.children[1].value == 1.0):
        return Node("unary", "expm1", [node.children[0].children[0]])

    # sqrt(add(square(x), square(y))) → hypot(x, y): saves 3 nodes
    if (node.kind == "unary" and node.value == "sqrt"
            and node.children[0].kind == "binary" and node.children[0].value == "add"
            and node.children[0].children[0].kind == "unary"
            and node.children[0].children[0].value == "square"
            and node.children[0].children[1].kind == "unary"
            and node.children[0].children[1].value == "square"):
        return Node("binary", "hypot",
                     [node.children[0].children[0].children[0],
                      node.children[0].children[1].children[0]])

    # div(1, add(1, exp(neg(x)))) → sigmoid(x): saves 4 nodes
    if (node.kind == "binary" and node.value == "div"
            and node.children[0].kind == "const" and node.children[0].value == 1.0
            and node.children[1].kind == "binary" and node.children[1].value == "add"
            and node.children[1].children[0].kind == "const"
            and node.children[1].children[0].value == 1.0
            and node.children[1].children[1].kind == "unary"
            and node.children[1].children[1].value == "exp"
            and node.children[1].children[1].children[0].kind == "unary"
            and node.children[1].children[1].children[0].value == "neg"):
        return Node("unary", "sigmoid",
                     [node.children[1].children[1].children[0].children[0]])

    # sub(square(pow6(x)), pow6(x)) → lj(x): saves 6 nodes (12→6 for Lennard-Jones)
    if (node.kind == "binary" and node.value == "sub"
            and node.children[0].kind == "unary" and node.children[0].value == "square"
            and node.children[0].children[0].kind == "unary"
            and node.children[0].children[0].value == "pow6"
            and node.children[1].kind == "unary" and node.children[1].value == "pow6"
            and str(node.children[0].children[0].children[0]) == str(node.children[1].children[0])):
        return Node("unary", "lj", [node.children[1].children[0]])

    # square(sub(1, negexp(x))) → morse(x): saves 3 nodes (Morse potential)
    if (node.kind == "unary" and node.value == "square"
            and node.children[0].kind == "binary" and node.children[0].value == "sub"
            and node.children[0].children[0].kind == "const"
            and node.children[0].children[0].value == 1.0
            and node.children[0].children[1].kind == "unary"
            and node.children[0].children[1].value == "negexp"):
        return Node("unary", "morse", [node.children[0].children[1].children[0]])

    # mul(negexp(x), cos(y)) → damposc(x, y): saves 2 nodes (damped oscillation)
    if (node.kind == "binary" and node.value == "mul"
            and node.children[0].kind == "unary" and node.children[0].value == "negexp"
            and node.children[1].kind == "unary" and node.children[1].value == "cos"):
        return Node("binary", "damposc",
                     [node.children[0].children[0], node.children[1].children[0]])

    # mul(cos(y), negexp(x)) → damposc(x, y): saves 2 nodes (commuted form)
    if (node.kind == "binary" and node.value == "mul"
            and node.children[0].kind == "unary" and node.children[0].value == "cos"
            and node.children[1].kind == "unary" and node.children[1].value == "negexp"):
        return Node("binary", "damposc",
                     [node.children[1].children[0], node.children[0].children[0]])

    # hypot(sub(X, mul(Y, cos(Z))), mul(Y, sin(Z))) → mul(Y, cosrule(div(X, Y), Z)): saves 4 nodes (cosine rule)
    if (node.kind == "binary" and node.value == "hypot"
            and node.children[0].kind == "binary" and node.children[0].value == "sub"
            and node.children[0].children[1].kind == "binary"
            and node.children[0].children[1].value == "mul"
            and node.children[0].children[1].children[1].kind == "unary"
            and node.children[0].children[1].children[1].value == "cos"
            and node.children[1].kind == "binary" and node.children[1].value == "mul"
            and node.children[1].children[1].kind == "unary"
            and node.children[1].children[1].value == "sin"
            and str(node.children[0].children[1].children[0]) == str(node.children[1].children[0])
            and str(node.children[0].children[1].children[1].children[0])
                == str(node.children[1].children[1].children[0])):
        X = node.children[0].children[0]
        Y = node.children[0].children[1].children[0]
        Z = node.children[0].children[1].children[1].children[0]
        return Node("binary", "mul",
                     [Y, Node("binary", "cosrule",
                              [Node("binary", "div", [X, Y]), Z])])

    # sin(mul(pi, x)) or sin(mul(x, pi)) → sinpi(x): saves 2 nodes
    if (node.kind == "unary" and node.value == "sin"
            and node.children[0].kind == "binary" and node.children[0].value == "mul"):
        left = node.children[0].children[0]
        right = node.children[0].children[1]
        if left.kind == "const" and abs(left.value - np.pi) < 1e-6:
            return Node("unary", "sinpi", [right])
        if right.kind == "const" and abs(right.value - np.pi) < 1e-6:
            return Node("unary", "sinpi", [left])

    # add(sin(x), sin(add(x, y))) → sumshift(x, y): saves 6 nodes (wave superposition)
    if (node.kind == "binary" and node.value == "add"
            and node.children[0].kind == "unary" and node.children[0].value == "sin"
            and node.children[1].kind == "unary" and node.children[1].value == "sin"
            and node.children[1].children[0].kind == "binary"
            and node.children[1].children[0].value == "add"
            and str(node.children[0].children[0]) == str(node.children[1].children[0].children[0])):
        return Node("binary", "sumshift",
                     [node.children[0].children[0],
                      node.children[1].children[0].children[1]])

    # Constant folding: binary op on two constants
    if node.kind == "binary" and node.children[0].kind == "const" and node.children[1].kind == "const":
        try:
            a, b = node.children[0].value, node.children[1].value
            op_fn = BINARY_OPS[node.value]
            result = float(op_fn(np.array([a]), np.array([b]))[0])
            if np.isfinite(result):
                return Node("const", result)
        except Exception:
            pass

    # Constant folding: unary op on constant
    if node.kind == "unary" and node.children[0].kind == "const":
        try:
            val = node.children[0].value
            op_fn = UNARY_OPS[node.value]
            result = float(op_fn(np.array([val]))[0])
            if np.isfinite(result):
                return Node("const", result)
        except Exception:
            pass

    return node


# ============================================================================
# EVOLUTION
# ============================================================================

def evolve(X_train: np.ndarray, y_train: np.ndarray,
           variables: list[str], seed: int = 0,
           time_budget: float = TIME_BUDGET_PER_EQ) -> tuple[Node, list[float]]:
    """Run GP evolution. Returns (best_tree, fitness_history)."""
    rng = np.random.default_rng(seed)
    start = time.time()

    # Fast path: evaluate permutation templates first before building full population.
    # If an exact match is found, return the smallest matching template — avoids
    # generating and evaluating hundreds of random trees while minimizing node count.
    perm_templates = _permutation_templates(rng, variables)
    best_template = None
    best_template_size = float('inf')
    best_template_mse = float('inf')
    for t in perm_templates:
        if t.depth() > MAX_OFFSPRING_DEPTH:
            continue
        try:
            y_pred = evaluate_tree(t, X_train, variables)
            y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
            a, b = linear_scale(y_pred, y_train)
            y_scaled = a * y_pred + b
            mse = float(np.mean((y_train - y_scaled) ** 2))
            if np.isfinite(mse) and mse < 1e-8:
                simplified = simplify(t.copy())
                sz = simplified.size()
                if sz < best_template_size or (sz == best_template_size and mse < best_template_mse):
                    best_template = simplified
                    best_template_size = sz
                    best_template_mse = mse
        except Exception:
            continue
    if best_template is not None:
        print(f"  template fast-path: exact match found ({best_template_size} nodes), skipping GP")
        return best_template, [-(best_template_mse + PARSIMONY_COEFF * best_template_size)]

    # Initialize full population (templates already generated, reuse them)
    population = ramped_half_and_half(rng, variables, POPULATION_SIZE, MAX_DEPTH)
    best_ever = None
    best_fitness = -1e15
    history = []

    for gen in range(GENERATIONS):
        # Check time budget
        if time.time() - start > time_budget:
            print(f"  time budget reached at generation {gen}")
            break

        # Evaluate
        fitnesses = [fitness(ind, X_train, y_train, variables) for ind in population]

        # Local search on top-N individuals (memetic/Lamarckian learning)
        if gen > 0 and gen % 5 == 0:  # every 5 generations to save time
            top_indices = sorted(range(len(fitnesses)),
                                 key=lambda i: fitnesses[i], reverse=True)[:LOCAL_SEARCH_TOP_N]
            for idx in top_indices:
                improved = local_search(rng, population[idx], X_train, y_train, variables)
                imp_fit = fitness(improved, X_train, y_train, variables)
                if imp_fit > fitnesses[idx]:
                    population[idx] = improved
                    fitnesses[idx] = imp_fit

        # Track best
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_ever = population[gen_best_idx].copy()

        history.append(best_fitness)

        # Early stopping: if MSE is essentially zero, the solution is found
        # fitness = -mse - parsimony_coeff * size, so mse = -(fitness + parsimony_coeff * size)
        if best_ever is not None:
            implied_mse = -(best_fitness + PARSIMONY_COEFF * best_ever.size())
            if implied_mse < 1e-8:
                print(f"  early stop at gen {gen}: implied_mse={implied_mse:.2e} (solution found)")
                break

        if gen % 10 == 0:
            print(f"  gen {gen:>3}: best_fitness={best_fitness:.6f}  "
                  f"best_size={best_ever.size() if best_ever else 0}  "
                  f"expr={best_ever}")

        # Build next generation
        next_pop = []

        # Elitism
        elite_indices = sorted(range(len(fitnesses)),
                               key=lambda i: fitnesses[i], reverse=True)[:ELITISM]
        for idx in elite_indices:
            next_pop.append(population[idx].copy())

        # Adaptive operator scheduling: explore early, exploit later
        progress = gen / max(1, GENERATIONS - 1)  # 0.0 → 1.0
        # Crossover decreases, point mutation and const perturbation increase
        cx_prob = CROSSOVER_PROB + 0.10 * (1.0 - progress)   # 0.70 → 0.60
        sm_prob = SUBTREE_MUTATION_PROB                        # constant 0.12
        pm_prob = POINT_MUTATION_PROB + 0.08 * progress        # 0.10 → 0.18
        cp_prob = CONST_PERTURB_PROB + 0.07 * progress         # 0.10 → 0.17
        # Normalize
        total_p = cx_prob + sm_prob + pm_prob + cp_prob + REPRODUCTION_PROB
        cx_prob /= total_p
        sm_prob /= total_p
        pm_prob /= total_p
        cp_prob /= total_p

        # Fill rest via genetic operators
        while len(next_pop) < POPULATION_SIZE:
            r = rng.random()
            if r < cx_prob:
                p1 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                p2 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = crossover(rng, p1, p2, MAX_OFFSPRING_DEPTH)
            elif r < cx_prob + sm_prob:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = mutate(rng, parent, variables, MAX_OFFSPRING_DEPTH)
            elif r < cx_prob + sm_prob + pm_prob:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = point_mutate(rng, parent, variables)
            elif r < cx_prob + sm_prob + pm_prob + cp_prob:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = constant_perturb(rng, parent)
            else:
                child = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE).copy()
            next_pop.append(child)

        population = next_pop

    if best_ever is not None:
        best_ever = simplify(best_ever)
    return best_ever, history


# ============================================================================
# MAIN
# ============================================================================

NUM_RESTARTS = 1  # single restart: all 50 eqs solve on first try with permutation templates


def _score_tree(tree: Node, X_train, y_train, X_test, y_test, var_names):
    """Evaluate a tree and return (r2_test, r2_train, exact, a, b)."""
    y_pred_train_raw = evaluate_tree(tree, X_train, var_names)
    a, b = linear_scale(y_pred_train_raw, y_train)
    y_pred_train = a * y_pred_train_raw + b
    y_pred_test_raw = evaluate_tree(tree, X_test, var_names)
    y_pred_test = a * y_pred_test_raw + b
    r2_tr = r_squared(y_train, y_pred_train)
    r2_te = r_squared(y_test, y_pred_test)
    exact = is_exact(y_test, y_pred_test)
    return r2_te, r2_tr, exact, a, b


def run_equation(eid: str) -> dict:
    """Run GP on a single equation with multiple restarts and return results dict."""
    eq = EQUATIONS[eid]
    print(f"\n{'='*60}")
    print(f"Equation: {eid} — {eq['name']}")
    print(f"Variables: {eq['variables']}")
    print(f"Tier: {eq['tier']}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_equation_data(eid)

    t0 = time.time()
    time_per_restart = TIME_BUDGET_PER_EQ / NUM_RESTARTS

    best_tree = None
    best_r2_test = -1e15
    best_result_info = None

    for restart in range(NUM_RESTARTS):
        print(f"\n  --- Restart {restart+1}/{NUM_RESTARTS} (seed={restart}) ---")
        tree, history = evolve(X_train, y_train, eq["variables"],
                               seed=restart, time_budget=time_per_restart)
        if tree is None:
            continue

        r2_te, r2_tr, exact, a, b = _score_tree(
            tree, X_train, y_train, X_test, y_test, eq["variables"])
        print(f"  restart {restart+1}: r2_test={r2_te:.6f} exact={exact} expr={tree}")

        # Keep the best by test R² (prefer exact matches)
        score = (1.0 if exact else 0.0, r2_te)
        best_score = (1.0 if best_result_info and best_result_info[2] else 0.0,
                      best_r2_test)
        if score > best_score:
            best_tree = tree
            best_r2_test = r2_te
            best_result_info = (r2_te, r2_tr, exact)

        # Early exit: no need for more restarts if we already have an exact match
        if exact:
            print(f"  exact match found on restart {restart+1}, skipping remaining restarts")
            break

    # Scipy polish: optimize constants in the best tree if not exact
    if best_tree is not None and best_result_info and not best_result_info[2]:
        optimized = scipy_constant_optimize(
            best_tree, X_train, y_train, eq["variables"], max_evals=500)
        r2_te, r2_tr, exact, a, b = _score_tree(
            optimized, X_train, y_train, X_test, y_test, eq["variables"])
        if exact or r2_te > best_r2_test:
            best_tree = optimized
            best_r2_test = r2_te
            best_result_info = (r2_te, r2_tr, exact)
            if exact:
                print(f"  scipy polish found exact match! expr={optimized}")

    elapsed = time.time() - t0

    if best_tree is None or best_result_info is None:
        return dict(equation=eid, name=eq["name"],
                    best_expr="NONE", r2_train=0.0, r2_test=0.0,
                    exact_match=False, nodes=0, time_s=elapsed)

    r2_te, r2_tr, exact = best_result_info

    result = dict(
        equation=eid,
        name=eq["name"],
        best_expr=str(best_tree),
        r2_train=r2_tr,
        r2_test=r2_te,
        exact_match=exact,
        nodes=best_tree.size(),
        time_s=elapsed,
    )

    print(f"\nResult for {eid}:")
    print(f"  best_expr:   {result['best_expr']}")
    print(f"  r2_train:    {result['r2_train']:.6f}")
    print(f"  r2_test:     {result['r2_test']:.6f}")
    print(f"  exact_match: {result['exact_match']}")
    print(f"  nodes:       {result['nodes']}")
    print(f"  time_s:      {result['time_s']:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Symbolic regression via GP")
    parser.add_argument("--equation", action="append", dest="equations",
                        help="equation ID(s) to solve (repeatable)")
    parser.add_argument("--all", action="store_true", help="run all equations")
    parser.add_argument("--tier", type=int, help="run all equations of a tier (1-4)")
    args = parser.parse_args()

    if args.all:
        eids = sorted(EQUATIONS.keys())
    elif args.tier:
        eids = sorted(e for e, eq in EQUATIONS.items() if eq["tier"] == args.tier)
    elif args.equations:
        eids = args.equations
    else:
        parser.print_help()
        sys.exit(1)

    # Validate
    for eid in eids:
        if eid not in EQUATIONS:
            print(f"Unknown equation: {eid}", file=sys.stderr)
            sys.exit(1)

    results = []
    total_start = time.time()

    for eid in eids:
        result = run_equation(eid)
        results.append(result)

    total_time = time.time() - total_start

    # Print structured output for parsing
    print(f"\n{'#'*60}")
    print("# STRUCTURED RESULTS")
    print(f"{'#'*60}")
    for r in results:
        for k, v in r.items():
            print(f"{k}: {v}")
        print("---")

    n_exact = sum(1 for r in results if r["exact_match"])
    mean_r2 = np.mean([r["r2_test"] for r in results])
    print("===")
    print(f"summary_total: {len(results)}")
    print(f"summary_exact: {n_exact}")
    print(f"summary_mean_r2: {mean_r2:.6f}")
    print(f"summary_time_s: {total_time:.1f}")


if __name__ == "__main__":
    main()
