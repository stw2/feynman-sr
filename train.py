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
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.15
POINT_MUTATION_PROB = 0.15
REPRODUCTION_PROB = 0.1
PARSIMONY_COEFF = 0.001  # penalize large trees
ELITISM = 5  # top-N individuals survive unchanged
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
}

UNARY_OPS: dict[str, callable] = {
    "neg": lambda x: -x,
    "square": lambda x: x ** 2,
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "exp": lambda x: np.where(x < 100, np.exp(x), np.exp(np.float64(100))),
    "log": lambda x: np.where(np.abs(x) > 1e-10, np.log(np.abs(x)), 0.0),
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


def ramped_half_and_half(rng: np.random.Generator, variables: list[str],
                         pop_size: int, max_depth: int) -> list[Node]:
    """Initialize population using ramped half-and-half."""
    pop = []
    for i in range(pop_size):
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
# CONSTANT OPTIMIZATION
# ============================================================================

def _collect_const_nodes(node: Node) -> list[Node]:
    """Collect all constant-valued leaf nodes in the tree."""
    consts = []
    if node.kind == "const":
        consts.append(node)
    for c in node.children:
        consts.extend(_collect_const_nodes(c))
    return consts


def optimize_constants(tree: Node, X: np.ndarray, y: np.ndarray,
                       var_names: list[str], max_evals: int = 100) -> Node:
    """Optimize all numeric constants in a tree via scipy L-BFGS-B.

    Collects constant nodes, packs their values into a vector, and
    minimizes MSE (with linear scaling) over those values.
    Returns a new tree with optimized constants.
    """
    opt_tree = tree.copy()
    const_nodes = _collect_const_nodes(opt_tree)
    if not const_nodes:
        return opt_tree

    x0 = np.array([n.value for n in const_nodes], dtype=np.float64)

    def objective(params):
        for i, n in enumerate(const_nodes):
            n.value = float(params[i])
        try:
            y_pred = evaluate_tree(opt_tree, X, var_names)
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
                          options={'maxfev': max_evals, 'xatol': 1e-8, 'fatol': 1e-10})
        # Apply best params
        for i, n in enumerate(const_nodes):
            n.value = float(result.x[i])
    except Exception:
        # Restore original values on failure
        for i, n in enumerate(const_nodes):
            n.value = float(x0[i])

    return opt_tree


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
    """Point mutation: change a single node's operator or value without altering structure."""
    mutant = tree.copy()
    nodes = _collect_nodes(mutant)
    target = rng.choice(nodes)
    if target.kind == "const":
        # Perturb constant by small amount or replace with a nice constant
        if rng.random() < 0.3:
            nice = [0.5, 1.0, 2.0, 3.0, -1.0, -2.0, np.pi, np.e, 0.25, 4.0, 6.0]
            target.value = float(rng.choice(nice))
        else:
            target.value *= (1.0 + rng.normal(0, 0.1))
    elif target.kind == "var":
        target.value = rng.choice(variables)
    elif target.kind == "binary":
        target.value = rng.choice(list(BINARY_OPS.keys()))
    elif target.kind == "unary":
        target.value = rng.choice(list(UNARY_OPS.keys()))
    return mutant


def tournament_select(rng: np.random.Generator, population: list[Node],
                      fitnesses: list[float], k: int) -> Node:
    """Tournament selection."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ============================================================================
# EVOLUTION
# ============================================================================

def _single_evolve(X_train: np.ndarray, y_train: np.ndarray,
                   variables: list[str], seed: int = 0,
                   time_budget: float = 30.0) -> tuple[Node, float]:
    """Run a single GP evolution pass. Returns (best_tree, best_fitness)."""
    rng = np.random.default_rng(seed)
    start = time.time()

    population = ramped_half_and_half(rng, variables, POPULATION_SIZE, MAX_DEPTH)
    best_ever = None
    best_fitness = -1e15

    for gen in range(GENERATIONS):
        if time.time() - start > time_budget:
            break

        fitnesses = [fitness(ind, X_train, y_train, variables) for ind in population]

        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_ever = population[gen_best_idx].copy()

        if gen % 20 == 0:
            print(f"  gen {gen:>3}: best_fitness={best_fitness:.6f}  "
                  f"best_size={best_ever.size() if best_ever else 0}  "
                  f"expr={best_ever}")

        next_pop = []

        # Elitism with periodic constant optimization
        elite_indices = sorted(range(len(fitnesses)),
                               key=lambda i: fitnesses[i], reverse=True)[:ELITISM]
        for rank, idx in enumerate(elite_indices):
            elite_ind = population[idx].copy()
            if gen % 10 == 0 and rank < 2:
                if time.time() - start < time_budget - 5:
                    elite_ind = optimize_constants(elite_ind, X_train, y_train, variables)
                    opt_fit = fitness(elite_ind, X_train, y_train, variables)
                    if opt_fit > best_fitness:
                        best_fitness = opt_fit
                        best_ever = elite_ind.copy()
            next_pop.append(elite_ind)

        while len(next_pop) < POPULATION_SIZE:
            r = rng.random()
            if r < CROSSOVER_PROB:
                p1 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                p2 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = crossover(rng, p1, p2, MAX_DEPTH)
            elif r < CROSSOVER_PROB + MUTATION_PROB:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = mutate(rng, parent, variables, MAX_DEPTH)
            elif r < CROSSOVER_PROB + MUTATION_PROB + POINT_MUTATION_PROB:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = point_mutate(rng, parent, variables)
            else:
                child = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE).copy()
            next_pop.append(child)

        population = next_pop

    return best_ever, best_fitness


def evolve(X_train: np.ndarray, y_train: np.ndarray,
           variables: list[str], seed: int = 0,
           time_budget: float = TIME_BUDGET_PER_EQ) -> tuple[Node, list[float]]:
    """Run GP with multiple restarts, keeping the best result.
    Uses constant optimization on the final best tree."""
    start = time.time()
    best_ever = None
    best_fitness = -1e15
    history = []
    run_idx = 0

    while True:
        elapsed = time.time() - start
        remaining = time_budget - elapsed
        if remaining < 5:
            break

        # Each run gets a portion of remaining time, at least 15s
        per_run = max(15.0, min(remaining * 0.5, 30.0))
        run_seed = seed + run_idx * 1000

        print(f"\n  --- Restart {run_idx} (seed={run_seed}, budget={per_run:.0f}s) ---")
        tree, fit = _single_evolve(X_train, y_train, variables,
                                   seed=run_seed, time_budget=per_run)
        history.append(fit)

        if tree is not None and fit > best_fitness:
            best_fitness = fit
            best_ever = tree.copy()
            print(f"  *** New best: fitness={best_fitness:.6f} expr={best_ever}")

        run_idx += 1

    # Final constant optimization with larger budget
    if best_ever is not None:
        best_ever = optimize_constants(best_ever, X_train, y_train, variables,
                                       max_evals=500)

    return best_ever, history


# ============================================================================
# MAIN
# ============================================================================

def run_equation(eid: str) -> dict:
    """Run GP on a single equation and return results dict."""
    eq = EQUATIONS[eid]
    print(f"\n{'='*60}")
    print(f"Equation: {eid} — {eq['name']}")
    print(f"Variables: {eq['variables']}")
    print(f"Tier: {eq['tier']}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_equation_data(eid)

    t0 = time.time()
    best_tree, history = evolve(X_train, y_train, eq["variables"])
    elapsed = time.time() - t0

    if best_tree is None:
        return dict(equation=eid, name=eq["name"],
                    best_expr="NONE", r2_train=0.0, r2_test=0.0,
                    exact_match=False, nodes=0, time_s=elapsed)

    y_pred_train_raw = evaluate_tree(best_tree, X_train, eq["variables"])
    # Apply linear scaling fitted on training data
    a, b = linear_scale(y_pred_train_raw, y_train)
    y_pred_train = a * y_pred_train_raw + b
    y_pred_test_raw = evaluate_tree(best_tree, X_test, eq["variables"])
    y_pred_test = a * y_pred_test_raw + b

    r2_train = r_squared(y_train, y_pred_train)
    r2_test = r_squared(y_test, y_pred_test)
    exact = is_exact(y_test, y_pred_test)

    result = dict(
        equation=eid,
        name=eq["name"],
        best_expr=str(best_tree),
        r2_train=r2_train,
        r2_test=r2_test,
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
