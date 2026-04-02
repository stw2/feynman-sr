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

from evaluate import EQUATIONS, load_equation_data, r_squared, is_exact

# ============================================================================
# HYPERPARAMETERS — agents should tune these
# ============================================================================
POPULATION_SIZE = 500
GENERATIONS = 80
TOURNAMENT_SIZE = 5
MAX_DEPTH = 6
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
REPRODUCTION_PROB = 0.1
PARSIMONY_COEFF = 0.001  # penalize large trees
ELITISM = 10  # top-N individuals survive unchanged
TIME_BUDGET_PER_EQ = 60  # seconds per equation

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
    "exp": lambda x: np.clip(np.exp(np.clip(x, -50, 50)), -1e15, 1e15),
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
    if UNARY_OPS and rng.random() < 0.25:
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


def fitness(node: Node, X: np.ndarray, y: np.ndarray,
            var_names: list[str]) -> float:
    """Fitness = negative MSE - parsimony penalty. Higher is better."""
    try:
        y_pred = evaluate_tree(node, X, var_names)
        y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
        mse = np.mean((y - y_pred) ** 2)
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


def tournament_select(rng: np.random.Generator, population: list[Node],
                      fitnesses: list[float], k: int) -> Node:
    """Tournament selection."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ============================================================================
# EVOLUTION
# ============================================================================

def evolve(X_train: np.ndarray, y_train: np.ndarray,
           variables: list[str], seed: int = 0,
           time_budget: float = TIME_BUDGET_PER_EQ) -> tuple[Node, list[float]]:
    """Run GP evolution. Returns (best_tree, fitness_history)."""
    rng = np.random.default_rng(seed)
    start = time.time()

    # Initialize
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

        # Track best
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_ever = population[gen_best_idx].copy()

        history.append(best_fitness)

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

        # Fill rest via genetic operators
        while len(next_pop) < POPULATION_SIZE:
            r = rng.random()
            if r < CROSSOVER_PROB:
                p1 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                p2 = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = crossover(rng, p1, p2, MAX_DEPTH)
            elif r < CROSSOVER_PROB + MUTATION_PROB:
                parent = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE)
                child = mutate(rng, parent, variables, MAX_DEPTH)
            else:
                child = tournament_select(rng, population, fitnesses, TOURNAMENT_SIZE).copy()
            next_pop.append(child)

        population = next_pop

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

    y_pred_train = evaluate_tree(best_tree, X_train, eq["variables"])
    y_pred_test = evaluate_tree(best_tree, X_test, eq["variables"])

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
