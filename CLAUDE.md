# Feynman Symbolic Regression — Agent Instructions

You are a research agent investigating symbolic regression on the Feynman
physics equation benchmark. Your goal: rediscover closed-form physics
equations from data using genetic programming (GP).

## Your task

Each session you **study one hypothesis**. A hypothesis is a concrete,
testable claim about how to improve the symbolic regression system.
Examples:
- "Adding sin/cos operators will enable discovery of oscillatory equations"
- "Increasing population to 1000 with tournament size 7 improves Tier 2 recovery"
- "Subtree crossover with depth-proportional selection reduces bloat"
- "Adding a constant-optimization pass (gradient descent on ERC values) after
  GP improves R² on equations with irrational constants"

## Files

| File | Role |
|---|---|
| `train.py` | **ONLY file you modify.** GP implementation. |
| `prepare.py` | Equation catalogue, data generation, evaluation. READ ONLY. |
| `substrate_client.py` | Substrate API client. READ ONLY. |
| `run.sh` | Runs train.py, saves to run.log. |

## Session workflow

### 1. Read context

Fetch the room state from Substrate to see what others have done and what
hypotheses are available:

```bash
python substrate_client.py context
```

Read the backlog to find pre-registered hypotheses waiting to be tested:

```bash
python substrate_client.py backlog
```

Also read any existing nanopubs in `.substrate/nanopubs/` to learn from
prior investigations (both successful and failed). These are your
externalized memory — do not repeat work that has already been done.

### 2. Pick a hypothesis

Either:
- **Claim** a backlog item: `python substrate_client.py claim <id>`
- **Propose** your own hypothesis based on gaps in existing nanopubs

Choose a hypothesis that is:
- Specific and testable (not "make GP better")
- Scoped to what you can test in one session
- Informed by prior nanopubs (don't repeat failed approaches unless
  you have a specific reason the prior attempt was flawed)

### 3. Register your investigation

```bash
python substrate_client.py investigate \
  "Title of investigation" \
  "Your hypothesis statement" \
  "substrate/<agent-name>/hypothesis-slug" \
  --backlog-item-id <optional>
```

This tells other agents what you're working on so they don't duplicate.
Send heartbeats periodically during long-running experiments:

```bash
python substrate_client.py heartbeat <investigation_id>
```

### 4. Implement & experiment

Modify **only** `train.py` to test your hypothesis. Common changes:
- Add/remove operators in `BINARY_OPS` or `UNARY_OPS`
- Tune hyperparameters (`POPULATION_SIZE`, `GENERATIONS`, `MAX_DEPTH`, etc.)
- Change selection method (tournament → lexicase, etc.)
- Add new GP operators (point mutation, hoist, etc.)
- Add post-GP optimization (constant tuning, simplification)
- Change tree initialization strategy
- Add diversity maintenance (fitness sharing, niching)

Run experiments:
```bash
./run.sh --tier 1                    # quick smoke test on easy equations
./run.sh --equation eq13             # test specific hard equation
./run.sh --tier 4                    # test on hard (trig/exp) equations
./run.sh --all                       # full benchmark run
```

**Build evidence.** Don't just run once. Test your hypothesis from
multiple angles:
- Run on equations where you expect improvement AND where you don't
- Compare before/after on the same equations
- Note both positive results and negative ones
- If results are noisy, run multiple seeds

### 5. Analyze results

Read `run.log` carefully. Look at:
- `r2_test` — how well does the discovered expression generalize?
- `exact_match` — did GP recover the exact symbolic form?
- `nodes` — expression complexity (simpler is better)
- `time_s` — computational cost
- `summary_exact` and `summary_mean_r2` — overall performance

### 6. Publish your findings

Generate a nanopub template:
```bash
python substrate_client.py nanopub-template <equation_id> "your hypothesis"
```

Fill in the template with your actual results and save it. Key fields:
- `assertion.statement`: one sentence describing what you found
- `assertion.claim`: `supported` | `refuted` | `inconclusive`
- `provenance.evidence.metrics`: actual numbers from your experiments
- `substrate.links.supports` / `attacks`: reference IDs of prior nanopubs
  that your finding supports or contradicts

Publish:
```bash
python substrate_client.py publish .substrate/nanopubs/<id>.json
```

### 7. Complete your investigation

```bash
python substrate_client.py complete <investigation_id>
```

## Equation tiers

| Tier | Equations | Count | Operators needed | Difficulty |
|------|-----------|-------|-----------------|------------|
| 1 | eq01–eq10 | 10 | `+`, `-`, `*`, `/` | Easy baseline |
| 2 | eq11–eq22 | 12 | `+`, `-`, `*`, `/`, `**` | Division/powers |
| 3 | eq23–eq32 | 10 | `sqrt` | Square roots |
| 4 | eq33–eq50 | 18 | `sin`, `cos`, `exp`, `log`, `arcsin` | Transcendental |

50 equations total. The baseline `train.py` only has `add`, `sub`, `mul`,
`div`, `neg`. Tier 3–4 equations are **impossible** without extending the
operator set.

## Rules

1. Only modify `train.py`. Never touch `prepare.py`, `substrate_client.py`,
   or `run.sh`.
2. Never ask for permission to continue. Run your full investigation.
3. Always publish a nanopub at the end, even if results are negative.
   Negative results prevent other agents from wasting time.
4. Keep `train.py` working — it must always be runnable via `./run.sh`.
5. Prefer simplicity. A 0.01 R² improvement that adds 100 lines of
   complexity is usually not worth it.
6. Commit your train.py changes before publishing the nanopub.
7. If Substrate is not configured (no env vars), work locally:
   read/write nanopubs directly in `.substrate/nanopubs/`.

## Baseline performance

The baseline should achieve:
- Tier 1: most equations solved exactly (R² > 0.99)
- Tier 2: some equations solved (R² > 0.9)
- Tier 3: poor without `sqrt` operator
- Tier 4: fails without `sin`/`cos`/`exp`

## Suggested first hypotheses

If the backlog is empty, good starting hypotheses are:
1. Add `sqrt` operator → unlock Tier 3 equations
2. Add `sin`/`cos` operators → unlock Tier 4 oscillatory equations
3. Add `exp` operator → unlock Tier 4 exponential equations
4. Increase population to 1000 → improve Tier 2 exact recovery rate
5. Add constant optimization (gradient descent on ERCs after GP converges)
6. Replace tournament with lexicase selection for better generalization
7. Add `pow` operator (x^y) → more flexible than x*x patterns
