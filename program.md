# Feynman Symbolic Regression

You are a research agent. Your goal: improve a symbolic regression system
that rediscovers physics equations from data.

## The problem

50 physics equations are encoded as numeric datasets in `equations/`. Each
equation has training and test data: input variables X and output values y.
The ground-truth formulas are hidden — you must discover them from data.

Run `python evaluate.py` (via `from evaluate import list_equations`) or
inspect `equations/metadata.json` to see available equations (IDs, names,
variable names, difficulty tiers 1–4).

## The system

`train.py` implements symbolic regression via genetic programming. It is
the **only file you modify**. Run experiments with:

```bash
./run.sh --equation eq01             # single equation
./run.sh --tier 1                    # all equations in a tier
./run.sh --all                       # full benchmark (50 equations)
```

Results are in `run.log`. Key metrics: `r2_test` (fit quality),
`exact_match` (perfect recovery), `nodes` (expression complexity).

Other files (`evaluate.py`, `substrate_client.py`, `run.sh`) are read-only.

## How you work

**One hypothesis per session. One branch per investigation.**

You investigate a single, specific claim about how to improve the system.
You design experiments to test it, run them, and report what you found.

**At the start of every session you MUST:**
1. Ensure you are on `main` (`git checkout main && git pull`)
2. Create a new branch: `git checkout -b substrate/agent/<slug>`
   — the slug should describe your hypothesis (e.g. `arcsin-operator`)
3. Read prior nanopubs so you don't repeat work
4. Run the benchmark to establish a baseline

**Never commit directly to main.** The loop harness decides whether to
merge your branch based on benchmark results. All your commits go on your
investigation branch.

Then form a hypothesis, implement changes to `train.py`, gather evidence,
and publish your findings.

## Coordination

You may be working alongside other agents. Substrate coordinates the work.

**At session start** — read the room state and prior findings:
```bash
python substrate_client.py context    # room state: backlog, investigations, nanopubs
python substrate_client.py backlog    # open tasks / hypotheses
```
Also read `.substrate/nanopubs/*.json` for prior findings. These are your
externalized memory. Both positive and negative results matter.

**Register what you're working on** so others don't duplicate:
```bash
python substrate_client.py investigate "title" "hypothesis" "branch-name"
```

**Publish a nanopub when done** — even if your results are negative:
```bash
python substrate_client.py nanopub-template <eq_id> "hypothesis"  # get template
# Fill in results, save to .substrate/nanopubs/<id>.json
python substrate_client.py publish .substrate/nanopubs/<id>.json
python substrate_client.py complete <investigation_id>
```

If a backlog item matches what you want to investigate, claim it first:
```bash
python substrate_client.py claim <backlog_item_id>
```

If Substrate is not configured (no `.env`), work locally — read/write
nanopubs directly in `.substrate/nanopubs/`.

## Nanopub format

Your nanopub should capture:
- What you hypothesized and why
- What you changed in `train.py`
- Quantitative results (R², exact matches, which equations, before/after)
- Whether the hypothesis was **supported**, **refuted**, or **inconclusive**
- Links to prior nanopubs your finding supports or contradicts

Use `python substrate_client.py nanopub-template` for the JSON structure.

## Rules

1. Only modify `train.py`.
2. Never commit to `main` — always work on your investigation branch.
3. Never ask for permission to continue. Run your full investigation.
4. Always publish a nanopub at the end.
5. Keep `train.py` runnable — don't break `./run.sh`.
6. Commit and push your changes before publishing the nanopub.
