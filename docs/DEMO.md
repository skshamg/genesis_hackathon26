# Demo + Video Run Sheet

## Live demo
**Run:** `python scripts/run_api.py`

Show:
- graph input
- predicted Sybil scores
- trusted benign seed concept
- structural features / trust propagation

Avoid running the full adversarial training live unless you have already benchmarked the runtime.

## Video / recorded research demo
**Run:** `python scripts/run_facebook_eval.py`

Recommended recording order:
1. Project title + problem.
2. Synthetic attacker graph generation.
3. Model training variants.
4. Gradient-adversarial training explanation.
5. Unseen Facebook graph transfer.
6. AUC + accuracy table.
7. Short limitation statement: one real benchmark does not establish universal robustness.
