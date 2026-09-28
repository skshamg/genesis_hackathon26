import numpy as np, time
from sklearn.metrics import roc_auc_score, f1_score
from sybil_shield.experiments.benchmark import make_sybil_graph, count_attack_edges, HONEST_FAMILIES, ALL_FAMILIES
from sybil_shield.core.detector import build_features, trust_scores, sweep_cut, pick_seeds
from sybil_shield.core.gcn import GCN

def cw(y):
    c = np.bincount(y, minlength=2).astype(float); return len(y) / (2 * c)

def make_train(n=36, seed=1):
    rng = np.random.default_rng(seed); gs = []
    for _ in range(n):
        g = int(np.exp(rng.uniform(np.log(3), np.log(300))))
        gs.append(make_sybil_graph(
            n_honest=int(rng.integers(220, 340)), honest_family=str(rng.choice(HONEST_FAMILIES)),
            sybil_family=str(rng.choice(ALL_FAMILIES)), n_farms=int(rng.integers(2, 5)),
            attack_edges=g, mimic=bool(rng.integers(0, 2)),
            target_honest_hubs=bool(rng.integers(0, 2)), seed=int(rng.integers(0, 2**31))))
    return gs

def train_gcn(graphs, **fk):
    prep = [build_features(g, seed=i, **fk) for i, g in enumerate(graphs)]
    m = GCN(n_features=prep[0][0].shape[1], hidden_dims=(16, 16), seed=0)
    for ep in range(120):
        for X, y, A, *_ in prep:
            m.fit_step(A, X, y, class_weight=cw(y), lr=0.02)
    return m

def main():
    t0 = time.time()
    train = make_train()
    variants = {"GCN struct-only": dict(use_trust=False), "GCN trust-only": dict(use_struct=False),
                "GCN trust+struct": dict()}
    models = {k: train_gcn(train, **v) for k, v in variants.items()}
    print(f"trained in {time.time()-t0:.0f}s")

    levels = [5, 10, 20, 40, 80, 160, 320]
    for split, fams in [("IN-DIST honest (ba/ws/sbm)", HONEST_FAMILIES), ("UNSEEN honest family (er)", ["er"])]:
        print(f"\n=== {split}: mean AUC (10 graphs/level, random sybil family, mimicry on) ===")
        names = ["PPR raw", "PPR/deg", "SybilRank"] + list(models) + ["sweep-cut F1"]
        print(f"{'attack_edges':>12s} " + " ".join(f"{n:>16s}" for n in names))
        for g in levels:
            res = {n: [] for n in names}
            for r in range(10):
                rng = np.random.default_rng(10_000 + g * 100 + r)
                G = make_sybil_graph(n_honest=int(rng.integers(250, 340)),
                    honest_family=str(rng.choice(fams)), sybil_family=str(rng.choice(ALL_FAMILIES)),
                    n_farms=3, attack_edges=g, mimic=True, seed=int(rng.integers(0, 2**31)))
                nodes = list(G.nodes()); y = np.array([G.nodes[n]["label"] for n in nodes])
                ts = trust_scores(G, nodes, pick_seeds(G, r))
                res["PPR raw"].append(roc_auc_score(y, -ts["ppr0.8"]))
                res["PPR/deg"].append(roc_auc_score(y, -ts["ppr0.8/deg"]))
                res["SybilRank"].append(roc_auc_score(y, -ts["sybilrank"]))
                mask, phi, cut = sweep_cut(G, nodes, ts["ppr0.8/deg"])
                res["sweep-cut F1"].append(f1_score(y, mask.astype(int)))
                for k, v in variants.items():
                    X, y2, A, *_ = build_features(G, seed=r, **v)
                    res[k].append(roc_auc_score(y2, models[k].predict_proba(A, X)[:, 1]))
            print(f"{g:>12d} " + " ".join(f"{np.mean(res[n]):>10.3f}±{np.std(res[n]):.2f}" for n in names))
if __name__ == "__main__":
    main()
