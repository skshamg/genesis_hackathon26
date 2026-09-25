import numpy as np, time
from sklearn.metrics import roc_auc_score
from sybil_shield.experiments.benchmark import make_sybil_graph, HONEST_FAMILIES, ALL_FAMILIES
from sybil_shield.core.detector import build_features, trust_scores, pick_seeds
from sybil_shield.core.attacks import attack_graph
from sybil_shield.experiments.eval_robust import make_train, train_gcn, cw
from sybil_shield.core.gcn import GCN

def train_on(graphs, epochs=120):
    prep = [build_features(g, seed=i) for i, g in enumerate(graphs)]
    m = GCN(n_features=prep[0][0].shape[1], hidden_dims=(16, 16), seed=0)
    for ep in range(epochs):
        for X, y, A, *_ in prep:
            m.fit_step(A, X, y, class_weight=cw(y), lr=0.02)
    return m

def main():
    t0 = time.time()
    base = make_train(36, seed=1)                       # clean/random-attack-edge graphs
    rng = np.random.default_rng(5)
    modes = ["random", "hubs", "trusted"]
    # adversarial set: SAME 36 graphs, each additionally attacked (features recomputed at prep time)
    adv = [attack_graph(g, int(rng.integers(20, 150)), modes[i % 3], rng, seed_knowledge=i) for i, g in enumerate(base)]
    std_model = train_on(base)
    adv_model = train_on(adv)
    print(f"trained in {time.time()-t0:.0f}s (both models: 36 graphs x 120 epochs)")

    tests = []
    for r in range(12):
        rr = np.random.default_rng(777 + r)
        tests.append((make_sybil_graph(n_honest=int(rr.integers(250, 340)),
            honest_family=str(rr.choice(HONEST_FAMILIES)), sybil_family=str(rr.choice(ALL_FAMILIES)),
            n_farms=3, attack_edges=10, mimic=True, seed=int(rr.integers(0, 2**31))), r))

    def score_all(G, r):
        nodes = list(G.nodes()); y = np.array([G.nodes[n]["label"] for n in nodes])
        ts = trust_scores(G, nodes, pick_seeds(G, r))
        X, _, A, *_ = build_features(G, seed=r)
        return {"PPR/deg": roc_auc_score(y, -ts["ppr0.8/deg"]),
                "SybilRank": roc_auc_score(y, -ts["sybilrank"]),
                "GCN standard": roc_auc_score(y, std_model.predict_proba(A, X)[:, 1]),
                "GCN adv-trained": roc_auc_score(y, adv_model.predict_proba(A, X)[:, 1])}

    names = ["PPR/deg", "SybilRank", "GCN standard", "GCN adv-trained"]
    print(f"\nAUC after pipeline-level attack (base 10 attack edges + N extra; 12 graphs; features recomputed)")
    print(f"{'attack':>8s} {'extra':>6s} " + " ".join(f"{n:>16s}" for n in names))
    for mode in modes:
        for extra in [0, 20, 50, 100, 200]:
            res = {n: [] for n in names}
            for G, r in tests:
                Ga = attack_graph(G, extra, mode, np.random.default_rng(r), seed_knowledge=r) if extra else G
                for k, v in score_all(Ga, r).items(): res[k].append(v)
            print(f"{mode:>8s} {extra:>6d} " + " ".join(f"{np.mean(res[n]):>10.3f}±{np.std(res[n]):.2f}" for n in names))
main()
