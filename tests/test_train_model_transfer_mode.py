from sybil_shield.experiments.train_model import train_model


def test_train_model_accepts_transfer_mode():
    meta = train_model(
        model_type="gradient_adversarial",
        training_mode="transfer",
        num_graphs=2,
        min_nodes=80,
        max_nodes=120,
        sybil_fraction_min=0.10,
        sybil_fraction_max=0.35,
        epochs=1,
        learning_rate=0.01,
        attack_every=1,
        attack_budget=5,
        recompute_every=1,
        hidden_dims=(4, 4),
        seed_frac=0.05,
        seed=7,
    )

    assert meta["params"]["training_mode"] == "transfer"
    assert "auc" in meta["metrics"]
    assert "accuracy" in meta["metrics"]
