import networkx as nx
import numpy as np


def _connected_ws(n, k=6, p=0.08, seed=None):
    """Connected Watts-Strogatz graph."""
    g = nx.watts_strogatz_graph(n, k, p, seed=seed)

    if nx.is_connected(g):
        return g

    return g.subgraph(
        max(nx.connected_components(g), key=len)
    ).copy()


def _make_region(n, family="ba", seed=None):
    """Generate one region from a topology family."""
    rng = np.random.default_rng(seed)

    if family == "ba":
        m = min(3, max(1, n - 1))
        return nx.barabasi_albert_graph(n, m, seed=seed)

    if family == "ws":
        k = min(6, n - 1)

        if k % 2:
            k -= 1

        k = max(2, k)

        return _connected_ws(
            n,
            k=k,
            p=0.08,
            seed=seed
        )

    if family == "er":
        p = float(rng.uniform(0.05, 0.15))

        return nx.erdos_renyi_graph(
            n,
            p,
            seed=seed
        )

    if family == "sbm":
        sizes = [n // 2, n - n // 2]

        p_in = float(rng.uniform(0.08, 0.16))
        p_out = float(rng.uniform(0.01, 0.04))

        probs = [
            [p_in, p_out],
            [p_out, p_in],
        ]

        return nx.stochastic_block_model(
            sizes,
            probs,
            seed=seed
        )

    raise ValueError(
        f"Unknown topology family: {family}"
    )


def _largest_connected_component(g):
    """Return largest connected component with integer IDs."""
    if len(g) == 0:
        return g.copy()

    if nx.is_connected(g):
        return nx.convert_node_labels_to_integers(g)

    component = max(
        nx.connected_components(g),
        key=len
    )

    return nx.convert_node_labels_to_integers(
        g.subgraph(component).copy()
    )


def make_labeled_graph(
    n_honest=250,
    honest_family="ba",
    sybil_family="er",
    n_farms=3,
    farm_size_range=(15, 40),
    choke_points_range=(2, 9),
    camouflage_fraction=0.15,
    infiltration_fraction=0.20,
    degree_matched=False,
    seed=None,
):
    """
    Generate a synthetic labeled graph.

    Node label:
        0 = honest
        1 = sybil
    """

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # Honest region
    # -------------------------------------------------

    honest = _make_region(
        n_honest,
        family=honest_family,
        seed=int(
            rng.integers(
                0,
                2**32 - 1
            )
        ),
    )

    honest = _largest_connected_component(
        honest
    )

    G = nx.Graph()

    G.add_nodes_from(
        honest.nodes()
    )

    G.add_edges_from(
        honest.edges()
    )

    nx.set_node_attributes(
        G,
        0,
        "label"
    )

    next_id = n_honest

    honest_nodes = list(
        G.nodes()
    )

    sybil_nodes = []

    # -------------------------------------------------
    # Sybil farms
    # -------------------------------------------------

    for _ in range(n_farms):

        farm_size = int(
            rng.integers(
                farm_size_range[0],
                farm_size_range[1] + 1
            )
        )

        farm = _make_region(
            farm_size,
            family=sybil_family,
            seed=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            ),
        )

        farm = _largest_connected_component(
            farm
        )

        mapping = {
            old: next_id + i
            for i, old in enumerate(
                farm.nodes()
            )
        }

        farm = nx.relabel_nodes(
            farm,
            mapping
        )

        G.add_nodes_from(
            farm.nodes()
        )

        G.add_edges_from(
            farm.edges()
        )

        nx.set_node_attributes(
            G,
            {
                node: 1
                for node in farm.nodes()
            },
            "label"
        )

        farm_nodes = list(
            farm.nodes()
        )

        sybil_nodes.extend(
            farm_nodes
        )

        # Connect the farm to the honest
        # region through choke points.

        n_chokes = int(
            rng.integers(
                choke_points_range[0],
                choke_points_range[1] + 1
            )
        )

        n_chokes = min(
            n_chokes,
            len(farm_nodes),
            len(honest_nodes),
        )

        for _ in range(n_chokes):

            s = int(
                rng.choice(
                    farm_nodes
                )
            )

            h = int(
                rng.choice(
                    honest_nodes
                )
            )

            G.add_edge(
                s,
                h
            )

        next_id += farm_size

    # -------------------------------------------------
    # Camouflage edges
    # -------------------------------------------------

    n_camouflage = int(
        camouflage_fraction
        * len(sybil_nodes)
    )

    for _ in range(
        n_camouflage
    ):

        if not sybil_nodes or not honest_nodes:
            break

        s = int(
            rng.choice(
                sybil_nodes
            )
        )

        h = int(
            rng.choice(
                honest_nodes
            )
        )

        G.add_edge(
            s,
            h
        )

    # -------------------------------------------------
    # Infiltration edges
    # -------------------------------------------------

    n_infiltration = int(
        infiltration_fraction
        * len(honest_nodes)
    )

    for _ in range(
        n_infiltration
    ):

        if not sybil_nodes or not honest_nodes:
            break

        h = int(
            rng.choice(
                honest_nodes
            )
        )

        s = int(
            rng.choice(
                sybil_nodes
            )
        )

        G.add_edge(
            h,
            s
        )

    # -------------------------------------------------
    # Optional degree matching
    # -------------------------------------------------

    if (
        degree_matched
        and sybil_nodes
        and honest_nodes
    ):

        honest_degrees = np.array(
            [
                G.degree(n)
                for n in honest_nodes
            ],
            dtype=float
        )

        target_degree = float(
            np.median(
                honest_degrees
            )
        )

        for s in sybil_nodes:

            current = G.degree(s)

            if current >= target_degree:
                continue

            candidates = [
                h
                for h in honest_nodes
                if not G.has_edge(s, h)
            ]

            if not candidates:
                continue

            needed = int(
                min(
                    target_degree - current,
                    len(candidates)
                )
            )

            if needed > 0:

                chosen = rng.choice(
                    candidates,
                    size=needed,
                    replace=False
                )

                for h in chosen:
                    G.add_edge(
                        s,
                        int(h)
                    )

    # -------------------------------------------------
    # Final cleanup
    # -------------------------------------------------

    G = _largest_connected_component(
        G
    )

    for node in G.nodes():

        if "label" not in G.nodes[node]:
            G.nodes[node]["label"] = 0

    return G


def make_camouflaged_graph(
    n_honest=300,
    n_farms=4,
    farm_size_range=(20, 50),
    choke_points_range=(5, 10),
    n_camouflage_edges=40,
    infiltration_fraction=0.25,
    seed=None,
):
    """
    Explicit hard benchmark.

    Honest nodes use BA topology and
    Sybil farms use ER topology.
    """

    G = make_labeled_graph(
        n_honest=n_honest,
        honest_family="ba",
        sybil_family="er",
        n_farms=n_farms,
        farm_size_range=farm_size_range,
        choke_points_range=choke_points_range,
        camouflage_fraction=0.0,
        infiltration_fraction=infiltration_fraction,
        degree_matched=False,
        seed=seed,
    )

    rng = np.random.default_rng(
        seed
    )

    honest_nodes = [
        n
        for n in G.nodes()
        if G.nodes[n]["label"] == 0
    ]

    sybil_nodes = [
        n
        for n in G.nodes()
        if G.nodes[n]["label"] == 1
    ]

    for _ in range(
        n_camouflage_edges
    ):

        if not honest_nodes or not sybil_nodes:
            break

        h = int(
            rng.choice(
                honest_nodes
            )
        )

        s = int(
            rng.choice(
                sybil_nodes
            )
        )

        if not G.has_edge(h, s):
            G.add_edge(
                h,
                s
            )

    return G


def make_dataset(
    n_graphs=16,
    seed=0,
    difficulty="mixed"
):
    """
    Generate a dataset of synthetic graphs.

    difficulty:
        easy
        mixed
        hard

    Returns:
        list of (graph, is_hard)
    """

    rng = np.random.default_rng(
        seed
    )

    dataset = []

    for _ in range(
        n_graphs
    ):

        if difficulty == "easy":

            choke_range = (1, 3)

            camouflage = float(
                rng.uniform(
                    0.00,
                    0.08
                )
            )

            infiltration = float(
                rng.uniform(
                    0.00,
                    0.10
                )
            )

            degree_matched = False

        elif difficulty == "hard":

            choke_range = (5, 12)

            camouflage = float(
                rng.uniform(
                    0.15,
                    0.35
                )
            )

            infiltration = float(
                rng.uniform(
                    0.20,
                    0.45
                )
            )

            degree_matched = True

        elif difficulty == "mixed":

            choke_range = (2, 9)

            camouflage = float(
                rng.uniform(
                    0.05,
                    0.25
                )
            )

            infiltration = float(
                rng.uniform(
                    0.08,
                    0.35
                )
            )

            degree_matched = bool(
                rng.integers(
                    0,
                    2
                )
            )

        else:

            raise ValueError(
                "difficulty must be "
                "'easy', 'mixed', or 'hard'"
            )

        families = [
            "ba",
            "ws",
            "er",
            "sbm"
        ]

        honest_family = str(
            rng.choice(
                families
            )
        )

        sybil_family = str(
            rng.choice(
                families
            )
        )

        G = make_labeled_graph(
            n_honest=int(
                rng.integers(
                    220,
                    320
                )
            ),

            honest_family=honest_family,

            sybil_family=sybil_family,

            n_farms=int(
                rng.integers(
                    2,
                    5
                )
            ),

            farm_size_range=(
                15,
                50
            ),

            choke_points_range=choke_range,

            camouflage_fraction=camouflage,

            infiltration_fraction=infiltration,

            degree_matched=degree_matched,

            seed=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            ),
        )

        dataset.append(
            (
                G,
                difficulty == "hard"
            )
        )

    return dataset


def make_cross_distribution_test(
    n_graphs=8,
    seed=999
):
    """
    Generate held-out graphs using
    varied topology distributions.
    """

    rng = np.random.default_rng(
        seed
    )

    families = [
        "ba",
        "ws",
        "er",
        "sbm"
    ]

    graphs = []

    for _ in range(
        n_graphs
    ):

        honest_family = str(
            rng.choice(
                families
            )
        )

        sybil_family = str(
            rng.choice(
                families
            )
        )

        G = make_labeled_graph(
            n_honest=int(
                rng.integers(
                    250,
                    350
                )
            ),

            honest_family=honest_family,

            sybil_family=sybil_family,

            n_farms=int(
                rng.integers(
                    3,
                    6
                )
            ),

            farm_size_range=(
                20,
                60
            ),

            choke_points_range=(
                7,
                14
            ),

            camouflage_fraction=float(
                rng.uniform(
                    0.20,
                    0.40
                )
            ),

            infiltration_fraction=float(
                rng.uniform(
                    0.30,
                    0.55
                )
            ),

            degree_matched=True,

            seed=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            ),
        )

        graphs.append(
            G
        )

    return graphs


if __name__ == "__main__":

    print(
        "Testing graph generator..."
    )

    mixed = make_dataset(
        n_graphs=4,
        seed=42,
        difficulty="mixed"
    )

    hard = make_dataset(
        n_graphs=4,
        seed=123,
        difficulty="hard"
    )

    cross = make_cross_distribution_test(
        n_graphs=4,
        seed=999
    )

    for name, dataset in [
        (
            "mixed",
            [g for g, _ in mixed]
        ),
        (
            "hard",
            [g for g, _ in hard]
        ),
        (
            "cross",
            cross
        ),
    ]:

        print(
            f"\\n{name}:"
        )

        for i, G in enumerate(
            dataset
        ):

            labels = [
                G.nodes[n]["label"]
                for n in G.nodes()
            ]

            print(
                f"  graph {i}: "
                f"nodes={G.number_of_nodes()}, "
                f"edges={G.number_of_edges()}, "
                f"honest={labels.count(0)}, "
                f"sybil={labels.count(1)}"
            )