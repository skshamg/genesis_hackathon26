<<<<<<< HEAD
# genesis-hackathon26

# SYBIL — Topological Trust Engine

> **Privacy-preserving Sybil detection using graph topology, trust propagation, and adversarial learning.**

SYBIL is a topology-only trust engine designed to detect suspicious Sybil communities in social graphs without relying on identity information, profile data, text, biometrics, or other personal-content signals.

Instead of asking **"Who is this user?"**, SYBIL asks:

> **"How is this user connected to the rest of the graph?"**

---

## Overview

Sybil attackers create large numbers of fake identities and coordinate them to appear legitimate.

Although individual fake accounts may be difficult to distinguish from legitimate users, coordinated Sybil communities can leave structural patterns in a graph:

* Dense connections within the Sybil region
* Artificially constructed communities
* Unusual trust-flow patterns
* Narrow connections between Sybil and legitimate regions
* Attempts to connect Sybil nodes to highly trusted nodes
* Structural differences between organic and coordinated communities

SYBIL uses these graph-level signals to estimate the probability that a node belongs to a Sybil region.

The system is designed around four ideas:

**DETECT → ATTACK → DEFEND → EXPLAIN**

---

## Core Pipeline

```text
                  Graph
                    │
                    ▼
          Graph / Trust Analysis
                    │
                    ▼
          Structural Feature Extraction
                    │
                    ▼
             ┌─────────────┐
             │     GCN     │
             └─────────────┘
                    │
                    ▼
            Sybil Probability
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Explanation          Adversarial
                            Testing
                              │
                              ▼
                       Robust Training
```

The same prediction pipeline is exposed through a FastAPI backend.

---

# Why Topology?

Traditional Sybil and fraud detection systems may use information such as:

* Phone numbers
* Email addresses
* Device identifiers
* IP addresses
* Biometrics
* Profile information
* User-generated content

SYBIL deliberately avoids these signals.

The goal is to investigate whether **graph structure alone** contains enough information to identify coordinated artificial communities.

This makes the system particularly useful as a research framework for settings where identity or content-based analysis is undesirable or unavailable.

---

# Trust Propagation

SYBIL uses trusted seed nodes to propagate structural trust through the graph.

The system currently uses:

* Personalized PageRank
* Multiple PageRank settings
* SybilRank-style trust propagation
* Trust-to-degree relationships

Rather than treating raw trust scores as absolute values, trust signals can be rank-normalized within each graph.

This allows the detector to operate across graphs with different scales and degree distributions.

---

# Structural Features

The current feature pipeline includes:

* Node degree
* Clustering coefficient
* k-core information
* Personalized PageRank
* PPR / degree relationships
* SybilRank-style signals

These features are provided to the graph neural network as node-level structural representations.

The project is also investigating a learned-topology model in which the neural network learns structural representations directly from the graph instead of relying entirely on manually engineered features.

---

# Graph Neural Network

SYBIL uses a Graph Convolutional Network (GCN).

A simplified GCN layer is:

$$
H^{(l+1)}
=
\sigma(\hat{A}H^{(l)}W^{(l)}+b^{(l)})
$$

where:

* \(H^{(l)}\) is the node representation at layer \(l\)
* \(\hat A\) is the normalized adjacency matrix
* \(W^{(l)}\) is a learnable weight matrix
* \(b^{(l)}\) is the bias
* \(\sigma\) is the activation function

The current implementation uses a NumPy-based GCN with manually implemented forward propagation, backpropagation, and Adam optimization rather than relying on a dedicated GNN framework.

---

# Adversarial Attacks

A detector should not only be evaluated against random or fixed synthetic examples.

SYBIL therefore includes graph attackers designed to modify the graph in ways that make Sybil nodes appear more legitimate.

Current attack modes include:

### Random

Sybil nodes add connections to randomly selected legitimate nodes.

### Hubs

Sybil nodes preferentially connect toward structurally important nodes.

### Trusted

Sybil nodes target highly trusted nodes according to trust-propagation signals.

### Gradient-based attack

The GCN can expose gradients with respect to graph structure.

The attacker can use these gradients to identify graph modifications that have a large effect on the detector's loss.

This creates an adversarial feedback loop:

```text
Graph
  ↓
Detector
  ↓
Loss
  ↓
Graph gradients
  ↓
Attacker
  ↓
Modified graph
  ↓
Detector
```

---

# Robust Models

SYBIL currently evaluates multiple detector variants.

| Model                | Training strategy                                      |
| -------------------- | ------------------------------------------------------ |
| **Standard**         | Clean synthetic graphs                                 |
| **Random-robust**    | Random attack augmentation                             |
| **Trusted-robust**   | Trusted attack augmentation                            |
| **Mixed-robust**     | Combination of random, hub, and trusted attacks        |
| **Learned-topology** | Experimental model learning structural representations |

The goal is not simply to maximize performance against one attack.

The goal is to understand how different threat models affect **generalization and robustness**.

---

# An Important Adversarial Finding

Early experiments revealed an important weakness in the trusted attack model.

The initial trusted attacker used a static rule:

```text
PPR / degree
      ↓
top-decile nodes
      ↓
attack targets
```

The same basic targeting rule was used repeatedly across training graphs.

This allowed the trusted-robust detector to potentially learn the **specific fingerprint of the attacker** rather than learning a general boundary between legitimate and Sybil structures.

This resulted in substantially worse zero-shot transfer performance for the trusted model.

The next version of the attacker is therefore intended to use a more diverse and adaptive targeting strategy.

---

# Facebook Transfer Experiment

SYBIL was evaluated on a Facebook graph topology separate from the synthetic training graphs.

The experiment contained:

* **8,078 nodes**
* **176,568 unique edges**
* 100 benign training seeds
* Separate benign and Sybil evaluation populations

The Facebook graph provides a real social-network topology, while the Sybil region is synthetically constructed.

Therefore, this experiment should **not** be described as evaluation against naturally occurring Facebook Sybil ground truth.

Instead, it is a:

> **Zero-shot transfer evaluation on a real social-network topology with synthetic Sybil construction.**

### Results

| Model          |        AUC | Accuracy |
| -------------- | ---------: | -------: |
| Standard       | **0.8420** |   78.28% |
| Random-robust  | **0.8592** |   82.08% |
| Trusted-robust | **0.5598** |   44.85% |
| Mixed-robust   | **0.8572** |   73.88% |
| Ensemble       | **0.8531** |   75.63% |

The results show that the topology-based approach transfers reasonably well across graph distributions, with three models achieving approximately **0.84–0.86 AUC**.

The trusted model's substantially lower performance is being investigated as a threat-model design issue rather than being treated as evidence that trusted-node attacks are inherently undetectable.

---

# Why AUC?

Accuracy depends heavily on the chosen classification threshold and class distribution.

AUC is therefore used as a primary evaluation metric because it measures how well the detector separates Sybil and benign nodes across possible thresholds.

Current experimental targets are approximately:

* **0.85 AUC:** solid topology-transfer performance
* **0.90 AUC:** strong target
* **0.90+ AUC:** ambitious future target

Accuracy is reported alongside AUC but is not treated as the sole measure of detector quality.

---

# FastAPI Prediction API

SYBIL exposes its prediction pipeline through FastAPI.

### Endpoint

```http
POST /predict
```

Default development address:

```text
http://localhost:8000/predict
```

The endpoint accepts a graph representation and runs the four current detector variants through the shared prediction pipeline.

Conceptually:

```text
Graph Input
    │
    ▼
Feature Extraction
    │
    ▼
FastAPI /predict
    │
    ├── Standard model
    ├── Random-robust model
    ├── Trusted-robust model
    └── Mixed-robust model
    │
    ▼
Model Outputs
```

This separates the detection engine from the frontend and allows the same models to be accessed programmatically.

---

# Explainability

SYBIL is designed to provide structural evidence alongside predictions.

Rather than simply returning:

```text
Sybil probability: 0.91
```

the system aims to answer:

> **Why does this node look suspicious?**

Possible evidence includes:

* Unusual trust-flow behavior
* Neighborhood structure
* Local clustering
* Core structure
* Trust/degree relationships
* Community connectivity
* Structural changes introduced by adversarial attacks

These explanations are intended as **structural evidence**, not proof that an account is malicious.

---

# Architecture

```text
                    ┌───────────────────────┐
                    │       Frontend        │
                    │ Graph / Visualization │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       FastAPI         │
                    │       /predict        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Graph Feature Layer  │
                    │                       │
                    │ PPR / SybilRank       │
                    │ Degree / Clustering   │
                    │ k-core / structural   │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
         Standard          Robust Models     Learned Model
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Prediction / AUC    │
                    │   + Explainability    │
                    └───────────────────────┘
```

---

# Repository Structure

A typical project structure is:

```text
SYBIL/
│
├── gcn.py
├── detector.py
├── attacks.py
├── benchmark.py
├── eval_multi_model.py
├── real_facebook_eval.py
│
├── api/
│   └── ...
│
├── frontend/
│   └── ...
│
├── data/
│   └── ...
│
└── README.md
```

Core components:

| Component               | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| `gcn.py`                | GCN implementation and gradient computation         |
| `detector.py`           | Trust propagation and structural feature extraction |
| `attacks.py`            | Synthetic/adversarial graph attacks                 |
| `benchmark.py`          | Synthetic graph generation and benchmarking         |
| `eval_multi_model.py`   | Training/evaluation of robust detector variants     |
| `real_facebook_eval.py` | Zero-shot Facebook topology evaluation              |
| `api/`                  | FastAPI prediction interface                        |
| `frontend/`             | Visualization and interaction layer                 |

---

# Experimental Philosophy

The project is intentionally structured around controlled experiments.

Rather than asking only:

> "Does the detector work?"

SYBIL investigates:

1. Does topology contain useful Sybil signals?
2. Do trust-propagation methods improve detection?
3. Does a GCN outperform simpler trust-propagation baselines?
4. Does adversarial training improve robustness?
5. Does robustness transfer between attack strategies?
6. What happens under graph distribution shift?
7. Can the model learn useful structural representations automatically?
8. Can an adaptive attacker expose weaknesses in the detector?

---

# Current Research Direction

The next iterations focus on three areas.

### 1. Adaptive trusted attacker

Replace the static top-decile targeting rule with a more diverse/adaptive attack strategy.

### 2. Better structural features

Investigate richer neighborhood and relational features instead of relying only on individual node statistics.

### 3. Learned topology features

Introduce a fifth model that learns structural representations directly from the graph.

The central research question becomes:

> **Can a topology-only GNN learn representations that remain useful under adaptive structural attacks and graph distribution shift?**

---

# Limitations

SYBIL is a research prototype.

Important limitations include:

* Current Sybil training data is synthetically constructed.
* The Facebook transfer experiment does not use naturally occurring Facebook Sybil labels.
* Results are dependent on the graph generation and attack models.
* Static attackers can produce misleading robustness measurements.
* AUC and accuracy alone do not establish real-world fraud-detection effectiveness.
* Structural explanations indicate evidence, not malicious intent.
* The current learned-topology model is experimental.

These limitations are part of the research problem rather than being hidden from evaluation.

---

# Future Work

Potential future directions include:

* Adaptive gradient-based attackers
* More diverse trusted-node attack strategies
* Learned structural representations
* Graph attention mechanisms
* GraphSAGE / message-passing architectures
* Community-aware features
* Multi-hop trust-flow features
* Better adversarial training
* Larger real-world graph datasets
* Naturally labeled Sybil datasets
* Calibration and threshold analysis
* Interactive graph explanations
* Model comparison and ablation studies

---

# Project Philosophy

SYBIL is based on a simple idea:

> **You should not need to know who someone is to study how they interact with a network.**

By focusing on topology, trust propagation, and adversarial robustness, the project explores whether coordinated artificial identities can be detected from the structure they create.

**Detect the structure.
Attack the detector.
Defend against the attack.
Explain the evidence.**
=======
# SYBIL-SHIELD

Synthetic-to-real Sybil detection using trust propagation, structural graph features, GCNs, attack augmentation, and gradient-adversarial training.

## Project layout

```text
SYBIL-SHIELD/
├── src/sybil_shield/
│   ├── core/          # detector, graph features, GCN, attacks
│   ├── experiments/   # training/evaluation scripts
│   └── api/           # FastAPI inference service
├── scripts/           # the few commands you actually run
├── tests/             # API tests
├── data/              # put datasets here (not bundled)
├── results/           # generated outputs
└── docs/              # report/demo notes
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## What to run

### 1. Main demo / presentation
Start the API:

```bash
python scripts/run_api.py
```

Then use the frontend/demo client you are presenting to POST a graph to `/predict`. The API accepts nodes, edges, and optional labels. Labels should not be supplied for a zero-shot target-graph demo.

### 2. Main evaluation video
Run the synthetic-to-real Facebook transfer experiment:

```bash
python scripts/run_facebook_eval.py
```

This is the research/demo command to show the trained variants, validation diagnostic, Facebook transfer AUC/accuracy, and ensemble result. It expects the Facebook dataset paths configured in `src/sybil_shield/experiments/real_facebook_eval.py`.

### 3. Synthetic training / methodology demo
```bash
python scripts/run_training.py
```

### 4. Visuals
```bash
python scripts/run_visualization.py
```

## Suggested 48-hour hackathon demo flow

1. **Live graph input:** show a small graph and the SYBIL-SHIELD prediction output.
2. **Explain the idea:** only a small trusted benign seed set is needed on the target graph; no target Sybil labels are required for feature construction.
3. **Show synthetic attacker training:** standard, random-augmented, mixed-augmented, and gradient-adversarial variants.
4. **Show transfer:** run or display the Facebook transfer result and AUC/accuracy.
5. **Show structure:** briefly highlight trust propagation + structural/camouflage features + GCN.

## Important

Datasets are intentionally not included in the source archive. Put them under `data/` or update the dataset paths in the relevant experiment configuration. Generated caches, `__pycache__`, and old experimental clutter are excluded.
>>>>>>> f345065 (Initial project commit)
