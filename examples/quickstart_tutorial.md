# 🚀 PyNeuraLogic Quickstart Tutorial

Welcome to PyNeuraLogic — a **differentiable logic programming** library. You declare logical rules,
and the backend compiles them into trainable neural computation graphs. This tutorial walks you through
the complete workflow: from defining a logic dataset to training a model and visualizing the results.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Core Concepts (60-second primer)](#2-core-concepts-60-second-primer)
3. [Step 1 — Define the Model (Template)](#3-step-1--define-the-model-template)
4. [Step 2 — Create a Logic Dataset](#4-step-2--create-a-logic-dataset)
5. [Step 3 — Visualize the Template & Samples](#5-step-3--visualize-the-template--samples)
6. [Step 4 — Discover Derivable Queries](#6-step-4--discover-derivable-queries)
7. [Step 5 — Build & Train](#7-step-5--build--train)
8. [Step 6 — Evaluate & Inspect Results](#8-step-6--evaluate--inspect-results)

---

## 1. Installation

```bash
pip install neuralogic
```

**Requirements:** Java ≥ 1.8 must be installed. The first call to any core function auto-starts the JVM.

For visualizations, install [Graphviz](https://graphviz.org/download/) and ensure the `dot` executable is on your `PATH`.

---

## 2. Core Concepts (60-second primer)

| Concept | What it is | Example |
|---|---|---|
| **Relation** (`R`) | A typed predicate — the atom of the language | `R.edge(0, 1)` — "there is an edge from 0 to 1" |
| **Variable** (`V`) | A placeholder that unifies across a rule | `V.X`, `V.Y` |
| **Constant** (`C`) | A named entity | `C.circle`, `C.rectangle` |
| **Rule** | An implication `head <= body` | `R.h(V.X) <= R.f(V.X)` |
| **Weight** (`[...]`) | Learnable parameters attached to a relation | `R.h(V.X)[5, 10]` — a [5×10] matrix |
| **Model** | A collection of rules (the template) | `model = Model()` |
| **Dataset** | A set of (query, example fact) pairs | `dataset.add(query, [facts...])` |
| **Module** | A pre-built rule-expander (GCN, LSTM, etc.) | `GCNConv(1, 5, "h1", "h0", "edge")` |

Rules combine via `<=` (implication), bodies chain with `&`, metadata attaches with `|`:

```python
(R.h(V.X)[5, 10] <= (R.f(V.Y)[10, 20], R.edge(V.Y, V.X))) | [Transformation.RELU, Aggregation.AVG]
```

Weights follow **`[output_dim, input_dim]`** convention.

### 2.1 Metadata: Controlling how rules behave

Every rule and predicate can carry **metadata** — settings that control how the computation flows.
Metadata is always attached with `|`, whether on a rule or on a predicate.

**Three key dimensions:**

| Field | What it controls | Common values |
|---|---|---|
| **Transformation** | Activation function applied to the rule's output | `RELU`, `TANH`, `SIGMOID`, `SOFTMAX`, `IDENTITY` |
| **Aggregation** | How multiple groundings of the same rule are combined | `SUM` (default), `AVG`, `MAX`, `COUNT` |
| **Combination** | How body literals combine before the head | `SUM` (default), `PRODUCT`, `ELPRODUCT`, `CONCAT` |

**On a rule** — metadata is appended with `|`:

```python
# This rule's output gets ReLU, and multiple groundings are averaged
(R.h(V.X) <= (R.f(V.Y), R.edge(V.Y, V.X))) | [Transformation.RELU, Aggregation.AVG]
```

**On a predicate** — use `/` to pin the arity, then `|` to attach metadata:

```python
# Apply tanh to predicate h only when it has arity 2 (e.g., R.h(1, 2) but not R.h(1))
R.h / 2 | [F.tanh]
```

The `/` operator creates a `Predicate` of a specific arity from a bare relation name; `|` then
binds the metadata to that predicate, exactly as it does on rules. This is what modules like
`GCNConv` use internally — their second expansion rule sets per-predicate metadata. In custom
templates you can mix rule-level and predicate-level metadata freely.

---

## 3. Step 1 — Define the Model (Template)

We'll build a 2-layer graph convolutional network (GCN) on a small 3-node graph. The template
consists of two `GCNConv` modules, each expanding into logical rules under the hood.

```python
from neuralogic.core import Model, Settings, Transformation
from neuralogic.nn.module import GCNConv
from neuralogic.nn.optim import Adam

# Create an empty model (template)
model = Model()

# Layer 1: 1 input feature → 4 hidden dims
model += GCNConv(
    in_channels=1,
    out_channels=4,
    output_name="h1",        # name of the output predicate
    feature_name="node_feature",
    edge_name="edge",
    activation=Transformation.RELU,
)

# Layer 2: 4 hidden dims → 2 output dims (binary classification)
model += GCNConv(
    in_channels=4,
    out_channels=2,
    output_name="predict",   # this is our final prediction predicate
    feature_name="h1",
    edge_name="edge",
)

# See what the template looks like as logic rules
print(model)
```

**Output (simplified):**

```
{4, 1} h1(X) :- {4, 1} node_feature(Y), {1} edge(Y, X).
{4} h1 :- {4}... .
{2, 4} predict(X) :- {2, 4} h1(Y), {1} edge(Y, X).
{2} predict :- {2}... .
```

Each `GCNConv` expands into two kinds of rules:
- **A parameterized rule** with learnable weight matrices (message passing)
- **A metadata rule** defining activation and aggregation per predicate

### Aside: The `F` shorthand for transformations in rule bodies

Beyond the metadata `| [Transformation.RELU]` syntax, PyNeuraLogic provides a compact `F`
shorthand that wraps any function directly around a relation inside a rule body:

```python
from neuralogic.core import F

# Apply tanh to R.b before combining with R.a in the rule head
model += (R.h <= F.tanh(R.b) & R.a)
```

`F` covers **all** transformation and aggregation functions — `F.relu`, `F.tanh`, `F.sigmoid`,
`F.softmax`, `F.avg`, `F.sum`, `F.max`, and more. This is especially useful for injecting
non-linearities at specific points in a rule body without adding separate metadata rules.

Functions can also be **composed and nested**:

```python
# Apply relu first, then tanh
model += (R.h <= F.tanh(F.relu(R.b)) & R.a)
```

---

## 4. Step 2 — Create a Logic Dataset

PyNeuraLogic's logic dataset consists of **samples** — each pairs a **query** (what to predict)
with a list of **example facts** (the input). Let's create a small 3-node graph:

```
  (0) --- (1)
   |       |
   +---(2)-+
```

Node 0 has feature `[0.5]`, node 1 has `[1.0]`, node 2 has `[-0.5]`.
We want to predict binary labels: node 0 → class 1, node 1 → class 0, node 2 → class 1.

> **Note:** Facts added to a dataset are always **implicitly fixed** (non-learnable) — there is no
> need to call `.fixed()` on them. That method is only relevant when writing template rules, not
> when populating a `Dataset`.

```python
from neuralogic.core import R  # R is the Relation factory
from neuralogic.dataset import Dataset, Sample

# Create dataset from individual Samples
dataset = Dataset()

# One example graph shared by all samples
example = [
    R.node_feature(0)[0.5],
    R.node_feature(1)[1.0],
    R.node_feature(2)[-0.5],
    R.edge(0, 1)[1],
    R.edge(1, 0)[1],
    R.edge(1, 2)[1],
    R.edge(2, 1)[1],
    R.edge(2, 0)[1],
    R.edge(0, 2)[1],
]

# Each sample: (query, example_facts)
dataset.add_samples([
    Sample(R.predict(0)[[1, 0]], example),  # node 0 → one-hot class 0
    Sample(R.predict(1)[[0, 1]], example),  # node 1 → one-hot class 1
    Sample(R.predict(2)[[1, 0]], example),  # node 2 → one-hot class 0
])

print(f"Dataset has {len(dataset)} samples")
```

**Key points:**
- `R.node_feature(0)[0.5]` — node 0 has feature value 0.5 (fixed automatically)
- `R.edge(0, 1)[1]` — an edge from node 0 to node 1 with weight 1 (fixed automatically)
- `R.predict(0)[[1, 0]]` — the query says "predict class 0 for node 0" (one-hot encoded)
- Predicate names (`node_feature`, `edge`, `predict`) must match the feature/edge/output names in your model

### Alternative: `Dataset.add()` shorthand

You can also use the compact `add()` method:

```python
dataset2 = Dataset()
dataset2.add(R.predict(0)[[1, 0]], example)
dataset2.add(R.predict(1)[[0, 1]], example)
dataset2.add(R.predict(2)[[1, 0]], example)
```

---

## 5. Step 3 — Visualize the Template & Samples

PyNeuraLogic can render both the template (model architecture) and the grounded/neuralized samples
(computation graphs). This requires **Graphviz** installed on your system.

All drawing functions accept a `show` parameter (default `True`) — when `True`, the image is
displayed inline in Jupyter or popped up via matplotlib in a script. Set `show=False` and pass a
`filename` to save to disk instead.

### 5.1 Template Visualization

The template must be built before it can be drawn:

```python
# Build the model first (visualization requires a built model)
model.build(
    Settings(optimizer=Adam(lr=0.01))
)

# Draw the template — shows the rule structure with weight matrices
model.draw(
    show=True,          # display inline (Jupyter) or pop-up window (script)
    img_type="png",
    value_detail=0,     # 0=compact, 1=detailed, 2=super-detailed values
)
```

To save to a file instead:

```python
model.draw(filename="template.png", show=False)
```

### 5.2 Grounding Visualization

A **grounded** sample shows how the logical rules instantiate against concrete input facts
*before* neuralization — this is the symbolic structure before weight initialization.
You need to ground the dataset explicitly (not via `build_dataset`, which combines both steps):

```python
from neuralogic.utils.visualize import draw_grounding

# Step 1: Ground the dataset (logical instantiation only, no weights)
grounded_dataset = model.ground(dataset)

# Step 2: Draw the grounding of the first sample
draw_grounding(grounded_dataset[0], show=True, value_detail=0)

# Or use the Grounding object's own draw method:
grounded_dataset[0].draw(show=True)
```

This reveals the symbolic computation graph — which atoms derive from which, what the
predicate dependencies are, and how variable substitutions propagate.

### 5.3 Neuralized Sample Visualization

A **neuralized** sample is the result after weight initialization — it shows the actual
computation graph with tensor dimensions and learnable parameters. You can neuralize the
grounded dataset and draw individual samples:

```python
from neuralogic.utils.visualize import draw_sample

# Neuralize the grounded dataset
built_dataset = grounded_dataset.neuralize()

# Draw the first neuralized sample
draw_sample(built_dataset[0], show=True, value_detail=0)

# Or use the NeuralSample's own draw method:
built_dataset[0].draw(show=True)
```

### 5.4 DOT Source Export

For further customization in external Graphviz tools:

```python
from neuralogic.utils.visualize import model_to_dot_source, sample_to_dot_source

print(model_to_dot_source(model))
print(sample_to_dot_source(built_dataset[0]))
```

---

## 6. Step 4 — Discover Derivable Queries

**Before** creating your dataset, you can ask the model what queries are even possible
for a given input. This is especially powerful when working with complex relational
structures where the space of possible labels isn't obvious.

### 6.1 `derivable_queries()` — What can I ask?

Given an example (input facts), the model tells you which output relations can be
derived, with what variable bindings:

```python
# Discover what the model can derive from our example graph
queries = model.derivable_queries(example)

print("Derivable queries:")
for q in queries:
    print(f"  {q}")
```

**Example output:**

```
Derivable queries:
  predict(0)
  predict(1)
  predict(2)
  h1(0)
  h1(1)
  h1(2)
```

This tells you:
- The `predict` predicate can be queried for nodes 0, 1, and 2 (good — those are the nodes we want to classify)
- The hidden `h1` predicate is also derivable (it's the intermediate layer)

### 6.2 `query()` / `q()` — What result do I get?

You can also perform a concrete query to see the **substitutions** (variable bindings)
that the template can produce:

```python
# Query: what values of X make predict(X) derivable?
results = model.query(R.predict(V.X), example)
print("Query results (substitutions):", results)
```

**Example output:**

```
Query results (substitutions): [{'X': 0}, {'X': 1}, {'X': 2}]
```

### 6.3 Using this to design your dataset

This workflow lets you **explore first, then create**:

1. Define the model template (rules)
2. Write a single example (input facts)
3. Call `derivable_queries()` to see what's queryable
4. Based on the derivable queries, create labeled samples via `Dataset.add()`

This inversion of the typical ML workflow is one of PyNeuraLogic's key strengths —
the template itself defines the space of possible learning targets.

---

## 7. Step 5 — Build & Train

Now we compile the model and train it. Note that `epochs` is passed to `model.train()`,
not to `Settings`.

```python
from neuralogic.core import Settings
from neuralogic.nn.optim import Adam
from neuralogic.nn.loss import MSE

# Configure training
settings = Settings(
    optimizer=Adam(lr=0.01),
    error_function=MSE(),
)

# Build compiles the model through the Java backend
model.build(settings)

# Build the dataset (grounds facts + neuralizes weights in one step)
built_dataset = model.build_dataset(dataset)

# Train! epochs is passed here, not in Settings
results = model.train(built_dataset, epochs=300)

# results is a list of (target, output, error) tuples
for target, output, error in results[-1:]:  # last epoch
    print(f"Final epoch — Target: {target}, Output: {output}, Error: {error}")
```

**What happens under the hood:**
1. **Grounding** — the template rules are instantiated against the input facts, creating concrete computation nodes
2. **Neuralization** — learnable weights are assigned and initialized
3. **Training loop** — forward + backward passes update weights via the optimizer

### 7.1 Save & Load Weights

```python
# Save trained weights
model.save("my_model.pkl")

# Later: rebuild the same architecture and load
model2 = Model()
model2 += GCNConv(1, 4, "h1", "node_feature", "edge", activation=Transformation.RELU)
model2 += GCNConv(4, 2, "predict", "h1", "edge")
model2.build(settings)
model2.load("my_model.pkl")
```

---

## 8. Step 6 — Evaluate & Inspect Results

### 8.1 Test Mode (Evaluation without side effects)

```python
# Test returns raw outputs — no gradient computation
outputs = model.test(built_dataset)
for i, out in enumerate(outputs):
    print(f"Sample {i}: {out}")
```

### 8.2 Call Mode (Forward pass, invalidates cache first)

```python
# Shortcut: model(built_dataset) calls forward
predictions = model(built_dataset)
```

### 8.3 Manual Inspection

```python
import numpy as np

for i, (sample, pred) in enumerate(zip(dataset.samples, predictions)):
    target_label = np.argmax(sample.query.weight)   # one-hot → class index
    pred_label = np.argmax(pred)
    print(f"Node {i}: target={target_label}, predicted={pred_label}, "
          f"values={[round(v, 4) for v in pred]}")
```

### 8.4 State Dict Inspection

```python
state = model.state_dict()
print(f"Number of learnable weights: {len(state['weights'])}")
for idx, name in state['weight_names'].items():
    print(f"  Weight {idx} ('{name}'): shape={state['weights'][idx].shape}")
```


