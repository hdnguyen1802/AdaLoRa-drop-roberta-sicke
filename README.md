# AdaLoRA‑Drop: Parameter‑Efficient RoBERTa Fine‑Tuning on SICK‑E

> **TL;DR**: We combine AdaLoRA’s adaptive rank allocation with LoRA‑Drop style layer sharing to fine‑tune RoBERTa‑base for textual entailment (SICK‑E) while training **<0.1%** of the model’s parameters, with only a small accuracy drop vs. full fine‑tuning.

---

## Motivation
Full model fine‑tuning of RoBERTa‑base (~125M params) is compute‑ and memory‑heavy—overkill for a small dataset like **SICK‑E (~9.8k pairs)**. Adapter‑style methods (LoRA/AdaLoRA) update a tiny set of weights while keeping the backbone frozen. Our approach, **AdaLoRA‑Drop**, pushes parameter efficiency further by **(1)** adapting rank per layer (AdaLoRA) and **(2)** **sharing** adapter modules across low‑importance layers (inspired by LoRA‑Drop’s ΔWx‑based importance).

---

## Key ideas
- **Two‑stage procedure**
  1) **Importance estimation** on a small data slice using ΔWx energy; sort layers and mark a minimal set that covers 90–95% cumulative importance as **high‑importance**.
  2) **Fine‑tuning with AdaLoRA**: give **individual** AdaLoRA modules to high‑importance layers; **share one module per shape group** across the remaining low‑importance layers.
- **Tunable budget** via the cumulative importance threshold (e.g., 95% vs. 90%).
- **Edge‑friendly**: tiny adapter sizes cut memory/latency and enable cheaper GPUs or on‑device scenarios.

---

## Results (SICK‑E test accuracy vs. trainable params)

| Model | Trainable Params | Test Acc. |
|---|---:|---:|
| Full fine‑tune (RoBERTa‑base) | 124,647,939 | 0.8863 |
| LoRA‑all | 1,256,451 | 0.8939 |
| LoRA‑Drop@95 | 1,062,915 | 0.8900 |
| AdaLoRA‑all | 682,272 | 0.8805 |
| **AdaLoRA‑Drop@95 (ours)** | **133,698** | **0.8744** |
| **AdaLoRA‑Drop@90 (ours)** | **66,846** | **0.8736** |

> **Takeaway**: **AdaLoRA‑Drop@90** achieves a **>99.9% reduction** in trainable parameters with only ~1–1.5% absolute accuracy drop vs. full fine‑tuning.

---

## Repository structure
```
.
adalora-drop-roberta-sicke/
├── ada-drop.ipynb
├── SICK_train.txt
├── SICK_test.txt
├── SICK_trial.txt
├── AdaLoRA-Drop-Parameter-Efficient-RoBERTa-Fine-Tuning-on-SICK-E.pptx
└── README.md

```
*Note:* Only the notebook and slides are included initially; `src/` and scripts are optional helpers if you want a scriptified version.

---

## Quickstart

### 1) Environment
- Python ≥ 3.10
- PyTorch (CUDA recommended)
- `transformers`, `datasets`, `accelerate`
- `peft` (for LoRA/AdaLoRA), `tqdm`

Create an environment and install:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install torch torchvision torchaudio  # choose the right CUDA build for your system
pip install transformers datasets accelerate peft tqdm
```

### 2) Data: SICK‑E (local files)
Place the three SICK files under `data/` exactly as follows:

- `SICK_train.txt`
- `SICK_test.txt`
- `SICK_trial.txt`

They are **tab-separated** with headers `pair_ID, sentence_A, sentence_B, relatedness_score, entailment_judgment`.

Example loader (pandas):
```python
import pandas as pd
train = pd.read_csv('SICK_train.txt', sep='	')
trial = pd.read_csv('SICK_trial.txt', sep='	')
test  = pd.read_csv('SICK_test.txt',  sep='	')
```

### 3) Run the notebook
Open **`ada-drop.ipynb`** and run all cells. The notebook covers:
- Importance estimation on a small subset (e.g., ~10% of train split)
- Layer ranking & thresholding (90% / 95%)
- AdaLoRA‑Drop fine‑tuning (target rank small, e.g., r=4)
- Evaluation on the SICK‑E test set

## Method details

### Stage 1 — Importance estimation
- Attach low‑rank adapters with small rank to targeted layers.
- Train briefly (e.g., 3 epochs on ~10% of SICK‑E) to gather **ΔWx** statistics.
- Compute per‑layer importance \(I_i \propto \mathbb{E}[\lVert \Delta W_i x \rVert^2]\) and sort.
- Choose layers that cover **90–95%** cumulative importance as high‑importance.

### Stage 2 — AdaLoRA‑Drop fine‑tuning
- Reset a fresh **RoBERTa‑base** classifier (`num_labels=3`).
- Apply **AdaLoRA** with a small target rank (e.g., `r=4`).
- **High‑importance** layers → independent AdaLoRA modules.
- **Low‑importance** layers → **share one module per shape group**.
- Train ~5 epochs with early stopping by validation accuracy.

---

## Reproducing the table
- Use the notebook’s experiment grid to toggle: **Full FT**, **LoRA‑all**, **LoRA‑Drop@95**, **AdaLoRA‑all**, **AdaLoRA‑Drop@95/90**.
- Track **trainable parameter counts** by summing adapter parameters only.
- Report **test accuracy** on SICK‑E.

---

## Notes & limitations
- Currently evaluated only on **SICK‑E** with **RoBERTa‑base**.
- ΔWx energy is a simple importance proxy; other signals (e.g., gradient‑based) might further improve selection.

---

## References
- **LoRA Without Regret**
- **AdaLoRA: Adaptive Budget Allocation for Parameter‑Efficient Fine‑Tuning**
- **LoRA‑Drop: Efficient LoRA Parameter Pruning based on Output Evaluation**

(See the slides for concise conceptual diagrams and comparisons.)

---
