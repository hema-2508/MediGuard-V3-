# GraphSAGE Drug-Drug Interaction (DDI) Prediction

A production-ready pipeline that predicts whether two drugs interact,
using RDKit molecular features (Morgan fingerprints + physicochemical
descriptors) and a 2-layer GraphSAGE encoder over a drug-drug
interaction graph.

## Project layout

```
config.py             # all paths & hyperparameters
feature_generator.py  # RDKit descriptors + Morgan fingerprint, with disk caching
dataset.py            # CSV loading, drug vocabulary, feature matrix, train/val/test splits
graph_builder.py       # builds the PyG graph (node=drug, edge=known interaction)
model.py               # 2-layer GraphSAGE encoder + pair classifier head
trainer.py             # AdamW + BCEWithLogitsLoss + early stopping + checkpointing
evaluate.py             # Accuracy/Precision/Recall/F1/MCC/ROC-AUC/PR-AUC/Confusion Matrix
predict.py              # inference on new SMILES pairs (handles cold-start drugs)
utils.py                # seeding, logging, early stopping, checkpoint I/O
main.py                 # CLI entry point wiring everything together
```

## Install

```bash
pip install -r requirements.txt
```

## Data

Place the following files under `./data/`:

- `train_clean.csv` (DrugBankDDI, primary training set)
- `test_clean.csv` (DrugBankDDI, held-out test set)
- `biosnap_ddi.csv` (BioSNAPDDI, external validation set)

Each CSV must contain exactly these columns:

| column | meaning                              |
|--------|----------------------------------------|
| label  | 0 = No Interaction, 1 = Interaction     |
| smile1 | SMILES string of Drug A                 |
| smile2 | SMILES string of Drug B                 |

## Usage

Train on DrugBankDDI (features/vocab/graph are computed once and cached
under `./cache/`):

```bash
python main.py train --epochs 200
```

Evaluate the best checkpoint on the DrugBank test set:

```bash
python main.py evaluate
```

Run external validation on BioSNAPDDI:

```bash
python main.py external_validate
```

Score a single new pair (works even for drugs never seen in training —
they are added to the graph as isolated nodes and encoded via
GraphSAGE's self/root transformation):

```bash
python main.py predict --smiles1 "CC(=O)OC1=CC=CC=C1C(=O)O" --smiles2 "CC(=O)Nc1ccc(O)cc1"
```

Run the whole pipeline end-to-end:

```bash
python main.py all --epochs 200
```

## Design notes

- **Graph construction**: the message-passing `edge_index` is built
  only from `label == 1` pairs in the *training* split. This keeps the
  transductive setup leak-free — validation/test/external labels never
  influence the graph structure the GNN encodes over.
- **Cold-start drugs**: any drug seen only at test/external/predict
  time becomes an isolated node; `SAGEConv`'s root/self transformation
  still produces a usable embedding for it.
- **Pair representation**: `[A, B, |A-B|, A*B]` (4 × hidden_dim) is fed
  into an MLP head with BatchNorm + Dropout(0.3) to produce a single
  interaction logit.
- **Feature caching**: Morgan fingerprints + descriptors are cached to
  `cache/feature_cache.pkl` keyed by canonical SMILES, and the
  descriptor z-score scaler (fit once on the DrugBank training
  molecules) is cached to `cache/descriptor_scaler.pkl`, so repeated
  runs and repeated drugs across datasets never re-run RDKit chemistry.
- **Early stopping**: monitors validation ROC-AUC (`Config.EARLY_STOPPING_METRIC`)
  with patience 15; the single best checkpoint (by that metric) is
  kept at `checkpoints/best_model.pt`.