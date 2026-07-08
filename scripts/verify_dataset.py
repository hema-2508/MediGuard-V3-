"""
verify_dataset.py

Verify DrugBankDDI / BioSNAPDDI datasets before training.

Checks:
1. Dataset shape
2. Label distribution
3. Missing values
4. Duplicate rows
5. Duplicate drug pairs
6. Pair leakage (train vs test)
7. Unique drugs
8. Shared drugs
9. Valid SMILES
10. Basic molecular statistics

Usage:
python verify_dataset.py --dataset data/drugbankddi

or

python verify_dataset.py --dataset data/biosnapddi
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------
# Read Dataset
# ---------------------------------------------------

def load_csv(path):
    df = pd.read_csv(
        path,
        low_memory=False
    )

    # Ensure correct column names
    df.columns = ["label", "smile1", "smile2"]

    # Convert label to integer
    df["label"] = df["label"].astype(int)

    return df


# ---------------------------------------------------
# Canonical Pair
# ---------------------------------------------------

def canonical_pair(a, b):
    return tuple(sorted((a, b)))


# ---------------------------------------------------
# Main
# ---------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to dataset folder (drugbankddi or biosnapddi)"
)

args = parser.parse_args()

dataset_dir = Path(args.dataset)

train_path = dataset_dir / "train_clean.csv"
test_path = dataset_dir / "test_clean.csv"

train = load_csv(train_path)
test = load_csv(test_path)

print("=" * 70)
print("DATASET")
print("=" * 70)
print(dataset_dir.name)

# ---------------------------------------------------
# Shapes
# ---------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(f"Train : {len(train):,}")
print(f"Test  : {len(test):,}")

# ---------------------------------------------------
# Labels
# ---------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

for name, df in [("Train", train), ("Test", test)]:

    print(f"\n{name}")

    counts = df.label.value_counts().sort_index()

    total = len(df)

    for label, count in counts.items():

        print(
            f"Label {label} : "
            f"{count:,} "
            f"({100*count/total:.2f}%)"
        )

# ---------------------------------------------------
# Missing values
# ---------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print("\nTrain")
print(train.isnull().sum())

print("\nTest")
print(test.isnull().sum())

# ---------------------------------------------------
# Duplicate rows
# ---------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print("Train :", train.duplicated().sum())
print("Test  :", test.duplicated().sum())

# ---------------------------------------------------
# Duplicate drug pairs
# ---------------------------------------------------

train_pairs = train.apply(
    lambda x: canonical_pair(x.smile1, x.smile2),
    axis=1
)

test_pairs = test.apply(
    lambda x: canonical_pair(x.smile1, x.smile2),
    axis=1
)

print("\n" + "=" * 70)
print("DUPLICATE PAIRS")
print("=" * 70)

print("Train :", train_pairs.duplicated().sum())
print("Test  :", test_pairs.duplicated().sum())

train_pairs = train.copy()
test_pairs = test.copy()

train_pairs["pair"] = train_pairs.apply(
    lambda x: tuple(sorted([x["smile1"], x["smile2"]])),
    axis=1
)

test_pairs["pair"] = test_pairs.apply(
    lambda x: tuple(sorted([x["smile1"], x["smile2"]])),
    axis=1
)

overlap = train_pairs.merge(
    test_pairs,
    on="pair",
    suffixes=("_train", "_test")
)

print(overlap[[
    "pair",
    "label_train",
    "label_test"
]].head(20))

# ---------------------------------------------------
# Leakage
# ---------------------------------------------------

pair_overlap = set(train_pairs).intersection(set(test_pairs))

print("\n" + "=" * 70)
print("TRAIN / TEST PAIR LEAKAGE")
print("=" * 70)

print("Shared Pairs :", len(pair_overlap))

# ---------------------------------------------------
# Unique Drugs
# ---------------------------------------------------

train_drugs = set(train.smile1) | set(train.smile2)
test_drugs = set(test.smile1) | set(test.smile2)

all_drugs = train_drugs | test_drugs

print("\n" + "=" * 70)
print("UNIQUE DRUGS")
print("=" * 70)

print("Train :", len(train_drugs))
print("Test  :", len(test_drugs))
print("Total :", len(all_drugs))

# ---------------------------------------------------
# Drug overlap
# ---------------------------------------------------

drug_overlap = train_drugs.intersection(test_drugs)

print("\n" + "=" * 70)
print("TRAIN / TEST DRUG OVERLAP")
print("=" * 70)

print("Shared Drugs :", len(drug_overlap))

print(
    "Overlap Percentage : "
    f"{100*len(drug_overlap)/len(test_drugs):.2f}%"
)

overlap = list(pair_overlap)

print(f"Total overlap: {len(overlap)}")

for pair in overlap[:20]:
    print(pair)

# ---------------------------------------------------
# SMILES Validation
# ---------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING SMILES")
print("=" * 70)

invalid = []

mol_weights = []

for smi in all_drugs:

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        invalid.append(smi)
    else:
        mol_weights.append(
            Descriptors.MolWt(mol)
        )

print("Total Unique SMILES :", len(all_drugs))
print("Valid SMILES        :", len(all_drugs) - len(invalid))
print("Invalid SMILES      :", len(invalid))

if len(invalid):

    print("\nFirst Invalid SMILES")

    for s in invalid[:10]:
        print(s)

# ---------------------------------------------------
# Molecular Statistics
# ---------------------------------------------------

print("\n" + "=" * 70)
print("MOLECULAR WEIGHT STATISTICS")
print("=" * 70)

print(f"Minimum : {min(mol_weights):.2f}")
print(f"Maximum : {max(mol_weights):.2f}")
print(f"Mean    : {sum(mol_weights)/len(mol_weights):.2f}")

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Train Samples          : {len(train):,}")
print(f"Test Samples           : {len(test):,}")
print(f"Unique Drugs           : {len(all_drugs):,}")
print(f"Duplicate Rows         : {train.duplicated().sum()+test.duplicated().sum()}")
print(f"Duplicate Drug Pairs   : {train_pairs.duplicated().sum()+test_pairs.duplicated().sum()}")
print(f"Pair Leakage           : {len(pair_overlap)}")
print(f"Invalid SMILES         : {len(invalid)}")

if (
    len(pair_overlap) == 0
    and len(invalid) == 0
):
    print("\n✅ Dataset verification PASSED")
else:
    print("\n⚠ Dataset requires inspection")