import pandas as pd
from pathlib import Path

# Change this to drugbankddi or biosnapddi
DATASET = "biosnapddi"

dataset_path = Path(f"data/{DATASET}")

train = pd.read_csv(dataset_path / "train.csv")
test = pd.read_csv(dataset_path / "test.csv")

print("Original Train:", len(train))
print("Original Test :", len(test))

# ----------------------------
# Remove exact duplicate rows
# ----------------------------

train = train.drop_duplicates().reset_index(drop=True)
test = test.drop_duplicates().reset_index(drop=True)

print("\nAfter removing duplicate rows")
print("Train:", len(train))
print("Test :", len(test))

# ----------------------------
# Remove train-test leakage
# ----------------------------

def pair(row):
    return tuple(sorted([row["smile1"], row["smile2"]]))

train["pair"] = train.apply(pair, axis=1)
test["pair"] = test.apply(pair, axis=1)

train_pairs = set(train["pair"])

# Remove leaked pairs from test
test = test[~test["pair"].isin(train_pairs)]

print("\nAfter removing leaked pairs")
print("Test:", len(test))

# Remove helper column
train = train.drop(columns=["pair"])
test = test.drop(columns=["pair"])

# ----------------------------
# Save cleaned files
# ----------------------------

train.to_csv(dataset_path / "train_clean.csv", index=False)
test.to_csv(dataset_path / "test_clean.csv", index=False)

print("\nSaved:")
print(dataset_path / "train_clean.csv")
print(dataset_path / "test_clean.csv")