
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

audit = pd.read_csv("reports/dataset_audit.csv")

# Keep only valid, unique images
df = audit[
    (audit["corrupt"] == False) &
    (audit["duplicate"] == False)
].copy()

# Binary labels: Real = 0, AI-generated = 1
df["label"] = df["class"].map({
    "Real": 0,
    "AI-generated": 1
})

# Stratified split, preserving class proportions
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=SEED,
    stratify=temp["label"]
)

output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

for name, split in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    split[["path", "label", "sha256"]].to_csv(
        output_dir / f"{name}.csv",
        index=False
    )

    print(f"\n{name.upper()} ({len(split)} images)")
    print(split["class"].value_counts())

# Check that no image content appears in multiple splits
train_hashes = set(train["sha256"])
val_hashes = set(val["sha256"])
test_hashes = set(test["sha256"])

assert train_hashes.isdisjoint(val_hashes)
assert train_hashes.isdisjoint(test_hashes)
assert val_hashes.isdisjoint(test_hashes)

assert len(train) + len(val) + len(test) == 974

print("\nSplits saved successfully.")
print("No duplicate leakage detected.")
