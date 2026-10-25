
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import hashlib

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "AiArtData": "AI-generated",
    "RealArt": "Real",
}

records = []

for class_folder, label in LABELS.items():
    class_path = DATA_DIR / class_folder

    for path in sorted(class_path.rglob("*")):
        if not path.is_file():
            continue

        record = {
            "path": str(path),
            "class": label,
            "file_size_bytes": path.stat().st_size,
            "width": None,
            "height": None,
            "color_mode": None,
            "format": None,
            "sha256": None,
            "corrupt": False,
        }

        try:
            sha256 = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256.update(chunk)
            record["sha256"] = sha256.hexdigest()

            with Image.open(path) as img:
                record["width"] = img.width
                record["height"] = img.height
                record["color_mode"] = img.mode
                record["format"] = img.format
                img.verify()

        except (OSError, ValueError, UnidentifiedImageError):
            record["corrupt"] = True

        records.append(record)

df = pd.DataFrame(records)

# Identify duplicate files using exact SHA-256 hashes
df["duplicate"] = (
    df["sha256"].duplicated(keep="first")
    & df["sha256"].notna()
)

df.to_csv(OUTPUT_DIR / "dataset_audit.csv", index=False)

print("\n========== DATASET AUDIT ==========")

print("\nFiles per class:")
print(df.groupby("class").size())

print("\nImage dimensions and file sizes:")
print(
    df.groupby("class")[
        ["width", "height", "file_size_bytes"]
    ].agg(["min", "median", "max"])
)

print("\nCorrupt files:", int(df["corrupt"].sum()))
print("Exact duplicate files:", int(df["duplicate"].sum()))

print("\nImage formats:")
print(df.groupby(["class", "format"]).size())

print("\nAudit saved to reports/dataset_audit.csv")
