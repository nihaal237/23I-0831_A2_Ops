
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "app" / "instance" / "visionguard.db"
TARGET = ROOT / "instance" / "visionguard.db"

if not SOURCE.exists() or not TARGET.exists():
    raise FileNotFoundError(
        "Source or target database is missing."
    )

columns = [
    "filename",
    "prediction",
    "ai_probability",
    "threshold",
    "model_name",
    "created_at",
]

with sqlite3.connect(SOURCE) as source:
    source.row_factory = sqlite3.Row

    with sqlite3.connect(TARGET) as target:
        target.execute("BEGIN")

        try:
            existing = {
                tuple(row)
                for row in target.execute(
                    "SELECT " + ", ".join(columns) +
                    " FROM predictions"
                ).fetchall()
            }

            copied = 0

            rows = source.execute(
                "SELECT " + ", ".join(columns) +
                " FROM predictions ORDER BY id"
            ).fetchall()

            for row in rows:
                values = tuple(row[column] for column in columns)

                if values in existing:
                    continue

                placeholders = ", ".join(["?"] * len(columns))

                target.execute(
                    "INSERT INTO predictions (" +
                    ", ".join(columns) +
                    ") VALUES (" + placeholders + ")",
                    values
                )

                existing.add(values)
                copied += 1

            target.commit()

        except Exception:
            target.rollback()
            raise

        total = target.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

print("Records copied:", copied)
print("Total predictions:", total)
print("Migration completed successfully!")
