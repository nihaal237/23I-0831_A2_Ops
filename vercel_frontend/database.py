
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv(Path(__file__).with_name(".env"))


def get_connection():
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
        connect_timeout=10,
    )


def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    ai_probability DOUBLE PRECISION NOT NULL,
                    threshold DOUBLE PRECISION NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ
                        NOT NULL DEFAULT NOW()
                )
            """)

    print("Neon prediction table is ready!")


def save_prediction(filename, result):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    filename,
                    prediction,
                    ai_probability,
                    threshold,
                    model_name
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    filename,
                    result["prediction"],
                    result["ai_probability"],
                    result["threshold"],
                    result["model"],
                ),
            )
            return cursor.fetchone()["id"]


def get_predictions(limit=100):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            records = cursor.fetchall()

            cursor.execute("SELECT COUNT(*) AS total FROM predictions")
            total = cursor.fetchone()["total"]

    return records, total


if __name__ == "__main__":
    create_tables()
