import os
import uuid
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, RealDictCursor

load_dotenv()

HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
DB_USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")
JOBS_TABLE_NAME = '"image_editable"'


def get_db_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=DB_USER,
        password=PASSWORD,
        port=PORT,
    )


def init_jobs_table() -> None:
    """Create jobs table if it does not exist."""
    query = """
    CREATE TABLE IF NOT EXISTS "image-editable" (
        job_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        status TEXT NOT NULL,
        polygons JSONB NOT NULL,
        input_image BYTEA NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        completed_at TIMESTAMPTZ,
        error_message TEXT,
        result JSONB
    );
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
        conn.commit()


def create_job(
    user_id: str, image_bytes: bytes, polygons_payload: list[list[dict[str, float]]]
) -> str:
    """Insert a new ROI processing job and return job_id."""
    job_id = str(uuid.uuid4())
    query = f"""
    INSERT INTO {JOBS_TABLE_NAME} (job_id, user_id, status, polygons, input_image, created_at, updated_at)
    VALUES (%s, %s, 'pending', %s, %s, NOW(), NOW());
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (job_id, user_id, Json(polygons_payload), psycopg2.Binary(image_bytes)),
            )
        conn.commit()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    """Fetch a single job record by id."""
    query = f"SELECT * FROM {JOBS_TABLE_NAME} WHERE job_id = %s;"
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (job_id,))
            row = cursor.fetchone()
    return dict(row) if row else None


def mark_job_running(job_id: str) -> None:
    """Set job status to running."""
    query = f"""
    UPDATE {JOBS_TABLE_NAME}
    SET status = 'running', updated_at = NOW()
    WHERE job_id = %s;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (job_id,))
        conn.commit()


def mark_job_succeeded(job_id: str, result_payload: dict[str, Any]) -> None:
    """Set job status to succeeded with result payload."""
    query = f"""
    UPDATE {JOBS_TABLE_NAME}
    SET status = 'succeeded',
        result = %s,
        error_message = NULL,
        completed_at = NOW(),
        updated_at = NOW()
    WHERE job_id = %s;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (Json(result_payload), job_id))
        conn.commit()


def mark_job_failed(job_id: str, error_message: str) -> None:
    """Set job status to failed with error message."""
    query = f"""
    UPDATE {JOBS_TABLE_NAME}
    SET status = 'failed',
        error_message = %s,
        completed_at = NOW(),
        updated_at = NOW()
    WHERE job_id = %s;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (error_message, job_id))
        conn.commit()