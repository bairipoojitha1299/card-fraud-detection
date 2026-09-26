"""Tiny PostgreSQL helpers built on the psql client (no driver needed)."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PG_DEFAULTS = {
    "PGHOST": "localhost",
    "PGPORT": "5432",
    "PGUSER": "postgres",
    "PGPASSWORD": "postgres",
    "PGDATABASE": "fraud",
}


def _env() -> dict:
    env = os.environ.copy()
    for key, value in PG_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def _psql(args: list[str]) -> str:
    out = subprocess.run(["psql", "-v", "ON_ERROR_STOP=1", *args],
                         cwd=ROOT, env=_env(), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout


def run_sql_file(path: str | Path) -> str:
    """Execute a SQL script (relative paths resolve from the project root)."""
    return _psql(["-f", str(path)])


def query(sql: str, **read_csv_kwargs) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    return pd.read_csv(io.StringIO(_psql(["--csv", "-c", sql])), **read_csv_kwargs)
