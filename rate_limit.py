"""SQLite-backed per-IP daily rate limiter for the demo deployment.

The counter is intentionally lightweight: it lives in a single `usage.db` file
next to the app and resets if the file is wiped (which Streamlit Cloud does on
every redeploy). For a portfolio demo that's a feature, not a bug.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

DB_PATH = Path(__file__).parent / "usage.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
            ip    TEXT NOT NULL,
            day   TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (ip, day)
        )
        """
    )
    return conn


def _today() -> str:
    return date.today().isoformat()


def get_client_ip() -> str:
    """Best-effort client IP. Falls back to a per-session UUID when no headers
    are available (typically local dev). The fallback still gives each visitor
    a stable identity for the duration of their session."""
    try:
        headers = st.context.headers  # Streamlit >= 1.37
        forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    except Exception:
        pass

    if "_rl_fallback_id" not in st.session_state:
        st.session_state["_rl_fallback_id"] = f"local-{uuid.uuid4()}"
    return st.session_state["_rl_fallback_id"]


def check_and_increment(ip: str, cap: int = 5) -> tuple[bool, int]:
    """Atomically check today's count for `ip` and, if under `cap`, increment it.

    Returns (allowed, remaining_after_call).
    - allowed=True means this request was counted and may proceed.
    - allowed=False means the cap had already been reached; nothing was incremented.
    """
    today = _today()
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE ip = ? AND day = ?",
            (ip, today),
        ).fetchone()
        current = row[0] if row else 0

        if current >= cap:
            return False, 0

        new_count = current + 1
        conn.execute(
            """
            INSERT INTO usage (ip, day, count) VALUES (?, ?, ?)
            ON CONFLICT(ip, day) DO UPDATE SET count = excluded.count
            """,
            (ip, today, new_count),
        )
        conn.commit()
        return True, max(cap - new_count, 0)


def remaining(ip: str, cap: int = 5) -> int:
    today = _today()
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE ip = ? AND day = ?",
            (ip, today),
        ).fetchone()
    used = row[0] if row else 0
    return max(cap - used, 0)


def refund(ip: str) -> None:
    """Decrement today's counter for `ip` (clamped at 0). Used when an OpenAI
    call fails with `insufficient_quota` so the user isn't charged a slot for a
    request that produced no output."""
    today = _today()
    with _connect() as conn:
        conn.execute(
            "UPDATE usage SET count = MAX(count - 1, 0) WHERE ip = ? AND day = ?",
            (ip, today),
        )
        conn.commit()
