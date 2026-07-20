"""SQLite schema and deterministic offline payment fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    store_name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    operating_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS store_operator_access (
    operator_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    access_level TEXT NOT NULL,
    active INTEGER NOT NULL,
    PRIMARY KEY (operator_id, store_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE TABLE IF NOT EXISTS terminals (
    terminal_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    terminal_role TEXT NOT NULL,
    device_number TEXT NOT NULL,
    physical_serial TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL,
    installed_at TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE TABLE IF NOT EXISTS tid_assignments (
    tid_assignment_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    terminal_id TEXT NOT NULL,
    tid TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    assignment_status TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id)
);

CREATE TABLE IF NOT EXISTS activation_events (
    activation_event_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    terminal_id TEXT NOT NULL,
    activation_type TEXT NOT NULL,
    activation_status TEXT NOT NULL,
    tid_observed TEXT,
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id)
);

CREATE TABLE IF NOT EXISTS approval_events (
    approval_event_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    terminal_id TEXT NOT NULL,
    event_result TEXT NOT NULL,
    payment_channel TEXT NOT NULL,
    response_code TEXT,
    response_message TEXT,
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id)
);

CREATE TABLE IF NOT EXISTS support_routes (
    support_route_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    destination_type TEXT NOT NULL,
    destination_label TEXT NOT NULL,
    record_status TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);
"""


def create_database(path: str | Path = ':memory:') -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite fixture database with row access enabled."""
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def seed_s1(connection: sqlite3.Connection) -> None:
    connection.executemany('INSERT INTO stores VALUES (?, ?, ?, ?)', [
        ('STR-S1', 'Synthetic Cafe S1', 'Asia/Seoul', 'active'),
    ])
    connection.executemany('INSERT INTO store_operator_access VALUES (?, ?, ?, ?)', [
        ('OP-DEMO', 'STR-S1', 'review_case', 1),
    ])
    connection.executemany('INSERT INTO terminals VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
        ('TERM-S1-OLD', 'STR-S1', 'existing', 'DEV-S1-OLD', 'SERIAL-S1-OLD', 'activated', '2026-05-01T09:00:00+09:00', '2026-05-01T09:30:00+09:00'),
        ('TERM-S1-NEW', 'STR-S1', 'newly_installed', 'DEV-S1-NEW', 'SERIAL-S1-NEW', 'activated', '2026-06-20T14:30:00+09:00', '2026-06-20T15:00:00+09:00'),
    ])
    connection.executemany('INSERT INTO tid_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
        ('TIDA-S1-OLD', 'STR-S1', 'TERM-S1-OLD', 'TID-000100', '2026-05-01T09:30:00+09:00', None, 'active', '2026-05-01T09:30:00+09:00', '2026-05-01T09:30:01+09:00', '2026-05-01T09:30:02+09:00'),
        ('TIDA-S1-NEW', 'STR-S1', 'TERM-S1-NEW', 'TID-000100', '2026-06-20T15:00:00+09:00', None, 'active', '2026-06-20T15:00:00+09:00', '2026-06-20T15:00:01+09:00', '2026-06-20T15:00:02+09:00'),
    ])
    connection.executemany('INSERT INTO activation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', [
        ('ACT-S1-NEW', 'STR-S1', 'TERM-S1-NEW', 'terminal_open', 'succeeded', 'TID-000100', '2026-06-20T15:00:00+09:00', '2026-06-20T15:00:01+09:00', '2026-06-20T15:00:02+09:00'),
    ])
    connection.executemany('INSERT INTO approval_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
        ('APR-S1-001', 'STR-S1', 'TERM-S1-OLD', 'transport_error', 'card_terminal', 'SYN-APPROVAL-01', 'Authorization processing failed', '2026-06-20T15:07:00+09:00', '2026-06-20T15:07:01+09:00', '2026-06-20T15:07:02+09:00'),
        ('APR-S1-002', 'STR-S1', 'TERM-S1-OLD', 'transport_error', 'card_terminal', 'SYN-APPROVAL-01', 'Authorization processing failed', '2026-06-20T15:12:00+09:00', '2026-06-20T15:12:01+09:00', '2026-06-20T15:12:02+09:00'),
    ])
    connection.executemany('INSERT INTO support_routes VALUES (?, ?, ?, ?, ?, ?)', [
        ('ROUTE-S1', 'STR-S1', 'duplicate_tid', 'van_agency', 'Synthetic VAN install desk', 'active'),
    ])
    connection.commit()


__all__ = ['SCHEMA', 'create_database', 'open_database', 'seed_s1']


