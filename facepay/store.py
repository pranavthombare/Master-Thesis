"""SQLite persistence: accounts with encrypted face templates and hashed PINs, plus an audit log.

No face images are stored. Templates (128 floats) are encrypted with Fernet (AES-128-CBC + HMAC)
under a key from the environment, and PINs are hashed with scrypt.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    pin_salt BLOB NOT NULL,
    pin_hash BLOB NOT NULL,
    template BLOB NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    action TEXT NOT NULL,
    email TEXT,
    outcome TEXT NOT NULL,
    similarity REAL,
    liveness TEXT,
    detail TEXT
);
"""


def hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def load_or_create_key(secret_key: str | None, data_dir: Path) -> bytes:
    """The Fernet key from config, or (development only) one generated and kept in data_dir."""
    if secret_key:
        return secret_key.encode()
    path = data_dir / "secret.key"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(Fernet.generate_key())
        log.warning("FACEPAY_SECRET_KEY not set; generated a development key at %s", path)
    return path.read_bytes().strip()


@dataclass
class Account:
    email: str
    template: np.ndarray
    failed_attempts: int
    locked_until: float


class Store:
    def __init__(self, path: Path, key: bytes):
        self.path = path
        self._fernet = Fernet(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _encrypt(self, template: np.ndarray) -> bytes:
        return self._fernet.encrypt(template.astype(np.float32).tobytes())

    def _decrypt(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(self._fernet.decrypt(blob), dtype=np.float32)

    def create(self, email: str, pin: str, template: np.ndarray) -> bool:
        """Insert an account; False if the email is already enrolled."""
        salt = secrets.token_bytes(16)
        now = time.time()
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO users (email, pin_salt, pin_hash, template, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (email, salt, hash_pin(pin, salt), self._encrypt(template), now, now),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def get(self, email: str) -> Account | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT email, template, failed_attempts, locked_until FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row is None:
            return None
        return Account(row[0], self._decrypt(row[1]), row[2], row[3])

    def check_pin(self, email: str, pin: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT pin_salt, pin_hash FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            hash_pin(pin, b"\0" * 16)  # same cost whether or not the account exists
            return False
        return hmac.compare_digest(hash_pin(pin, row[0]), row[1])

    def all_templates(self) -> list[tuple[str, np.ndarray]]:
        with self._connect() as db:
            rows = db.execute("SELECT email, template FROM users").fetchall()
        return [(email, self._decrypt(blob)) for email, blob in rows]

    def update_template(self, email: str, template: np.ndarray) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE users SET template = ?, updated_at = ? WHERE email = ?",
                (self._encrypt(template), time.time(), email),
            )

    def delete(self, email: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM users WHERE email = ?", (email,))

    def record_failure(self, email: str, max_failures: int, lockout_seconds: float) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1, "
                "locked_until = CASE WHEN failed_attempts + 1 >= ? THEN ? ELSE locked_until END "
                "WHERE email = ?",
                (max_failures, time.time() + lockout_seconds, email),
            )

    def reset_failures(self, email: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE email = ?", (email,))

    def audit(
        self,
        action: str,
        email: str | None,
        outcome: str,
        similarity: float | None = None,
        liveness: str | None = None,
        **detail,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO audit (at, action, email, outcome, similarity, liveness, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), action, email, outcome, similarity, liveness, json.dumps(detail) if detail else None),
            )

    def recent_audit(self, limit: int = 20) -> list[dict]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
