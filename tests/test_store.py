import time

import numpy as np
import pytest
from cryptography.fernet import Fernet

from facepay.store import Store, load_or_create_key


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db.sqlite3", Fernet.generate_key())


def template(seed=0):
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def test_round_trip(store):
    assert store.create("a@example.com", "1234", template())
    account = store.get("a@example.com")
    np.testing.assert_array_equal(account.template, template())
    assert store.get("nobody@example.com") is None


def test_duplicate_email_is_refused(store):
    assert store.create("a@example.com", "1234", template())
    assert not store.create("a@example.com", "9999", template(1))


def test_pin_check(store):
    store.create("a@example.com", "1234", template())
    assert store.check_pin("a@example.com", "1234")
    assert not store.check_pin("a@example.com", "4321")
    assert not store.check_pin("nobody@example.com", "1234")


def test_templates_and_pins_are_not_stored_in_clear(store):
    t = template()
    store.create("a@example.com", "123456", t)
    raw = b"".join(f.read_bytes() for f in store.path.parent.glob(store.path.name + "*"))
    assert t.tobytes()[:16] not in raw
    assert b"123456" not in raw


def test_wrong_key_cannot_read_templates(tmp_path):
    Store(tmp_path / "db.sqlite3", Fernet.generate_key()).create("a@example.com", "1234", template())
    with pytest.raises(Exception):  # noqa: B017 - InvalidToken
        Store(tmp_path / "db.sqlite3", Fernet.generate_key()).get("a@example.com")


def test_lockout_after_max_failures(store):
    store.create("a@example.com", "1234", template())
    for _ in range(3):
        store.record_failure("a@example.com", max_failures=3, lockout_seconds=60)
    account = store.get("a@example.com")
    assert account.failed_attempts == 3
    assert account.locked_until > time.time()
    store.reset_failures("a@example.com")
    assert store.get("a@example.com").locked_until == 0


def test_update_and_delete(store):
    store.create("a@example.com", "1234", template())
    store.update_template("a@example.com", template(5))
    np.testing.assert_array_equal(store.get("a@example.com").template, template(5))
    store.delete("a@example.com")
    assert store.get("a@example.com") is None


def test_audit_log(store):
    store.audit("pay", "a@example.com", "approved", 0.9, "ok", amount="1.00")
    (entry,) = store.recent_audit()
    assert entry["action"] == "pay" and entry["similarity"] == 0.9 and "1.00" in entry["detail"]


def test_development_key_is_created_once_with_private_permissions(tmp_path):
    key = load_or_create_key(None, tmp_path)
    assert load_or_create_key(None, tmp_path) == key
    assert (tmp_path / "secret.key").stat().st_mode & 0o777 == 0o600
    assert load_or_create_key("explicit", tmp_path) == b"explicit"
