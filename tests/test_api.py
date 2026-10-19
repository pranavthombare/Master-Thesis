from conftest import frame, turn

ALICE, BOB, CAROL = 1, 2, 3


def enrol(client, capture, email="alice@example.com", person=ALICE, pin="1234"):
    return client.post("/api/enrol", json={"email": email, "pin": pin, "consent": True, "capture": capture(person)})


def pay(client, capture, email="alice@example.com", person=ALICE, pin="1234", **kw):
    body = {"email": email, "pin": pin, "amount": "12.50", "merchant": "Café", "capture": capture(person, **kw)}
    return client.post("/api/pay", json=body)


def test_enrol_then_pay(client, capture):
    r = enrol(client, capture)
    assert r.status_code == 201, r.text
    r = pay(client, capture)
    assert r.status_code == 200, r.text
    assert r.json()["approved"] is True
    assert "similarity" not in r.json()  # scores are not disclosed to clients


def test_email_is_normalised(client, capture):
    assert enrol(client, capture, email="  Alice@Example.COM ").status_code == 201
    assert pay(client, capture, email="alice@example.com").status_code == 200


def test_enrolment_needs_consent(client, capture):
    r = client.post(
        "/api/enrol", json={"email": "a@example.com", "pin": "1234", "consent": False, "capture": capture(ALICE)}
    )
    assert r.status_code == 422


def test_wrong_pin_wrong_face_and_unknown_account_look_the_same(client, capture):
    enrol(client, capture)
    responses = [
        pay(client, capture, pin="9999"),
        pay(client, capture, person=BOB),
        pay(client, capture, email="nobody@example.com"),
    ]
    assert {r.status_code for r in responses} == {401}
    assert len({r.json()["detail"]["message"] for r in responses}) == 1


def test_liveness_failure_blocks_payment(client, capture):
    enrol(client, capture)
    r = pay(client, capture, comply=False)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "wrong_direction"


def test_still_photo_is_rejected(client, capture):
    enrol(client, capture)
    r = pay(client, capture, frames=[frame(ALICE, 0.5)] * 14)
    assert r.json()["detail"]["code"] == "turn_not_detected"


def test_challenge_cannot_be_replayed(client, capture):
    enrol(client, capture)
    body = {"email": "alice@example.com", "pin": "1234", "amount": "1.00", "capture": capture(ALICE)}
    assert client.post("/api/pay", json=body).status_code == 200
    r = client.post("/api/pay", json=body)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "expired"


def test_unknown_challenge_is_rejected(client):
    body = {
        "email": "a@example.com",
        "pin": "1234",
        "amount": "1.00",
        "capture": {"challenge_id": "made-up-challenge", "frames": turn(ALICE, "left")},
    }
    assert client.post("/api/pay", json=body).json()["detail"]["code"] == "expired"


def test_same_face_cannot_enrol_twice(client, capture):
    assert enrol(client, capture).status_code == 201
    r = enrol(client, capture, email="alias@example.com")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "duplicate_face"


def test_same_email_cannot_enrol_twice(client, capture):
    enrol(client, capture)
    r = enrol(client, capture, person=BOB)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "exists"


def test_lockout_after_repeated_failures(client, capture):
    enrol(client, capture)
    for _ in range(5):
        assert pay(client, capture, pin="0000").status_code == 401
    r = pay(client, capture)  # right PIN and face, but locked
    assert r.status_code == 423


def test_success_resets_failure_count(client, capture):
    enrol(client, capture)
    for _ in range(4):
        pay(client, capture, pin="0000")
    assert pay(client, capture).status_code == 200
    for _ in range(4):
        pay(client, capture, pin="0000")
    assert pay(client, capture).status_code == 200


def test_replacing_a_face_needs_the_current_face(client, capture):
    """The 2019 app let anyone overwrite any account's face. Now it takes PIN + the enrolled face."""
    enrol(client, capture)
    r = client.post("/api/account/face", json={"email": "alice@example.com", "pin": "1234", "capture": capture(BOB)})
    assert r.status_code == 401
    assert pay(client, capture, person=BOB).status_code == 401  # Bob still cannot pay as Alice


def test_replace_face(client, capture):
    enrol(client, capture)
    r = client.post("/api/account/face", json={"email": "alice@example.com", "pin": "1234", "capture": capture(ALICE)})
    assert r.status_code == 200
    assert pay(client, capture).status_code == 200


def test_delete_account(client, capture):
    enrol(client, capture)
    body = {"email": "alice@example.com", "pin": "1234", "capture": capture(ALICE)}
    assert client.post("/api/account/delete", json=body).status_code == 200
    assert pay(client, capture).status_code == 401
    assert enrol(client, capture).status_code == 201  # the face can be enrolled again


def test_identify(client, capture):
    enrol(client, capture)
    enrol(client, capture, email="bob@example.com", person=BOB)
    r = client.post("/api/identify", json={"capture": capture(BOB)})
    assert r.json()["email"] == "bob@example.com"
    r = client.post("/api/identify", json={"capture": capture(CAROL)})
    assert r.json()["email"] is None


def test_validation(client, capture):
    bad = [
        {"email": "not-an-email", "pin": "1234"},
        {"email": "a@example.com", "pin": "12"},
        {"email": "a@example.com", "pin": "abcd"},
    ]
    for creds in bad:
        r = client.post("/api/pay", json={**creds, "amount": "1.00", "capture": capture(ALICE)})
        assert r.status_code == 422
    body = {"email": "a@example.com", "pin": "1234", "amount": "-1", "capture": capture(ALICE)}
    r = client.post("/api/pay", json=body)
    assert r.status_code == 422


def test_garbage_frames_are_rejected(client, capture):
    enrol(client, capture)
    c = capture(ALICE, frames=["data:image/jpeg;base64,!!!notbase64"] * 10)
    r = client.post("/api/pay", json={"email": "alice@example.com", "pin": "1234", "amount": "1", "capture": c})
    assert r.status_code == 400


def test_oversized_request_is_rejected(client):
    r = client.post(
        "/api/pay",
        content=b"x" * 10,
        headers={"content-length": str(64 * 1024 * 1024), "content-type": "application/json"},
    )
    assert r.status_code == 413


def test_security_headers_and_frontend(client):
    r = client.get("/")
    assert r.status_code == 200 and "FacePay" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/vendor/oat/oat.min.css").status_code == 200


def test_audit_requires_admin_token(client, capture):
    enrol(client, capture)
    assert client.get("/api/audit").status_code == 404
    assert client.get("/api/audit", headers={"x-admin-token": "wrong"}).status_code == 404
    entries = client.get("/api/audit", headers={"x-admin-token": "s3cret"}).json()
    assert entries[0]["action"] == "enrol" and entries[0]["outcome"] == "enrolled"


def test_no_images_are_written(client, capture, settings):
    enrol(client, capture)
    pay(client, capture)
    written = {p.suffix for p in settings.data_dir.rglob("*") if p.is_file()}
    assert written <= {".sqlite3", ".key", ".sqlite3-wal", ".sqlite3-shm", ""}


def test_bad_content_length_is_rejected(client):
    r = client.post("/api/challenges", headers={"content-length": "abc"})
    assert r.status_code == 400


def test_lock_is_rechecked_after_liveness(client, capture):
    """Failures from concurrent attempts can land while this one is in liveness; it must still be refused."""
    enrol(client, capture)
    svc = client.app.state.service
    original = svc.live_face

    def live_face_while_others_fail(*args, **kwargs):
        result = original(*args, **kwargs)
        for _ in range(5):
            svc.store.record_failure("alice@example.com", max_failures=5, lockout_seconds=900)
        return result

    svc.live_face = live_face_while_others_fail
    assert pay(client, capture).status_code == 423
