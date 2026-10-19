"""HTTP API and static frontend.

Every sensitive action takes a *capture*: the frames a browser recorded while the user answered a
liveness challenge issued by ``POST /api/challenges``. Paying, replacing a face and deleting an
account also need the account's PIN and a 1:1 face match; failures are reported with one generic
message so the response never reveals which factor was wrong.
"""

import base64
import binascii
import hmac
import logging
import re
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import liveness
from .config import Settings
from .engine import FaceEngine, similarity
from .store import Account, Store, load_or_create_key

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_FRAME_BYTES = 1024 * 1024
FRAME_WIDTH = 640  # frames are downscaled to this width before detection
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")


class Capture(BaseModel):
    challenge_id: str = Field(min_length=8, max_length=64)
    frames: list[str] = Field(min_length=1, max_length=64, description="JPEG data URLs, oldest first")


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    pin: str = Field(pattern=r"^\d{4,8}$")

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL.match(v):
            raise ValueError("not a valid email address")
        return v


class EnrolRequest(Credentials):
    consent: Literal[True] = Field(description="Explicit consent to biometric processing (GDPR Art. 9(2)(a))")
    capture: Capture


class PayRequest(Credentials):
    amount: Decimal = Field(gt=0, le=10000, max_digits=7, decimal_places=2)
    merchant: str = Field(default="Demo merchant", max_length=80)
    capture: Capture


class AccountRequest(Credentials):
    capture: Capture


class IdentifyRequest(BaseModel):
    capture: Capture


def fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def jpeg_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, read without decoding; None if not a JPEG."""
    if raw[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            return None
        marker = raw[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
        elif marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers without a length
            i += 2
        elif marker == 0xDA:  # scan data before any frame header
            return None
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOFn: precision, height, width
            if i + 9 > len(raw):
                return None
            return int.from_bytes(raw[i + 7 : i + 9], "big"), int.from_bytes(raw[i + 5 : i + 7], "big")
        else:
            i += 2 + int.from_bytes(raw[i + 2 : i + 4], "big")
    return None


def decode_frame(data_url: str, max_pixels: int) -> np.ndarray:
    _, _, payload = data_url.partition(",") if data_url.startswith("data:") else ("", "", data_url)
    if len(payload) > MAX_FRAME_BYTES * 4 // 3 + 4:
        raise fail(413, "frame_too_large", "A frame is too large.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise fail(400, "bad_frame", "A frame could not be decoded.") from None
    # Check the declared size before decoding: a tiny file can declare an enormous image.
    size = jpeg_size(raw)
    if size is None or 0 in size:
        raise fail(400, "bad_frame", "Frames must be JPEG images.")
    if size[0] * size[1] > max_pixels:
        raise fail(413, "frame_too_large", "A frame is too large.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise fail(400, "bad_frame", "A frame could not be decoded.")
    if img.shape[1] > FRAME_WIDTH:
        img = cv2.resize(img, (FRAME_WIDTH, round(img.shape[0] * FRAME_WIDTH / img.shape[1])))
    return img


class Service:
    """The operations behind the routes, independent of HTTP framing."""

    def __init__(self, settings: Settings, engine: FaceEngine, store: Store):
        self.settings = settings
        self.engine = engine
        self.store = store
        self.challenges = liveness.ChallengeStore(settings.challenge_ttl)

    def live_face(self, capture: Capture, action: str, email: str | None) -> liveness.LivenessResult:
        """Redeem the challenge and run the liveness check; raises on failure."""
        challenge = self.challenges.redeem(capture.challenge_id)
        if challenge is None:
            raise fail(400, "expired", liveness.MESSAGES["expired"])
        frames = [decode_frame(f, self.settings.max_frame_pixels) for f in capture.frames]
        result = liveness.check(self.engine, frames, challenge.direction, self.settings.liveness)
        if not result.passed:
            self.store.audit(
                action,
                email,
                "liveness_failed",
                liveness=result.reason,
                peak=round(result.peak_displacement, 4),
                frames=len(frames),
            )
            raise fail(422, result.reason, result.message)
        return result

    def authenticate(self, creds: Credentials, capture: Capture, action: str) -> tuple[Account, float, np.ndarray]:
        """PIN + live face matching the enrolled template. Same error whatever fails.

        Returns the account, the match score and the live capture's frontal embedding.
        """
        self._refuse_if_locked(creds.email, action)
        live = self.live_face(capture, action, creds.email)
        # again: concurrent attempts all pass the first check while inference is serialised
        account = self._refuse_if_locked(creds.email, action)
        pin_ok = self.store.check_pin(creds.email, creds.pin)
        score = similarity(live.frontal_embedding, account.template) if account else None
        if account is None or not pin_ok or score < self.settings.verify_threshold:
            if account:
                self.store.record_failure(creds.email, self.settings.max_failures, self.settings.lockout_seconds)
            outcome = "unknown_account" if account is None else ("bad_pin" if not pin_ok else "face_mismatch")
            self.store.audit(action, creds.email, outcome, score, "ok")
            raise fail(401, "auth_failed", "Authentication failed: the PIN or face did not match.")
        self.store.reset_failures(creds.email)
        return account, score, live.frontal_embedding

    def _refuse_if_locked(self, email: str, action: str) -> Account | None:
        account = self.store.get(email)
        if account and account.locked_until > time.time():
            self.store.audit(action, email, "locked")
            raise fail(423, "locked", "Too many failed attempts. Try again later.")
        return account

    def best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        best, best_score = None, -1.0
        for email, template in self.store.all_templates():
            if (score := similarity(embedding, template)) > best_score:
                best, best_score = email, score
        return best, best_score

    def enrol(self, req: EnrolRequest) -> dict:
        if self.store.get(req.email) is not None:
            raise fail(409, "exists", "This email is already enrolled.")
        live = self.live_face(req.capture, "enrol", req.email)
        _, score = self.best_match(live.frontal_embedding)
        if score >= self.settings.verify_threshold:
            self.store.audit("enrol", req.email, "duplicate_face", score, "ok")
            raise fail(409, "duplicate_face", "This face is already enrolled to another account.")
        if not self.store.create(req.email, req.pin, live.frontal_embedding):
            raise fail(409, "exists", "This email is already enrolled.")
        self.store.audit("enrol", req.email, "enrolled", None, "ok", consent=True)
        return {"email": req.email, "message": "Enrolled. You can now pay with your face and PIN."}

    def pay(self, req: PayRequest) -> dict:
        _, score, _ = self.authenticate(req, req.capture, "pay")
        self.store.audit("pay", req.email, "approved", score, "ok", amount=str(req.amount), merchant=req.merchant)
        # Scores stay in the audit log: returning them would let an attacker tune a spoof toward the threshold.
        return {"approved": True, "message": f"Payment of {req.amount:.2f} to {req.merchant} approved."}

    def replace_face(self, req: AccountRequest) -> dict:
        _, score, live_template = self.authenticate(req, req.capture, "replace_face")
        self.store.update_template(req.email, live_template)
        self.store.audit("replace_face", req.email, "updated", score, "ok")
        return {"message": "Your face template was replaced with the one just captured."}

    def delete(self, req: AccountRequest) -> dict:
        _, score, _ = self.authenticate(req, req.capture, "delete")
        self.store.delete(req.email)
        self.store.audit("delete", req.email, "deleted", score, "ok")
        return {"message": "Your account and face template were deleted."}

    def identify(self, req: IdentifyRequest) -> dict:
        live = self.live_face(req.capture, "identify", None)
        email, score = self.best_match(live.frontal_embedding)
        if email is None or score < self.settings.identify_threshold:
            self.store.audit("identify", None, "no_match", score if email else None, "ok")
            return {"email": None, "message": "No enrolled account matches this face."}
        self.store.audit("identify", email, "match", score, "ok")
        return {"email": email, "message": f"This is {email}."}


SECURITY_HEADERS = [
    (
        b"content-security-policy",
        b"default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; object-src 'none'; "
        b"base-uri 'none'; frame-ancestors 'none'",
    ),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(self), microphone=(), geolocation=()"),
]


class SecurityHeaders:
    """Add SECURITY_HEADERS to every HTTP response (plain ASGI, so the request stream is untouched)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + SECURITY_HEADERS
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _BodyTooLarge(HTTPException):
    """An HTTPException, so FastAPI's body parsing passes it through as a 413 instead of a 400."""

    def __init__(self):
        super().__init__(413, {"code": "too_large", "message": "Request too large."})


class BodyLimit:
    """Reject request bodies over ``max_bytes``. Counts bytes as they arrive, so a chunked body
    without a Content-Length header is limited too."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and not length.isdigit():
            return await self._reject(send, 400, "bad_request", "Bad Content-Length.")
        if length is not None and int(length) > self.max_bytes:
            return await self._reject(send, 413, "too_large", "Request too large.")

        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send, 413, "too_large", "Request too large.")

    @staticmethod
    async def _reject(send, status: int, code: str, message: str) -> None:
        response = JSONResponse({"detail": {"code": code, "message": message}}, status)
        await response({"type": "http"}, None, send)


def create_app(settings: Settings | None = None, engine: FaceEngine | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        face_engine = engine or FaceEngine(settings.model_dir)
        store = Store(settings.db_path, load_or_create_key(settings.secret_key, settings.data_dir))
        app.state.service = Service(settings, face_engine, store)
        yield

    # The interactive /docs pages load scripts from a CDN, which the CSP below forbids.
    app = FastAPI(title="FacePay", version="2.0.0", lifespan=lifespan, docs_url=None, redoc_url=None)

    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeaders)  # outermost, so rejections get the headers too

    def service(request: Request) -> Service:
        return request.app.state.service

    Svc = Annotated[Service, Depends(service)]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/challenges")
    def new_challenge(svc: Svc) -> dict:
        c = svc.challenges.issue()
        if c is None:
            raise fail(503, "busy", "Too many checks in progress. Try again shortly.")
        return {"challenge_id": c.id, "direction": c.direction.value, "expires_in": svc.settings.challenge_ttl}

    @app.post("/api/enrol", status_code=201)
    def enrol(req: EnrolRequest, svc: Svc) -> dict:
        return svc.enrol(req)

    @app.post("/api/pay")
    def pay(req: PayRequest, svc: Svc) -> dict:
        return svc.pay(req)

    @app.post("/api/account/face")
    def replace_face(req: AccountRequest, svc: Svc) -> dict:
        return svc.replace_face(req)

    @app.post("/api/account/delete")
    def delete(req: AccountRequest, svc: Svc) -> dict:
        return svc.delete(req)

    @app.post("/api/identify")
    def identify(req: IdentifyRequest, svc: Svc) -> dict:
        return svc.identify(req)

    @app.get("/api/audit")
    def audit(svc: Svc, limit: int = 50, x_admin_token: Annotated[str | None, Header()] = None) -> list[dict]:
        token = svc.settings.admin_token
        if not token or not hmac.compare_digest((x_admin_token or "").encode(), token.encode()):
            raise fail(404, "not_found", "Not found.")
        return svc.store.recent_audit(min(limit, 500))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
