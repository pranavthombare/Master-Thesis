"""Runtime settings, read from FACEPAY_* environment variables.

The similarity thresholds are calibrated on LFW (research/results/lfw_results.json, pipeline D).
LFW is web photos of public figures, not webcam captures of customers: recalibrate on
in-domain data before relying on these numbers (docs/report.md, section 6).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .liveness import LivenessConfig

# Pipeline D operating points on LFW (all 29.6M impostor / 8.1k genuine pairs among View 2 images):
VERIFY_THRESHOLD = 0.4155  # FAR 1e-4, TAR 0.974
IDENTIFY_THRESHOLD = 0.4683  # FAR 1e-5, TAR 0.943


def _env(name: str, default):
    value = os.environ.get(f"FACEPAY_{name}")
    return default if value is None else type(default)(value)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data/app")))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", "models")))
    secret_key: str | None = field(default_factory=lambda: os.environ.get("FACEPAY_SECRET_KEY"))
    admin_token: str | None = field(default_factory=lambda: os.environ.get("FACEPAY_ADMIN_TOKEN"))
    # 1:1 verification (pay, update, delete) and enrolment duplicate check
    verify_threshold: float = field(default_factory=lambda: _env("VERIFY_THRESHOLD", VERIFY_THRESHOLD))
    # 1:N identification is stricter: the chance of some false match grows with the number enrolled
    identify_threshold: float = field(default_factory=lambda: _env("IDENTIFY_THRESHOLD", IDENTIFY_THRESHOLD))
    max_failures: int = field(default_factory=lambda: _env("MAX_FAILURES", 5))
    lockout_seconds: float = field(default_factory=lambda: _env("LOCKOUT_SECONDS", 900.0))
    challenge_ttl: float = field(default_factory=lambda: _env("CHALLENGE_TTL", 30.0))
    max_frame_pixels: int = 1920 * 1080
    liveness: LivenessConfig = field(default_factory=LivenessConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "facepay.sqlite3"
