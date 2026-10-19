"""Challenge-response liveness: a server-chosen head turn that a flat photo cannot perform.

The cue is the nose tip's affine coordinates (u, v) in a frame built from the eyes and mouth:

    nose - eye_mid = (u - 0.5) * (eye_r - eye_l) + v * (mouth - eye_mid)

where eye_l/eye_r are the eyes on the image's left/right and eye_mid, mouth are midpoints. A
frontal face has its nose on the eye-mouth midline, so u = 0.5. Any affine transform of a flat
picture leaves (u, v) unchanged, and perspective changes them only slightly, so tilting, moving or
rotating a printed photo or a phone screen barely moves u. A real head turning about its vertical
axis moves the protruding nose sideways relative to the eyes: with a nose tip ~2.5 cm in front of
eyes 6.3 cm apart, u changes by about 0.36 * tan(yaw), i.e. ~0.13 at 20 degrees and ~0.2 at 30
degrees (rough head geometry; see tests/test_liveness.py). The service asks for a turn in a random
direction and accepts a capture only if the trajectory starts frontal, moves the requested way by
``turn_displacement``, moves continuously, stays on one face, and stays the same person throughout.

The landmark detector, not the geometry, is the weak point: its errors on rotated or strongly
foreshortened faces can mimic a turn. Two guards bound that. The face must not rotate in the image
plane by more than ``max_roll_change`` degrees, measured from pixels (ECC alignment between
consecutive frames) because YuNet's eye landmarks are pulled toward level and under-report roll.
And the turn must register while the face is still wide (``min_turn_width``).

What this does not stop (see docs/report.md): replaying a video, or flipping through photos, of the
victim turning the requested way, and 3D masks. Those need passive presentation-attack detection.
"""

import math
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum

import cv2
import numpy as np

from .engine import FaceEngine, similarity


class Direction(StrEnum):
    """Direction as the user experiences it (the preview is mirrored, like a mirror)."""

    LEFT = "left"
    RIGHT = "right"


# Turning to the user's own left points the nose toward the camera image's right edge, so u grows
# in the raw (unmirrored) frame. Checked against LFW images in research/evaluate_liveness.py.
_SIGN = {Direction.LEFT: 1.0, Direction.RIGHT: -1.0}


@dataclass(frozen=True)
class LivenessConfig:
    min_frames: int = 6
    max_frames: int = 48
    min_detected_fraction: float = 0.8
    baseline_frames: int = 2
    frontal_tolerance: float = 0.1  # |u - 0.5| allowed in the opening frames
    turn_displacement: float = 0.15  # required change in u toward the requested side
    max_step: float = 0.12  # largest change in u between consecutive detected frames
    min_box_iou: float = 0.3  # consecutive face boxes must overlap (no jumping around)
    min_identity_similarity: float = 0.3  # every frame vs the opening frames' embedding
    max_second_face_ratio: float = 0.25  # a second face larger than this fraction of the first is rejected
    # Landmark detectors err systematically on rotated or strongly foreshortened faces, which can
    # mimic a turn (research/results/liveness_results.json). The face must therefore stay within
    # this many degrees of in-plane rotation of the opening frames ...
    max_roll_change: float = 15.0
    # ... and register while the face is still at least this wide (eye distance over eye-to-mouth
    # distance, relative to the opening frames). A real head reaches the turn at about 23 degrees of
    # yaw (width ~0.92); a tilted photo only fakes one when foreshortened beyond ~45 degrees.
    min_turn_width: float = 0.85


@dataclass
class Observation:
    """What liveness needs from one frame. ``u`` is None when no face was found."""

    u: float | None = None
    v: float | None = None
    roll: float | None = 0.0  # in-plane rotation since the first frame, degrees (None: lost track)
    width: float = 0.0  # eye distance / eye-to-mouth distance
    box: np.ndarray | None = None
    embedding: np.ndarray | None = None
    second_face_ratio: float = 0.0


@dataclass
class LivenessResult:
    passed: bool
    reason: str
    message: str
    trace: list[float | None] = field(default_factory=list)
    peak_displacement: float = 0.0
    frontal_embedding: np.ndarray | None = None


MESSAGES = {
    "ok": "Liveness check passed.",
    "frame_count": "The capture had too few or too many frames. Please try again.",
    "face_not_found": "Keep your face in view for the whole capture.",
    "multiple_faces": "Only one person should be in view.",
    "not_frontal_at_start": "Start by looking straight at the camera.",
    "head_tilted": "Keep your head level while you turn.",
    "wrong_direction": "You turned the other way. Follow the arrow.",
    "turn_not_detected": "Turn your head further in the direction shown.",
    "discontinuous_motion": "Turn smoothly, without moving out of view.",
    "identity_changed": "The face changed during the capture.",
    "expired": "The challenge expired or was already used. Please start again.",
}


def face_geometry(landmarks: np.ndarray) -> tuple[float, float, float, float]:
    """(u, v, roll, width) from YuNet's 5x2 landmarks.

    (u, v) are the nose tip's coordinates (u = 0.5 when frontal), roll the eye line's angle in
    degrees, and width the eye distance over the eye-to-mouth distance.
    """
    eye_a, eye_b, nose, mouth_a, mouth_b = landmarks.astype(np.float64)
    left, right = (eye_a, eye_b) if eye_a[0] <= eye_b[0] else (eye_b, eye_a)
    mid = (left + right) / 2
    mouth = (mouth_a + mouth_b) / 2
    basis = np.column_stack([right - left, mouth - mid])
    u, v = np.linalg.solve(basis, nose - mid)
    roll = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    width = np.linalg.norm(right - left) / np.linalg.norm(mouth - mid)
    return float(u + 0.5), float(v), roll, float(width)


def nose_coordinates(landmarks: np.ndarray) -> tuple[float, float]:
    """(u, v) of the nose tip; u is 0.5 for a frontal face."""
    u, v, _, _ = face_geometry(landmarks)
    return u, v


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    iw = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    ih = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = iw * ih
    return float(inter / (aw * ah + bw * bh - inter)) if inter > 0 else 0.0


def observe(engine: FaceEngine, frame: np.ndarray, with_embedding: bool = True) -> Observation:
    faces = engine.detect(frame)
    if not faces:
        return Observation()
    face = faces[0]
    u, v, _, width = face_geometry(face.landmarks)  # landmark roll is biased; see track_roll
    return Observation(
        u=u,
        v=v,
        width=width,
        box=face.box,
        embedding=engine.embed(frame, face) if with_embedding else None,
        second_face_ratio=faces[1].area / face.area if len(faces) > 1 else 0.0,
    )


def _fail(reason: str, trace, peak: float = 0.0) -> LivenessResult:
    return LivenessResult(False, reason, MESSAGES[reason], trace, peak)


def decide(observations: list[Observation], direction: Direction, cfg: LivenessConfig) -> LivenessResult:
    """Apply the challenge rules to per-frame observations (pure; no models involved)."""
    trace = [o.u for o in observations]
    if not cfg.min_frames <= len(observations) <= cfg.max_frames:
        return _fail("frame_count", trace)
    seen = [o for o in observations if o.u is not None]
    if len(seen) < max(cfg.baseline_frames + 1, cfg.min_detected_fraction * len(observations)):
        return _fail("face_not_found", trace)
    if any(o.second_face_ratio > cfg.max_second_face_ratio for o in seen):
        return _fail("multiple_faces", trace)

    baseline = seen[: cfg.baseline_frames]
    if any(abs(o.u - 0.5) > cfg.frontal_tolerance for o in baseline):
        return _fail("not_frontal_at_start", trace)
    if any(o.roll is None for o in seen):
        return _fail("discontinuous_motion", trace)
    u0 = float(np.mean([o.u for o in baseline]))
    roll0 = float(np.mean([o.roll for o in baseline]))
    width0 = float(np.mean([o.width for o in baseline]))
    if any(abs(o.roll - roll0) > cfg.max_roll_change for o in seen):
        return _fail("head_tilted", trace)
    disp = np.array([_SIGN[direction] * (o.u - u0) for o in seen])
    peak = float(disp.max())
    turned = [
        i
        for i, (d, o) in enumerate(zip(disp, seen, strict=True))
        if d >= cfg.turn_displacement and o.width >= cfg.min_turn_width * width0
    ]
    if not turned:
        reason = "wrong_direction" if -disp.min() >= cfg.turn_displacement else "turn_not_detected"
        return _fail(reason, trace, peak)

    turn_index = turned[0]
    for prev, cur in zip(seen[:turn_index], seen[1 : turn_index + 1], strict=True):
        if abs(cur.u - prev.u) > cfg.max_step or box_iou(prev.box, cur.box) < cfg.min_box_iou:
            return _fail("discontinuous_motion", trace, peak)

    frontal = None
    if all(o.embedding is not None for o in seen):
        frontal = np.mean([o.embedding for o in baseline], axis=0)
        frontal /= np.linalg.norm(frontal)
        if any(similarity(o.embedding, frontal) < cfg.min_identity_similarity for o in seen):
            return _fail("identity_changed", trace, peak)

    return LivenessResult(True, "ok", MESSAGES["ok"], trace, peak, frontal)


_ECC_SIZE = 96
_ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)


def _face_region(gray: np.ndarray, box: np.ndarray) -> np.ndarray:
    """A square around ``box`` (twice its size), resampled to a fixed size."""
    x, y, w, h = box
    cx, cy, half = x + w / 2, y + h / 2, max(w, h)
    src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
    dst = np.float32([[0, 0], [_ECC_SIZE, 0], [0, _ECC_SIZE]])
    return cv2.warpAffine(gray, cv2.getAffineTransform(src, dst), (_ECC_SIZE, _ECC_SIZE), flags=cv2.INTER_AREA)


def _rotation(prev: np.ndarray, cur: np.ndarray) -> float | None:
    """In-plane rotation (degrees) taking ``prev`` to ``cur``, or None if ECC does not converge."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(prev, cur, warp, cv2.MOTION_EUCLIDEAN, _ECC_CRITERIA, None, 5)
    except cv2.error:
        return None
    return math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0])))


def track_roll(frames: list[np.ndarray], observations: list[Observation]) -> list[Observation]:
    """Copies of ``observations`` whose ``roll`` is the face's in-plane rotation since the first frame.

    Measured from pixels: the region around the previous frame's face is aligned to the same region
    in the next frame (rotation + translation), and the rotations are summed. A rigid photo rotated
    in its plane reads its true angle; a head turning about its vertical axis reads close to zero.
    """
    out, total, prev = [], 0.0, None
    for frame, o in zip(frames, observations, strict=True):
        if o.u is None:
            out.append(o)
            continue
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
        if prev is not None and total is not None:
            step = _rotation(_face_region(prev[0], prev[1]), _face_region(gray, prev[1]))
            total = None if step is None else total + step
        out.append(replace(o, roll=total))
        prev = (gray, o.box)
    return out


def observe_sequence(engine: FaceEngine, frames: list[np.ndarray], with_embedding: bool = True):
    return track_roll(frames, [observe(engine, f, with_embedding) for f in frames])


def check(engine: FaceEngine, frames: list[np.ndarray], direction: Direction, cfg: LivenessConfig):
    if not cfg.min_frames <= len(frames) <= cfg.max_frames:
        return _fail("frame_count", [])
    return decide(observe_sequence(engine, frames), direction, cfg)


@dataclass
class Challenge:
    id: str
    direction: Direction
    expires_at: float


class ChallengeStore:
    """Single-use, short-lived challenges held in memory (one process; see docs for scaling out)."""

    def __init__(self, ttl_seconds: float = 30.0, max_outstanding: int = 10_000):
        self.ttl = ttl_seconds
        self.max_outstanding = max_outstanding
        self._challenges: dict[str, Challenge] = {}
        self._lock = threading.Lock()

    def issue(self) -> Challenge | None:
        """A new challenge, or None if too many are outstanding (the endpoint is unauthenticated)."""
        now = time.monotonic()
        challenge = Challenge(
            id=secrets.token_urlsafe(16),
            direction=secrets.choice(list(Direction)),
            expires_at=now + self.ttl,
        )
        with self._lock:
            self._challenges = {k: c for k, c in self._challenges.items() if c.expires_at > now}
            if len(self._challenges) >= self.max_outstanding:
                return None
            self._challenges[challenge.id] = challenge
        return challenge

    def redeem(self, challenge_id: str) -> Challenge | None:
        """Remove and return the challenge if it exists and has not expired."""
        with self._lock:
            challenge = self._challenges.pop(challenge_id, None)
        if challenge is None or challenge.expires_at < time.monotonic():
            return None
        return challenge
