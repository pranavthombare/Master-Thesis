"""Real models on real faces. Needs ``models/`` (downloaded on first use) and LFW in data/lfw:

uv run python research/download_lfw.py
uv run --group research pytest -m integration
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from facepay.config import IDENTIFY_THRESHOLD, VERIFY_THRESHOLD
from facepay.engine import FaceEngine, similarity
from facepay.liveness import Direction, LivenessConfig, check, nose_coordinates

ROOT = Path(__file__).resolve().parents[1]
LFW = ROOT / "data" / "lfw" / "lfw"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not LFW.is_dir(), reason="LFW not downloaded (research/download_lfw.py)"),
]


@pytest.fixture(scope="module")
def engine():
    return FaceEngine(ROOT / "models")


def embedding(engine, person, n):
    img = cv2.imread(str(LFW / person / f"{person}_{n:04d}.jpg"))
    faces = engine.detect(img)
    assert faces, f"no face in {person} {n}"
    return engine.embed(img, faces[0])


def test_thresholds_are_calibrated():
    assert 0 < VERIFY_THRESHOLD < IDENTIFY_THRESHOLD < 1


@pytest.mark.parametrize("person", ["Colin_Powell", "Tony_Blair", "Gerhard_Schroeder"])
def test_same_person_is_accepted(engine, person):
    assert similarity(embedding(engine, person, 1), embedding(engine, person, 2)) >= VERIFY_THRESHOLD


@pytest.mark.parametrize(("a", "b"), [("Colin_Powell", "Tony_Blair"), ("Tony_Blair", "Gerhard_Schroeder")])
def test_different_people_are_rejected(engine, a, b):
    assert similarity(embedding(engine, a, 1), embedding(engine, b, 1)) < VERIFY_THRESHOLD


def test_tilted_photo_fails_liveness(engine):
    from research.evaluate_liveness import render

    def frontality(path):
        faces = engine.detect(cv2.imread(str(path)))
        return abs(nose_coordinates(faces[0].landmarks)[0] - 0.5) if len(faces) == 1 else 1.0

    photo = cv2.imread(str(min(sorted((LFW / "Colin_Powell").glob("*.jpg"))[:30], key=frontality)))
    frames = [render(photo)] * 4 + [render(photo, yaw=a) for a in np.linspace(0, 50, 16)]
    for direction in Direction:
        result = check(engine, frames, direction, LivenessConfig())
        assert (result.passed, result.reason) == (False, "turn_not_detected")
