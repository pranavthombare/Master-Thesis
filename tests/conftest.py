import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from facepay.api import create_app
from facepay.config import Settings
from facepay.engine import Face

THRESHOLD = 0.5


class FakeEngine:
    """Stands in for FaceEngine. A frame is a flat colour: blue = person id * 20, green = u * 200.

    A red channel below 64 means "no face". Each person id has a fixed random embedding, so the
    whole API (liveness rules included) runs without the ONNX models.
    """

    def __init__(self):
        rng = np.random.default_rng(0)
        self.embeddings = {}
        for person in range(1, 12):
            e = rng.normal(size=128).astype(np.float32)
            self.embeddings[person] = e / np.linalg.norm(e)

    @staticmethod
    def _read(image):
        b, g, r = (float(np.median(image[..., c])) for c in range(3))
        return round(b / 20), g / 200, r

    def detect(self, image):
        person, u, r = self._read(image)
        if r < 64:
            return []
        landmarks = np.array([[40, 50], [60, 50], [40 + 20 * u, 65], [42, 80], [58, 80]], np.float32)
        return [Face(box=np.array([30, 30, 40, 60], np.float32), landmarks=landmarks, score=0.99, raw=None)]

    def embed(self, image, face):
        person, _, _ = self._read(image)
        return self.embeddings[person]


def frame(person: int, u: float, face: bool = True) -> str:
    img = np.zeros((64, 64, 3), np.uint8)
    img[...] = (person * 20, round(u * 200), 200 if face else 0)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


def turn(person: int, direction: str, frames: int = 14, peak: float = 0.25) -> list[str]:
    """A smooth head turn: 3 frontal frames, then u moves toward ``direction`` (left = u grows)."""
    sign = 1 if direction == "left" else -1
    us = [0.5] * 3 + [0.5 + sign * peak * (i + 1) / (frames - 3) for i in range(frames - 3)]
    return [frame(person, u) for u in us]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        model_dir=tmp_path / "models",
        secret_key=None,
        admin_token="s3cret",
        verify_threshold=THRESHOLD,
        identify_threshold=THRESHOLD,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, engine=FakeEngine())) as c:
        yield c


@pytest.fixture
def capture(client):
    """capture(person, comply=True): a challenge answered by ``person`` (turning the requested way if comply)."""

    def make(person: int, comply: bool = True, frames=None):
        challenge = client.post("/api/challenges").json()
        direction = challenge["direction"]
        if not comply:
            direction = "right" if direction == "left" else "left"
        return {"challenge_id": challenge["challenge_id"], "frames": frames or turn(person, direction)}

    return make
