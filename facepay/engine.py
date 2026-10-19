"""Face detection, alignment and embedding.

YuNet finds faces and five landmarks; SFace embeds a face after a similarity-transform alignment
to its 112x112 template. Embeddings are L2-normalised, so cosine similarity is a dot product.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import DETECTOR, RECOGNIZER, ensure_models


@dataclass(frozen=True)
class Face:
    """One detected face, in pixel coordinates of the image it was found in.

    ``landmarks`` rows are YuNet's order: subject's right eye, left eye, nose tip, right mouth
    corner, left mouth corner. The subject's right eye is on the image's left.
    """

    box: np.ndarray  # x, y, w, h
    landmarks: np.ndarray  # (5, 2)
    score: float
    raw: np.ndarray  # the 15-value YuNet row, needed by alignCrop

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class FaceEngine:
    """Thread-safe wrapper around the two ONNX models (OpenCV DNN nets are not re-entrant)."""

    embedding_dim = 128

    def __init__(self, model_dir: Path, detection_threshold: float = 0.8):
        paths = ensure_models(model_dir)
        self._detector = cv2.FaceDetectorYN.create(str(paths[DETECTOR]), "", (320, 320), detection_threshold, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(paths[RECOGNIZER]), "")
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Face]:
        """All faces in a BGR image, largest first."""
        h, w = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, rows = self._detector.detect(image)
        if rows is None:
            return []
        faces = [
            Face(box=row[0:4].copy(), landmarks=row[4:14].reshape(5, 2).copy(), score=float(row[14]), raw=row)
            for row in rows
        ]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def align(self, image: np.ndarray, face: Face) -> np.ndarray:
        """112x112 BGR crop aligned to SFace's five-point template."""
        with self._lock:
            return self._recognizer.alignCrop(image, face.raw)

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        with self._lock:
            feature = self._recognizer.feature(aligned)
        feature = feature.reshape(-1).astype(np.float32)
        return feature / np.linalg.norm(feature)

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        return self.embed_aligned(self.align(image, face))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised embeddings."""
    return float(np.dot(a, b))
