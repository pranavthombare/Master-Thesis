"""Pretrained ONNX models: where they come from and how they are verified.

Both models are from the OpenCV Zoo and run on OpenCV's DNN module, so the service needs no
deep-learning framework. Licences: YuNet is MIT, SFace is Apache-2.0.
"""

import hashlib
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"

MODELS = {
    DETECTOR: (
        f"{_ZOO}/face_detection_yunet/{DETECTOR}",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    RECOGNIZER: (
        f"{_ZOO}/face_recognition_sface/{RECOGNIZER}",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(model_dir: Path) -> dict[str, Path]:
    """Return paths to both models, downloading any that are missing or fail their checksum."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, checksum) in MODELS.items():
        path = model_dir / name
        if not path.exists() or _sha256(path) != checksum:
            log.info("downloading %s", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != checksum:
                tmp.unlink()
                raise RuntimeError(f"checksum mismatch for {name}")
            tmp.replace(path)
        paths[name] = path
    return paths
