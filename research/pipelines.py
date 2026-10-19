"""The face pipelines compared in the LFW study.

Each pipeline maps an image path to an L2-normalised embedding and reports whether a face was
actually acquired. When detection fails, every pipeline falls back to the same central crop
(LFW images are centred on the subject), so all 6,000 protocol pairs are always scored and the
failure-to-acquire rate is reported separately.

    A  2019 system as built: Haar cascade, *last* detection, 5 px padded crop, OpenFace nn4.small2
    B  A with YuNet detection (face nearest the image centre)
    C  B with five-point similarity alignment to a 96x96 template (scale/offset tuned on View 1)
    D  deployed system: YuNet + SFace's own five-point alignment + SFace (facepay.engine)
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from facepay.engine import Face, FaceEngine

ROOT = Path(__file__).resolve().parents[1]
HAAR = ROOT / "legacy" / "haarcascade_frontalface_default.xml"
OPENFACE = ROOT / "research" / "models" / "openface_nn4small2.onnx"
MODEL_DIR = ROOT / "models"

# ArcFace/SFace 112x112 five-point template (eyes, nose tip, mouth corners).
TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def central_crop(img: np.ndarray, size: int = 110) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return img[y : y + size, x : x + size]


def nearest_centre(faces: list[Face], shape) -> Face:
    cy, cx = shape[0] / 2, shape[1] / 2
    return min(faces, key=lambda f: (f.box[0] + f.box[2] / 2 - cx) ** 2 + (f.box[1] + f.box[3] / 2 - cy) ** 2)


class OpenFace:
    """The 2019 network (exported from legacy/ by export_openface_onnx.py), with its preprocessing."""

    def __init__(self):
        self.session = ort.InferenceSession(str(OPENFACE), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, crop96_bgr: np.ndarray) -> np.ndarray:
        # legacy/fr_utils.img_to_encoding: BGR->RGB, HWC->CHW, /255, rounded to 12 decimals
        x = np.around(np.transpose(crop96_bgr[..., ::-1], (2, 0, 1)) / 255.0, decimals=12)
        return self.session.run(None, {self.input: x[None].astype(np.float32)})[0][0]


@dataclass
class Result:
    embedding: np.ndarray
    acquired: bool


class PipelineA:
    name = "A: Haar + crop + OpenFace (2019)"

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(str(HAAR))
        self.net = OpenFace()

    def __call__(self, img: np.ndarray) -> Result:
        faces = self.cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
        crop = None
        if len(faces) > 0:
            x, y, w, h = faces[-1]  # legacy/camera.py keeps the last face the loop sees
            pad = 5
            crop = img[y - pad : y + h + pad, x - pad : x + w + pad]
            if crop.size == 0:  # negative start index wraps; the legacy app would crash here
                crop = None
        acquired = crop is not None
        crop = cv2.resize(crop if acquired else central_crop(img), (96, 96))
        return Result(self.net(crop), acquired)


class PipelineB:
    name = "B: YuNet + crop + OpenFace"

    def __init__(self, engine: FaceEngine):
        self.engine = engine
        self.net = OpenFace()

    def __call__(self, img: np.ndarray) -> Result:
        faces = self.engine.detect(img)
        if faces:
            x, y, w, h = nearest_centre(faces, img.shape).box.astype(int)
            pad = 5
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            crop = img[y0 : y + h + pad, x0 : x + w + pad]
        else:
            crop = central_crop(img)
        return Result(self.net(cv2.resize(crop, (96, 96))), bool(faces))


class PipelineC:
    name = "C: YuNet + 5-pt alignment + OpenFace"

    def __init__(self, engine: FaceEngine, scale: float, offset: float):
        # OpenFace was trained on tight crops (eyebrows to chin), so SFace's template is scaled up
        # about its centre and shifted up; both are tuned on LFW View 1 in evaluate_lfw.py.
        self.engine = engine
        self.net = OpenFace()
        self.template = ((TEMPLATE_112 / 112 - 0.5) * scale + 0.5 + [0, offset]) * 96
        self.cache_key = f"C_s{scale}_o{offset}"

    def __call__(self, img: np.ndarray) -> Result:
        faces = self.engine.detect(img)
        if faces:
            src = nearest_centre(faces, img.shape).landmarks.astype(np.float32)
            m, _ = cv2.estimateAffinePartial2D(src, self.template, method=cv2.LMEDS)
            crop = cv2.warpAffine(img, m, (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            crop = cv2.resize(central_crop(img), (96, 96))
        return Result(self.net(crop), bool(faces))


class PipelineD:
    name = "D: YuNet + alignment + SFace (deployed)"

    def __init__(self, engine: FaceEngine):
        self.engine = engine

    def __call__(self, img: np.ndarray) -> Result:
        faces = self.engine.detect(img)
        if faces:
            return Result(self.engine.embed(img, nearest_centre(faces, img.shape)), True)
        return Result(self.engine.embed_aligned(cv2.resize(central_crop(img), (112, 112))), False)
