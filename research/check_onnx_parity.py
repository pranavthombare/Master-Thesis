"""Check the exported OpenFace ONNX model reproduces the Keras model (run after the export).

uv run python -m research.check_onnx_parity
"""

import numpy as np
import onnxruntime as ort

from .pipelines import OPENFACE, ROOT


def main() -> None:
    fixture = np.load(ROOT / "data" / "openface_parity.npz")
    session = ort.InferenceSession(str(OPENFACE), providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: fixture["x"]})[0]
    diff = float(np.abs(out - fixture["y"]).max())
    print(f"max |keras - onnx| = {diff:.2e} over {len(out)} inputs")
    assert diff < 1e-5, "ONNX export does not match Keras"


if __name__ == "__main__":
    main()
