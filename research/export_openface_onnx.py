"""Export the 2019 OpenFace nn4.small2 Keras model (legacy/) to ONNX for the baseline.

Runs under the legacy stack (Python 3.7, TensorFlow 1.15, Keras 2.2.4, tf2onnx 1.9) because
legacy/inception_blocks.py and legacy/fr_utils.py only work there. See research/README.md.

Writes:
  research/models/openface_nn4small2.onnx   the frozen network, NCHW float32 input (N, 3, 96, 96)
  data/openface_parity.npz                  inputs + Keras outputs, checked against ONNX by
                                            research/check_onnx_parity.py in the modern env
"""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, "legacy")
sys.path.insert(0, LEGACY)
os.chdir(LEGACY)  # fr_utils.load_weights() reads ./weights

from keras import backend as K  # noqa: E402

K.set_learning_phase(0)
K.set_image_data_format("channels_first")

import tensorflow as tf  # noqa: E402
from fr_utils import load_weights_from_FaceNet  # noqa: E402
from inception_blocks import faceRecoModel  # noqa: E402

model = faceRecoModel(input_shape=(3, 96, 96))
load_weights_from_FaceNet(model)

sess = K.get_session()
out_name = model.output.op.name
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [out_name])
frozen = tf.graph_util.remove_training_nodes(frozen)

pb_path = os.path.join(ROOT, "data", "openface_nn4small2.pb")
onnx_path = os.path.join(ROOT, "research", "models", "openface_nn4small2.onnx")
os.makedirs(os.path.dirname(pb_path), exist_ok=True)
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
with tf.gfile.GFile(pb_path, "wb") as f:
    f.write(frozen.SerializeToString())

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--graphdef",
        pb_path,
        "--inputs",
        model.input.name,
        "--outputs",
        model.output.name,
        "--opset",
        "11",
        "--output",
        onnx_path,
    ]
)

# Parity fixture: random NCHW inputs in [0, 1) (the range legacy/fr_utils.img_to_encoding feeds
# the network) with the Keras outputs, so research/check_onnx_parity.py can compare.
rng = np.random.RandomState(0)
x = rng.rand(8, 3, 96, 96).astype(np.float32)
y = model.predict(x)
np.savez(os.path.join(ROOT, "data", "openface_parity.npz"), x=x, y=y)
print("input", model.input.name, "output", model.output.name, "->", onnx_path)
