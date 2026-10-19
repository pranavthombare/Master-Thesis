# Legacy: the 2019 MSc prototype

This folder preserves the Tkinter application submitted in December 2019 so it can be run and
used as the baseline (pipeline A) in [`research/`](../research). It is **not maintained**; the
current system is the [`facepay`](../facepay) package.

## Provenance

Most of the machine-learning code here is not original to this project:

| File | Source |
|---|---|
| `inception_blocks.py`, `fr_utils.py` | deeplearning.ai *Convolutional Neural Networks* course (Coursera), week 4 assignment "Face Recognition for the Happy House", which builds on Victor Sy Wang's [Keras-OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace) |
| `weights/*.csv` | OpenFace `nn4.small2` weights ([Amos et al., 2016](https://cmusatyalab.github.io/openface/), Apache-2.0), exported to CSV by Keras-OpenFace |
| `triplet_loss`, `verify`, `recognize` in `inference.py` | adapted from the same assignment (including the 0.7 distance threshold) |
| `haarcascade_frontalface_default.xml` | OpenCV (Viola–Jones detector) |
| `camera.py` | webcam capture window adapted from a public Tkinter/OpenCV snippet |

The original work in 2019 was the Tkinter application (`main.py`, the capture flows in
`camera.py`) and the pickle-file user store.

## Running it

The original pins (TensorFlow 1.8, Python ≤3.6) can no longer be installed. The closest stack
that still installs is in `requirements.txt`:

```sh
uv venv --python 3.7 .venv-legacy
VIRTUAL_ENV=.venv-legacy uv pip install -r legacy/requirements.txt
cd legacy
python -c "import pickle; pickle.dump({}, open('encoding.pickle', 'wb'), protocol=4)"  # empty user store
mkdir -p images
../.venv-legacy/bin/python main.py
```

Notes from running it in 2026:

- `opencv-python` (non-headless) 4.2 segfaults when Tk starts after `cv2` is imported on Linux;
  the headless build avoids it.
- With uv's standalone Python 3.7 you may need `TCL_LIBRARY`/`TK_LIBRARY` pointing at the
  interpreter's `lib/tcl8.6` and `lib/tk8.6`.
- Start-up takes about 75 s while 226 CSV weight files are parsed.

## Known defects (found by driving the app end to end)

| Behaviour | Where |
|---|---|
| Anyone can replace any account's face via *Update*, then pay as that account | `camera.py`, `check == 2` |
| *Recognise* accepts distance ≤ 0.85; two different enrolled people were 0.75–0.82 apart, so an unenrolled person is named as someone else | `inference.py:68` |
| *Add User* raises `TypeError` (encodes the image before it is captured) | `main.py:171` |
| A user added in a session is erased by the next *Recognise* or *Quit* (stale in-memory copy written back) | `main.py:105`, `main.py:157` |
| *Pay* / *Delete* with an unknown email raise `KeyError` with no message | `inference.py:45`, `main.py:142` |
| *Switch Camera* raises `AttributeError` | `camera.py:222` |
| The gallery shows every user twice | `main.py:42` and `main.py:44-55` |
| No liveness check: a photo of the account holder is accepted | — |
| Face images and embeddings stored unencrypted in the working directory | `camera.py`, `encoding.pickle` |

These are addressed by the new system and measured in [`docs/report.md`](../docs/report.md).
