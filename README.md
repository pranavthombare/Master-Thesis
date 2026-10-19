# FacePay

Face + PIN authentication for payments, with a head-turn liveness check designed to stop photos
and screens held up to the camera.

This repository started as an MSc project (2019): a Tkinter app that verified a webcam face
against a pickle file before "approving a payment". The 2026 revision turns it into a small
FastAPI service with a web UI, replaces the face pipeline, and, more importantly, measures it:

- **How accurate is it, at the error rates payments need?** An ablation on LFW from the 2019
  pipeline (A) to the deployed one (D), with thresholds calibrated at a fixed false-accept rate.
- **Can a photo fool it?** A server-chosen head-turn challenge whose cue does not change when a
  flat picture is moved, tested against simulated print, screen and bending attacks.
- **What does it not stop?** Documented and measured: replaying a video of the account holder.

The full write-up is [docs/report.md](docs/report.md).

## Results at a glance

LFW, 6,000-pair 10-fold accuracy and true accept rate at a false accept rate of 10⁻⁴
(measured over 29.6 M impostor pairs). Full tables: [docs/report.md](docs/report.md#6-results).

| Pipeline | Accuracy | TAR @ FAR 10⁻⁴ |
|---|---|---|
| A: 2019 system (Haar, unaligned crop, OpenFace) | 66.6 % | 0.8 % |
| B: + YuNet detector | 67.0 % | 0.6 % |
| C: + five-point alignment | 88.6 % | 7.5 % |
| **D: YuNet + SFace (deployed)** | **99.35 %** | **97.4 %** |

The 2019 "pay" threshold accepted **7.8 %** of impostor pairs on LFW. The service now uses the
FAR 10⁻⁴ threshold (cosine 0.416).

Liveness, against simulated attacks by an attacker who knows the challenge
([report §6.3](docs/report.md#63-rq3-liveness)):

| Attack | Accepted |
|---|---|
| Print or screen photo, tilted, rotated, slid, wobbled or bent (7 kinds) | 2 of 2,100 (0.1 %) |
| Swapping to a photo of the victim already turned | 0 of 175 |
| Victim's photo, then another person turning | 0 of 200 |
| **Continuous video of the victim turning** | **83 %, not stopped (documented)** |

The landmark detector turned out to report only about a tenth of a photo's real rotation, which
let rotated photos fake a turn. Rotation is now measured from pixels instead.

![FacePay UI](docs/img/ui-capture.png)

## Run it

```sh
uv sync
uv run facepay            # http://127.0.0.1:8000  (models download on first start, ~40 MB)
```

Open the page, enrol (email, PIN, consent, then follow the arrow), then pay. Browsers only allow
camera access on `localhost` or HTTPS. Configuration is by environment variable:

| Variable | Default | |
|---|---|---|
| `FACEPAY_SECRET_KEY` | generated dev key in the data dir | Fernet key encrypting face templates. **Set it in any real deployment.** |
| `FACEPAY_DATA_DIR` | `data/app` | SQLite database location |
| `FACEPAY_VERIFY_THRESHOLD` | calibrated, see `facepay/config.py` | cosine similarity for 1:1 matches |
| `FACEPAY_IDENTIFY_THRESHOLD` | calibrated | stricter threshold for 1:N search |
| `FACEPAY_MAX_FAILURES` / `FACEPAY_LOCKOUT_SECONDS` | 5 / 900 | account lockout |
| `FACEPAY_ADMIN_TOKEN` | unset (endpoint off) | enables `GET /api/audit` |

## How it works

```
browser (oat.ink UI)                         FastAPI service
────────────────────                         ─────────────────────────────────────────────
                       POST /api/challenges
  start camera  ─────────────────────────▶  issue single-use challenge {id, left|right}
  "look straight"
  "turn ← left"   ~30 JPEG frames
                       POST /api/pay  ─────▶  liveness: YuNet landmarks per frame → nose
                       {email, PIN,             coordinate u; frontal start, smooth turn the
                        amount, capture}        requested way, head level (pixel-tracked roll),
                                                one face, same person throughout
                                               verify: PIN (scrypt) + SFace embedding of the
                                                 frontal frames vs encrypted template
  approved / declined ◀───────────────────     audit log (no images stored)
```

| Endpoint | Needs | Does |
|---|---|---|
| `POST /api/challenges` | — | issue a liveness challenge (30 s, single use) |
| `POST /api/enrol` | email, PIN, consent, capture | create an account; refuses a face already enrolled |
| `POST /api/pay` | email, PIN, amount, capture | approve if PIN and live face both match |
| `POST /api/account/face` | email, PIN, capture | replace the face template (current face must match) |
| `POST /api/account/delete` | email, PIN, capture | delete the account and template |
| `POST /api/identify` | capture | 1:N search at a stricter threshold |

Failures for a wrong PIN, wrong face or unknown account return the same message. Five failures
lock the account for 15 minutes.

## Reproduce the research

```sh
uv sync --all-groups
uv run python research/download_lfw.py          # LFW + protocol files, checksummed (~180 MB)
uv run python -m research.evaluate_lfw          # pipelines A-D → research/results/lfw_*
uv run python -m research.evaluate_liveness     # attack simulation → research/results/liveness_*
uv run python -m research.roll_measurement      # landmark vs pixel rotation measurement
uv run python -m research.benchmark_latency     # latency
uv run pytest                                   # unit + API tests (integration ones need LFW)
uv run --with playwright==1.59.0 python scripts/e2e_browser.py   # UI end to end, fake webcam
```

`research/models/openface_nn4small2.onnx` is the 2019 network, exported from `legacy/` by
`research/export_openface_onnx.py` (needs the legacy Python 3.7 stack) and checked against Keras
by `research/check_onnx_parity.py` (max difference 2e-7).

## Layout

| Path | |
|---|---|
| `facepay/` | the service: `engine.py` (YuNet + SFace), `liveness.py`, `store.py`, `api.py`, `static/` (UI) |
| `research/` | LFW ablation, liveness attack simulation, results and figures |
| `tests/` | unit, API (fake engine) and integration (real models on LFW) tests |
| `scripts/e2e_browser.py` | drives the UI in headless Chromium with a synthetic webcam |
| `legacy/` | the 2019 Tkinter prototype, kept runnable as the baseline (see its README) |
| `docs/report.md` | the report |

## Credits and licences

- Face detection: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT); recognition: [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache-2.0), both from the OpenCV Zoo.
- UI: [Oat](https://oat.ink) 1.0.0 by Kailash Nadh (MIT), vendored in `facepay/static/vendor/oat`.
- Baseline network: OpenFace `nn4.small2` (Apache-2.0) via Keras-OpenFace and the deeplearning.ai
  Coursera assignment code in `legacy/` (see `legacy/README.md` for provenance).
- Evaluation data: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/), downloaded at run time, not redistributed.

This is a research prototype. It moves no money and is not certified for production use; see
[SECURITY.md](SECURITY.md).
