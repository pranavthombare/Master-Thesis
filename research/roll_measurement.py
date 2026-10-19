"""How well is in-plane rotation measured: from YuNet's eye landmarks, or from pixels (ECC)?

Clean frontal LFW photos are held up to the simulated camera and rotated in their plane in eight
steps to a known angle. For each angle the script reports the rotation read from the eye landmarks,
the rotation tracked photometrically by facepay.liveness.track_roll, and the error the rotation
induces in the nose coordinate u.

    uv run python -m research.roll_measurement      # -> research/results/roll_measurement.json
"""

import json

import cv2
import numpy as np

from facepay.engine import FaceEngine
from facepay.liveness import face_geometry, observe, observe_sequence

from .evaluate_liveness import LFW, RESULTS, render
from .pipelines import MODEL_DIR

ANGLES = (5, 10, 15, 20, 30, 40)
N_PHOTOS = 60
STEPS = 8


def clean_frontal_photos(engine: FaceEngine) -> list[np.ndarray]:
    photos = []
    for d in sorted((LFW / "lfw").iterdir()):
        img = cv2.imread(str(sorted(d.glob("*.jpg"))[0]))
        o = observe(engine, render(img), with_embedding=False)
        if o.u is not None and o.second_face_ratio == 0 and abs(o.u - 0.5) <= 0.05:
            photos.append(img)
        if len(photos) == N_PHOTOS:
            break
    return photos


def main() -> None:
    engine = FaceEngine(MODEL_DIR)
    photos = clean_frontal_photos(engine)
    rows = {}
    for angle in ANGLES:
        landmark, tracked, du = [], [], []
        for photo in photos:
            for sign in (1, -1):
                frames = [render(photo)] + [render(photo, roll=sign * angle * k / STEPS) for k in range(1, STEPS + 1)]
                faces0, faces1 = engine.detect(frames[0]), engine.detect(frames[-1])
                if not faces0 or not faces1:
                    continue
                u0, _, roll0, _ = face_geometry(faces0[0].landmarks)
                u1, _, roll1, _ = face_geometry(faces1[0].landmarks)
                landmark.append(abs(roll1 - roll0))
                du.append(abs(u1 - u0))
                last = observe_sequence(engine, frames, with_embedding=False)[-1]
                if last.roll is not None:
                    tracked.append(abs(last.roll))
        rows[angle] = {
            "sequences": len(landmark),
            "landmark_roll_median": float(np.median(landmark)),
            "tracked_roll_median": float(np.median(tracked)),
            "tracked_roll_abs_error_max": float(np.max(np.abs(np.array(tracked) - angle))),
            "tracking_lost": len(landmark) - len(tracked),
            "u_error_p95": float(np.percentile(du, 95)),
            "u_error_max": float(np.max(du)),
        }
        r = rows[angle]
        print(
            f"{angle:3d} deg  landmarks read {r['landmark_roll_median']:5.1f}  tracked {r['tracked_roll_median']:5.1f} "
            f"(max err {r['tracked_roll_abs_error_max']:4.1f}, lost {r['tracking_lost']})  "
            f"u error p95 {r['u_error_p95']:.3f} max {r['u_error_max']:.3f}"
        )
    (RESULTS / "roll_measurement.json").write_text(json.dumps({"photos": len(photos), "angles": rows}, indent=2))


if __name__ == "__main__":
    main()
