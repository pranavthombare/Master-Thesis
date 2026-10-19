"""Evaluate the head-turn liveness check (facepay/liveness.py) against simulated attacks on LFW.

Planar attacks render an LFW photo as a print held ~35 cm from a 640x480 webcam (pinhole model,
f = 600 px) and move it: tilt about each axis, hand-held wobble, and bending the print into an
arc (piecewise-planar strips). A phone screen is planar too, so it falls under the same model.
Every attack sequence opens with four still frontal frames, as an attacker would to pass the
"look straight" step, and is scored for both challenge directions.

Multi-photo attacks use identities with many LFW images:
  swap        frontal photo, then a turned photo of the same person
  flipbook    the same person's photos shown one after another in pose order
  replay      the same photos cross-faded into continuous footage: a stand-in for replaying a video
  impostor    frontal photo of one person, then turned photos of another

Bona fide behaviour cannot be measured from still photos. The script reports how far real faces'
nose coordinate ranges across LFW (the cue is not saturated by natural poses) and docs/report.md
describes the live protocol for measuring BPCER.

    uv run python -m research.evaluate_liveness       # -> research/results/liveness_*
"""

import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

from facepay.engine import FaceEngine
from facepay.liveness import (
    Direction,
    LivenessConfig,
    Observation,
    decide,
    nose_coordinates,
    observe,
    observe_sequence,
)

from .evaluate_lfw import LFW, read_pairs, unique_images
from .pipelines import MODEL_DIR, ROOT, nearest_centre

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = ROOT / "research" / "results"
CACHE = ROOT / "data" / "embeddings"
CANVAS_H, CANVAS_W = 480, 640
FOCAL = 600.0
PHOTO_CM = 12.0  # printed photo width
DISTANCE_CM = 35.0
LEAD_IN = 4  # frontal frames before the attacker starts moving
FRAMES = 24
RNG = np.random.default_rng(7)
LFW_SHAPE = (250, 250)
# right eye, left eye, nose, right mouth corner, left mouth corner (BGR)
LANDMARK_COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
CFG = LivenessConfig()
# the rules without the two guards added after the first simulation (head roll, foreshortening)
UNGUARDED = replace(CFG, max_roll_change=180.0, min_turn_width=0.0)


def decide_unguarded(observations: list[Observation], direction: Direction):
    """Decision without the guards: roll (and losing its track) plays no part."""
    return decide([replace(o, roll=0.0) for o in observations], direction, UNGUARDED)


# ---------------------------------------------------------------- rendering a held-up photo


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def render(
    photo: np.ndarray,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    tx=0.0,
    ty=0.0,
    z=DISTANCE_CM,
    bend=0.0,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """The photo as seen by the webcam. ``bend`` is the arc the print is curled into, in degrees."""
    h, w = photo.shape[:2]
    cm_per_px = PHOTO_CM / w
    rot = rotation(yaw, pitch, roll)
    canvas = background.copy() if background is not None else np.full((CANVAS_H, CANVAS_W, 3), 110, np.uint8)
    strips = 1 if bend == 0 else 25
    edges = np.linspace(0, w, strips + 1)
    radius = PHOTO_CM / math.radians(bend) if bend else None

    def project(x_px: float, y_px: float) -> tuple[float, float]:
        x = (x_px - w / 2) * cm_per_px
        y = (y_px - h / 2) * cm_per_px
        zz = 0.0
        if radius:  # curl around the vertical axis, edges toward the camera
            phi = x / radius
            x, zz = radius * math.sin(phi), -radius * (1 - math.cos(phi))
        px, py, pz = rot @ np.array([x, y, zz]) + np.array([tx, ty, z])
        return FOCAL * px / pz + CANVAS_W / 2, FOCAL * py / pz + CANVAS_H / 2

    for x0, x1 in zip(edges[:-1], edges[1:], strict=True):
        a, b = int(math.floor(x0)), min(int(math.ceil(x1)) + 1, w)
        src = np.float32([[a, 0], [b, 0], [b, h], [a, h]])
        dst = np.float32([project(a, 0), project(b, 0), project(b, h), project(a, h)])
        m = cv2.getPerspectiveTransform((src - [a, 0]).astype(np.float32), dst)
        strip = np.ascontiguousarray(photo[:, a:b])
        warped = cv2.warpPerspective(strip, m, (CANVAS_W, CANVAS_H))
        mask = cv2.warpPerspective(np.full(strip.shape[:2], 255, np.uint8), m, (CANVAS_W, CANVAS_H))
        canvas[mask > 0] = warped[mask > 0]
    return canvas


def ramp(n: int, end: float) -> np.ndarray:
    return end * (1 - np.cos(np.linspace(0, math.pi, n))) / 2  # ease in/out, like a hand


PLANAR_ATTACKS = {
    # name: function(sign) -> list of render kwargs, FRAMES long
    "tilt (yaw 60°)": lambda s: [dict(yaw=s * a) for a in ramp(FRAMES, 60)],
    "tilt (pitch 40°)": lambda s: [dict(pitch=s * a) for a in ramp(FRAMES, 40)],
    "rotate in-plane (40°)": lambda s: [dict(roll=s * a) for a in ramp(FRAMES, 40)],
    "slide sideways (8 cm)": lambda s: [dict(tx=s * a) for a in ramp(FRAMES, 8)],
    "hand-held wobble": lambda s: [
        dict(
            yaw=s * a + 6 * math.sin(i / 2),
            pitch=10 * math.sin(i / 3),
            roll=8 * math.sin(i / 4),
            tx=2 * math.sin(i / 5),
            z=DISTANCE_CM + 4 * math.sin(i / 6),
        )
        for i, a in enumerate(ramp(FRAMES, 45))
    ],
    "bend print (120° arc)": lambda s: [dict(bend=max(b, 1e-3)) for b in ramp(FRAMES, 120)],
    "bend 90° + tilt 45°": lambda s: [
        dict(bend=max(b, 1e-3), yaw=s * a) for b, a in zip(ramp(FRAMES, 90), ramp(FRAMES, 45), strict=True)
    ],
}


# ---------------------------------------------------------------- helpers


def lfw_landmarks(engine: FaceEngine, paths: list[Path]) -> dict[Path, tuple[float, float]]:
    """(u, v) of the face nearest the centre of each image (cached)."""
    cache = CACHE / "liveness_uv_v2.npz"
    rel = np.array([str(p.relative_to(LFW)) for p in paths])
    if cache.exists():
        d = np.load(cache)
        if np.array_equal(d["paths"], rel):
            return {p: tuple(uv) if ok else None for p, uv, ok in zip(paths, d["uv"], d["ok"], strict=True)}
    uv, ok = [], []
    for p in tqdm(paths, desc="landmarks", leave=False, disable=None):
        faces = engine.detect(cv2.imread(str(p)))
        if faces:
            face = nearest_centre(faces, LFW_SHAPE)
            uv.append(nose_coordinates(face.landmarks))
            ok.append(True)
        else:
            uv.append((np.nan, np.nan))
            ok.append(False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=rel, uv=np.array(uv), ok=np.array(ok))
    return {p: tuple(x) if k else None for p, x, k in zip(paths, uv, ok, strict=True)}


def direction_contact_sheet(uv: dict[Path, tuple], engine: FaceEngine) -> None:
    """Most extreme u at each end, with landmarks, to confirm which way 'left' is."""
    valid = sorted(((v[0], p) for p, v in uv.items() if v), key=lambda t: t[0])
    picks = [("lowest u", p, u) for u, p in valid[:4]] + [("highest u", p, u) for u, p in valid[-4:]]
    tiles = []
    for label, p, u in picks:
        img = cv2.imread(str(p))
        faces = engine.detect(img)
        face = nearest_centre(faces, LFW_SHAPE)
        for (x, y), colour in zip(face.landmarks, LANDMARK_COLOURS, strict=True):
            cv2.circle(img, (int(x), int(y)), 3, colour, -1)
        cv2.putText(img, f"{label} u={u:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(img)
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(RESULTS / "liveness_direction_check.png"), sheet)


# ---------------------------------------------------------------- experiments


def raw_peak(observations: list[Observation], direction: Direction) -> float:
    """Largest nose displacement toward ``direction`` relative to the lead-in, ignoring every other rule."""
    sign = 1.0 if direction is Direction.LEFT else -1.0
    seen = [o.u for o in observations if o.u is not None]
    u0 = float(np.mean(seen[:LEAD_IN]))
    return max(sign * (u - u0) for u in seen)


def planar_attacks(engine: FaceEngine, candidates: list[Path], n: int) -> tuple[dict, int]:
    """Best-case attacker: picks clean frontal photos, knows the challenge, and moves the photo
    whichever way suits it best. Returns per-attack statistics and the number of photos used."""
    stats = defaultdict(
        lambda: {"sequences": 0, "passed": 0, "passed_unguarded": 0, "peaks": [], "reasons": defaultdict(int)}
    )
    used = 0
    for p in tqdm(candidates, desc="planar attacks", leave=False, disable=None):
        if used == n:
            break
        photo = cv2.imread(str(p))
        background = np.clip(RNG.normal(110, 12, (CANVAS_H, CANVAS_W, 3)), 0, 255).astype(np.uint8)
        still = render(photo, background=background)
        first = observe(engine, still, with_embedding=False)
        if first.u is None or first.second_face_ratio > 0 or abs(first.u - 0.5) > 0.05:
            continue  # an attacker would choose a photo with one face that looks frontal on camera
        used += 1
        for name, trajectory in PLANAR_ATTACKS.items():
            variants = [
                observe_sequence(
                    engine,
                    [still] * LEAD_IN + [render(photo, background=background, **kw) for kw in trajectory(sign)],
                    with_embedding=False,
                )
                for sign in (1, -1)
            ]
            for direction in Direction:
                decisions = [decide(obs, direction, CFG) for obs in variants]
                passed = next((d for d in decisions if d.passed), None)
                s = stats[name]
                s["sequences"] += 1
                s["passed"] += passed is not None
                s["passed_unguarded"] += any(decide_unguarded(obs, direction).passed for obs in variants)
                s["peaks"].append(max(raw_peak(obs, direction) for obs in variants))
                s["reasons"][(passed or decisions[0]).reason] += 1
    return stats, used


def identity_photos(engine: FaceEngine, min_images: int) -> dict[str, list[tuple[float, Path]]]:
    """(u, photo) per person, u measured on the photo as held up to the camera; single-face only."""
    people = {}
    for d in sorted((LFW / "lfw").iterdir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) >= min_images:
            photos = []
            for f in files:
                o = observe(engine, render(cv2.imread(str(f))), with_embedding=False)
                if o.u is not None and o.second_face_ratio == 0:
                    photos.append((o.u, f))
            people[d.name] = sorted(photos)
    return people


def held(path: Path, background: np.ndarray) -> np.ndarray:
    """The camera frame of a photo held still in front of it."""
    return render(cv2.imread(str(path)), background=background)


def multi_photo_attacks(engine: FaceEngine, people: dict, trials: int) -> dict:
    out = {
        k: {"sequences": 0, "passed": 0, "passed_unguarded": 0, "reasons": defaultdict(int)}
        for k in ("swap", "flipbook", "replay", "impostor")
    }
    names = [n for n, photos in people.items() if any(abs(u - 0.5) <= 0.05 for u, _ in photos)]
    for t in tqdm(range(trials), desc="multi-photo attacks", leave=False, disable=None):
        name = names[t % len(names)]
        photos = people[name]
        direction = Direction.LEFT if t % 2 == 0 else Direction.RIGHT
        sign = 1 if direction is Direction.LEFT else -1
        frontal = min(photos, key=lambda x: abs(x[0] - 0.5))
        background = np.clip(RNG.normal(110, 12, (CANVAS_H, CANVAS_W, 3)), 0, 255).astype(np.uint8)
        lead = [held(frontal[1], background)] * LEAD_IN

        turned = [ph for ph in photos if sign * (ph[0] - frontal[0]) >= CFG.turn_displacement + 0.02]
        if turned:  # swap straight to a turned photo, held still
            frames = lead + [held(turned[0][1], background)] * (FRAMES - LEAD_IN)
            record(out["swap"], observe_sequence(engine, frames), direction)

        # flipbook: walk through the person's photos in pose order, small steps
        path, u = [], frontal[0]
        for pu, pp in sorted(photos, key=lambda x: sign * x[0]):
            step = sign * (pu - u)
            if 0.01 <= step <= CFG.max_step * 0.9:
                path.append(pp)
                u = pu
            if sign * (u - frontal[0]) >= CFG.turn_displacement + 0.02:
                break
        if sign * (u - frontal[0]) >= CFG.turn_displacement + 0.02:
            frames = lead + [held(pp, background) for pp in path for _ in range(2)]
            record(out["flipbook"], observe_sequence(engine, frames), direction)
            # replay: the same photos cross-faded into continuous footage, a stand-in for a video
            stills = [lead[0]] + [held(pp, background) for pp in path]
            frames = list(lead)
            for a, b in zip(stills, stills[1:], strict=False):
                frames += [cv2.addWeighted(a, 1 - k / 3, b, k / 3, 0) for k in (1, 2, 3)]
            record(out["replay"], observe_sequence(engine, frames[: CFG.max_frames]), direction)

        other = names[(t * 7 + 3) % len(names)]
        other_turned = [ph for ph in people[other] if sign * (ph[0] - 0.5) >= 0.05] if other != name else []
        if other_turned:  # someone else's turned photos after the victim's frontal one
            other_turned.sort(key=lambda x: sign * x[0])
            frames = lead + [held(pp, background) for _, pp in other_turned[: FRAMES - LEAD_IN]]
            record(out["impostor"], observe_sequence(engine, frames), direction)
    return out


def record(bucket: dict, observations: list[Observation], direction: Direction) -> None:
    result = decide(observations, direction, CFG)
    bucket["sequences"] += 1
    bucket["passed"] += result.passed
    bucket["passed_unguarded"] += decide_unguarded(observations, direction).passed
    bucket["reasons"][result.reason] += 1


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine(MODEL_DIR)
    paths = unique_images(read_pairs(LFW / "pairs.txt"))
    uv = lfw_landmarks(engine, paths)
    direction_contact_sheet(uv, engine)

    us = np.array([v[0] for v in uv.values() if v])
    candidates = [p for p, v in uv.items() if v and abs(v[0] - 0.5) <= 0.1]
    RNG.shuffle(candidates)

    planar, photos_used = planar_attacks(engine, candidates, n=150)
    people = identity_photos(engine, min_images=20)
    multi = multi_photo_attacks(engine, people, trials=200)
    within = {n: float(max(u for u, _ in ph) - min(u for u, _ in ph)) for n, ph in people.items() if len(ph) >= 2}

    results = {
        "config": CFG.__dict__,
        "lfw_u": {
            "images": int(len(us)),
            "mean": float(us.mean()),
            "sd": float(us.std()),
            "fraction_beyond_turn_displacement": float(np.mean(np.abs(us - np.median(us)) >= CFG.turn_displacement)),
        },
        "within_identity_u_range": {
            "identities": len(within),
            "median": float(np.median(list(within.values()))),
            "fraction_at_least_turn_displacement": float(
                np.mean(np.array(list(within.values())) >= CFG.turn_displacement)
            ),
        },
        "planar_attack_photos": photos_used,
        "planar_attacks": {
            name: {
                "sequences": s["sequences"],
                "passed": s["passed"],
                "pass_rate": s["passed"] / s["sequences"],
                "passed_unguarded": s["passed_unguarded"],
                "peak_displacement_max": float(np.max(s["peaks"])),
                "peak_displacement_p99": float(np.percentile(s["peaks"], 99)),
                "reasons": dict(s["reasons"]),
            }
            for name, s in planar.items()
        },
        "multi_photo_attacks": {
            k: {
                **{kk: vv for kk, vv in v.items() if kk != "reasons"},
                "reasons": dict(v["reasons"]),
                "pass_rate": v["passed"] / v["sequences"] if v["sequences"] else None,
            }
            for k, v in multi.items()
        },
    }
    (RESULTS / "liveness_results.json").write_text(json.dumps(results, indent=2))
    plot(planar, us)
    for name, r in results["planar_attacks"].items():
        print(
            f"{name:24s} pass {r['passed']:4d}/{r['sequences']:4d} (unguarded {r['passed_unguarded']:4d})  "
            f"peak max {r['peak_displacement_max']:.3f}  p99 {r['peak_displacement_p99']:.3f}  {r['reasons']}"
        )
    for name, r in results["multi_photo_attacks"].items():
        print(
            f"{name:24s} pass {r['passed']:4d}/{r['sequences']:4d} (unguarded {r['passed_unguarded']:4d})  "
            f"{r['reasons']}"
        )
    print(json.dumps({k: results[k] for k in ("lfw_u", "within_identity_u_range")}, indent=2))


def plot(planar, us) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in planar.items():
        a1.hist(s["peaks"], bins=np.linspace(-0.05, 0.3, 50), histtype="step", linewidth=1.5, label=name)
    a1.axvline(CFG.turn_displacement, color="black", linestyle="--", label="required turn")
    a1.set_xlabel("peak nose displacement toward the requested side")
    a1.set_ylabel("attack sequences")
    a1.set_title("Simulated photo attacks", fontsize=10)
    a1.legend(fontsize=7)
    a2.hist(us - np.median(us), bins=60, color="tab:blue", alpha=0.8)
    for x in (-CFG.turn_displacement, CFG.turn_displacement):
        a2.axvline(x, color="black", linestyle="--")
    a2.set_xlabel("u - median(u) across real LFW faces")
    a2.set_title("Natural pose variation in LFW", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "liveness_attacks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
