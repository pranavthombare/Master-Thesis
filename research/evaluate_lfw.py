"""Evaluate pipelines A-D on Labeled Faces in the Wild.

Protocols
  * LFW View 2 (pairs.txt): 10-fold verification accuracy, the threshold for each fold chosen on
    the other nine; ROC AUC and equal error rate over all 6,000 pairs.
  * Low-FAR operating points: every pair among the 7,701 images used by pairs.txt
    (~29.6M impostor pairs, ~8.1k genuine pairs) to estimate TAR at FAR down to 1e-5, which is the
    regime that matters for authorising payments.
  * Pipeline C's alignment template (scale, vertical offset) is tuned on LFW View 1
    (pairsDevTrain.txt) only.

    uv run python research/download_lfw.py
    uv run python -m research.evaluate_lfw          # -> research/results/
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from facepay.engine import FaceEngine

from .pipelines import MODEL_DIR, ROOT, PipelineA, PipelineB, PipelineC, PipelineD

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LFW = ROOT / "data" / "lfw"
CACHE = ROOT / "data" / "embeddings"
RESULTS = ROOT / "research" / "results"
FAR_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)
LEGACY_DISTANCES = {"verify (pay) d<=0.7": 0.7, "recognise d<=0.85": 0.85}


def image_path(name: str, n: str) -> Path:
    return LFW / "lfw" / name / f"{name}_{int(n):04d}.jpg"


def read_pairs(path: Path) -> list[list[tuple[Path, Path, bool]]]:
    """Folds of (image, image, same_person) from an LFW pairs file."""
    lines = path.read_text().split("\n")
    header = lines[0].split()
    n_folds, n = (int(header[0]), int(header[1])) if len(header) == 2 else (1, int(header[0]))
    rows = [line.split() for line in lines[1:] if line.strip()]
    folds, i = [], 0
    for _ in range(n_folds):
        fold = [(image_path(r[0], r[1]), image_path(r[0], r[2]), True) for r in rows[i : i + n]]
        i += n
        fold += [(image_path(r[0], r[1]), image_path(r[2], r[3]), False) for r in rows[i : i + n]]
        i += n
        folds.append(fold)
    return folds


def unique_images(folds) -> list[Path]:
    return sorted({p for fold in folds for a, b, _ in fold for p in (a, b)})


def embed_all(pipeline, key: str, paths: list[Path]):
    """Embeddings, acquisition flags and mean ms/image; cached in data/embeddings/<key>.npz."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{key}.npz"
    rel = np.array([str(p.relative_to(LFW)) for p in paths])
    if cache.exists():
        d = np.load(cache)
        if np.array_equal(d["paths"], rel):
            return d["emb"], d["acq"], float(d["ms"])
    embs, acq = [], []
    start = time.perf_counter()
    for p in tqdm(paths, desc=key, leave=False, disable=None):
        r = pipeline(cv2.imread(str(p)))
        embs.append(r.embedding)
        acq.append(r.acquired)
    ms = (time.perf_counter() - start) / len(paths) * 1000
    emb, acq = np.asarray(embs, np.float32), np.asarray(acq)
    np.savez(cache, paths=rel, emb=emb, acq=acq, ms=ms)
    return emb, acq, ms


def pair_scores(folds, index: dict[Path, int], emb: np.ndarray):
    scores = [np.array([emb[index[a]] @ emb[index[b]] for a, b, _ in fold]) for fold in folds]
    labels = [np.array([same for _, _, same in fold]) for fold in folds]
    return scores, labels


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising accuracy for the rule `same if score >= t`."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n = len(s)
    # accepting s[i:] is correct for every same pair in s[i:] and every different pair in s[:i]
    same_from = np.concatenate([np.cumsum(y[::-1])[::-1], [0]])
    diff_before = np.concatenate([[0], np.cumsum(~y)])
    i = int(np.argmax(same_from + diff_before))
    if i == 0:
        return float(s[0]) - 1e-6
    if i == n:
        return float(s[-1]) + 1e-6
    return float((s[i - 1] + s[i]) / 2)


def tenfold_accuracy(scores, labels) -> tuple[float, float]:
    accs = []
    for k in range(len(scores)):
        tr_s = np.concatenate([s for j, s in enumerate(scores) if j != k])
        tr_l = np.concatenate([lab for j, lab in enumerate(labels) if j != k])
        t = best_threshold(tr_s, tr_l)
        accs.append(np.mean((scores[k] >= t) == labels[k]))
    return float(np.mean(accs)), float(np.std(accs))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def all_pairs(emb: np.ndarray, identities: np.ndarray):
    """Genuine and impostor cosine scores over every unordered pair of images."""
    n = len(emb)
    genuine, impostor = [], np.empty(n * (n - 1) // 2, np.float32)
    filled = 0
    for i in range(n - 1):
        s = emb[i + 1 :] @ emb[i]
        same = identities[i + 1 :] == identities[i]
        genuine.append(s[same])
        k = int((~same).sum())
        impostor[filled : filled + k] = s[~same]
        filled += k
    return np.sort(np.concatenate(genuine)), np.sort(impostor[:filled])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def operating_point(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict:
    """Rates for the rule `accept if score >= threshold` (both arrays sorted ascending)."""
    false_accepts = len(impostor) - int(np.searchsorted(impostor, threshold, side="left"))
    true_accepts = len(genuine) - int(np.searchsorted(genuine, threshold, side="left"))
    lo, hi = wilson(true_accepts, len(genuine))
    return {
        "threshold": float(threshold),
        "far": false_accepts / len(impostor),
        "false_accepts": false_accepts,
        "tar": true_accepts / len(genuine),
        "tar_ci95": [lo, hi],
        "frr": 1 - true_accepts / len(genuine),
    }


def threshold_for_far(impostor: np.ndarray, far: float) -> float:
    """Smallest threshold whose FAR does not exceed `far` (impostor sorted ascending)."""
    k = int(far * len(impostor))  # at most k impostors may score >= threshold
    return float(np.nextafter(impostor[len(impostor) - k - 1], np.float32(np.inf)))


def tune_template(engine: FaceEngine) -> tuple[tuple[float, float], dict]:
    dev = read_pairs(LFW / "pairsDevTrain.txt")
    paths = unique_images(dev)
    index = {p: i for i, p in enumerate(paths)}
    aucs = {}
    # full grid; the optimum is interior (a smaller first grid had its optimum on the edge)
    scales = (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
    offsets = (-0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0)
    for scale, offset in ((sc, of) for sc in scales for of in offsets):
        emb, _, _ = embed_all(PipelineC(engine, scale, offset), f"dev_C_s{scale}_o{offset}", paths)
        s, lab = pair_scores(dev, index, emb)
        aucs[f"{scale},{offset}"] = float(roc_auc_score(lab[0], s[0]))
        print(f"  View 1 template scale {scale} offset {offset:+.2f}: AUC {aucs[f'{scale},{offset}']:.4f}")
    best = max(aucs, key=aucs.get)
    scale, offset = (float(x) for x in best.split(","))
    return (scale, offset), aucs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    engine = FaceEngine(MODEL_DIR)
    print("Tuning pipeline C's alignment template on View 1 (pairsDevTrain.txt)")
    (scale, offset), template_aucs = tune_template(engine)

    folds = read_pairs(LFW / "pairs.txt")
    paths = unique_images(folds)
    index = {p: i for i, p in enumerate(paths)}
    identities = np.array([p.parent.name for p in paths])
    pipelines = {
        "A": PipelineA(),
        "B": PipelineB(engine),
        "C": PipelineC(engine, scale, offset),
        "D": PipelineD(engine),
    }

    results, curves, distributions = {}, {}, {}
    for key, pipeline in pipelines.items():
        emb, acq, ms = embed_all(pipeline, getattr(pipeline, "cache_key", key), paths)
        scores, labels = pair_scores(folds, index, emb)
        flat_s, flat_l = np.concatenate(scores), np.concatenate(labels)
        acc, acc_sd = tenfold_accuracy(scores, labels)
        genuine, impostor = all_pairs(emb, identities)

        r = {
            "name": pipeline.name,
            "images": len(paths),
            "failure_to_acquire": float(1 - acq.mean()),
            "ms_per_image": ms,
            "lfw_accuracy": acc,
            "lfw_accuracy_sd": acc_sd,
            "auc": float(roc_auc_score(flat_l, flat_s)),
            "eer": eer(flat_s, flat_l),
            "genuine_pairs": len(genuine),
            "impostor_pairs": len(impostor),
            "operating_points": {
                f"{far:g}": operating_point(genuine, impostor, threshold_for_far(impostor, far)) for far in FAR_TARGETS
            },
        }
        if key == "A":  # what the 2019 app's fixed distance thresholds actually did
            r["legacy_thresholds"] = {
                label: {"distance": d, **operating_point(genuine, impostor, 1 - d * d / 2)}
                for label, d in LEGACY_DISTANCES.items()
            }
        if key == "C":
            r["template"] = {"scale": scale, "offset": offset}
            r["template_view1_auc"] = template_aucs
        results[key] = r

        fars = np.logspace(-6, 0, 300)
        idx = np.clip((len(impostor) * (1 - fars)).astype(int), 0, len(impostor) - 1)
        tars = 1 - np.searchsorted(genuine, impostor[idx], side="left") / len(genuine)
        curves[key] = (fars, tars)
        distributions[key] = (flat_s[flat_l], flat_s[~flat_l])
        ops = r["operating_points"]
        print(
            f"{pipeline.name:42s} acc {acc:.4f}±{acc_sd:.4f}  AUC {r['auc']:.4f}  EER {r['eer']:.4f}  "
            f"TAR@1e-3 {ops['0.001']['tar']:.4f}  TAR@1e-4 {ops['0.0001']['tar']:.4f}  "
            f"FTA {r['failure_to_acquire']:.4f}  {ms:.1f} ms/img"
        )

    (RESULTS / "lfw_results.json").write_text(json.dumps(results, indent=2))
    plot_roc(curves, pipelines)
    plot_distributions(distributions, results)
    print(f"wrote {RESULTS}")


def plot_roc(curves, pipelines) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, (fars, tars) in curves.items():
        ax.plot(fars, tars, label=pipelines[key].name, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False accept rate (impostor pairs accepted)")
    ax.set_ylabel("True accept rate")
    ax.set_title("LFW, all pairs among the View 2 images")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "roc_lfw.png", dpi=150)
    plt.close(fig)


def plot_distributions(distributions, results) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    bins = np.linspace(-0.4, 1, 71)
    for ax, key in zip(axes, ("A", "D"), strict=True):
        genuine, impostor = distributions[key]
        ax.hist(impostor, bins=bins, alpha=0.6, density=True, label="different people")
        ax.hist(genuine, bins=bins, alpha=0.6, density=True, label="same person")
        ax.set_title(results[key]["name"], fontsize=9)
        ax.set_xlabel("cosine similarity (LFW View 2 pairs)")
    for label, d in LEGACY_DISTANCES.items():
        axes[0].axvline(1 - d * d / 2, color="black", linestyle="--" if "0.7" in label else ":", label=label)
    t = results["D"]["operating_points"]["0.0001"]["threshold"]
    axes[1].axvline(t, color="black", linestyle="--", label=f"FAR 1e-4 threshold ({t:.3f})")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "score_distributions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
