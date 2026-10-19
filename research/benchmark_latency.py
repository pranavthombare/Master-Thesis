"""CPU latency of each pipeline and of one full liveness check, on an otherwise idle machine.

uv run python -m research.benchmark_latency      # -> research/results/latency.json
"""

import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from facepay.engine import FaceEngine
from facepay.liveness import Direction, LivenessConfig, check

from .evaluate_lfw import LFW, RESULTS, read_pairs, unique_images
from .pipelines import MODEL_DIR, PipelineA, PipelineB, PipelineC, PipelineD

N_IMAGES = 300
REPEATS = 5


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def per_image_ms(pipeline, images: list[np.ndarray]) -> float:
    for img in images[:10]:  # warm-up
        pipeline(img)
    start = time.perf_counter()
    for img in images:
        pipeline(img)
    return (time.perf_counter() - start) / len(images) * 1000


def main() -> None:
    t = time.perf_counter()
    engine = FaceEngine(MODEL_DIR)
    load_s = time.perf_counter() - t

    template = json.loads((RESULTS / "lfw_results.json").read_text())["C"]["template"]
    images = [cv2.imread(str(p)) for p in unique_images(read_pairs(LFW / "pairs.txt"))[:N_IMAGES]]
    pipelines = {
        "A": PipelineA(),
        "B": PipelineB(engine),
        "C": PipelineC(engine, template["scale"], template["offset"]),
        "D": PipelineD(engine),
    }
    results = {
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "engine_load_s": load_s,
        "ms_per_image": {k: per_image_ms(p, images) for k, p in pipelines.items()},
    }

    # one liveness check as the service runs it: 30 webcam-sized frames, every frame detected and embedded
    canvas = np.full((480, 640, 3), 90, np.uint8)
    canvas[40:440, 120:520] = cv2.resize(images[0], (400, 400))
    frames = [canvas.copy() for _ in range(30)]
    check(engine, frames, Direction.LEFT, LivenessConfig())
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        check(engine, frames, Direction.LEFT, LivenessConfig())
        times.append(time.perf_counter() - t)
    results["check_30_frames_ms"] = float(np.median(times) * 1000)

    (RESULTS / "latency.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
