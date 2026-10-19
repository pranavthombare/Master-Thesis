"""End-to-end check of the web UI in headless Chromium with a synthetic webcam.

The webcam feed is built from one LFW person's photos ordered by head pose (frontal, then a
gradual turn one way), cross-faded into continuous footage and written as a Y4M file for
Chromium's fake capture device. That is a video replay - exactly the attack the liveness check is
documented not to stop - so this exercises the full stack (camera capture, challenge, API,
storage), not liveness security.
The challenge direction is random and each feed turns one fixed way, so attempts that draw the
other direction are retried (every attempt is printed).

    uv run python research/download_lfw.py
    uv run --with playwright==1.59.0 python scripts/e2e_browser.py --screenshots docs/img
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facepay.engine import FaceEngine  # noqa: E402
from facepay.liveness import face_geometry  # noqa: E402

LFW = ROOT / "data" / "lfw" / "lfw"
FPS = 30
W, H = 640, 480
RETRYABLE = (
    "Follow the arrow",
    "Turn your head further",
    "Start by looking straight",
    "Turn smoothly",
    "Keep your head level",
)


def frame(photo: Path) -> np.ndarray:
    canvas = np.full((H, W, 3), (70, 80, 90), np.uint8)
    canvas[40:440, 120:520] = cv2.resize(cv2.imread(str(photo)), (400, 400), interpolation=cv2.INTER_CUBIC)
    return canvas


def poses(engine: FaceEngine, person: str) -> list[tuple[float, float, Path]]:
    """(u, roll, photo) for single-face photos, measured on the frame the server will see."""
    out = []
    for f in sorted((LFW / person).glob("*.jpg")):
        faces = engine.detect(frame(f))
        if len(faces) == 1:
            u, _, roll, _ = face_geometry(faces[0].landmarks)
            out.append((u, roll, f))
    return sorted(out)


def turn_sequence(engine: FaceEngine, person: str, reach: float = 0.2, step: float = 0.07):
    """(direction, photos): a frontal photo, then photos whose u moves one way in small steps."""
    photos = poses(engine, person)
    u0, roll0, frontal = min(photos, key=lambda x: abs(x[0] - 0.5))
    level = [(u, p) for u, roll, p in photos if abs(roll - roll0) <= 8]  # the head-level rule allows 15
    for direction, sign in (("right", -1), ("left", 1)):  # left = u grows (facepay/liveness.py)
        path, u = [frontal], u0
        for pu, pp in sorted(level, key=lambda x: sign * x[0]):
            if step * 0.3 <= sign * (pu - u) <= step:
                path.append(pp)
                u = pu
            if sign * (u - u0) >= reach:
                return direction, path
    raise RuntimeError(f"{person}: not enough pose variety for a turn sequence")


def write_video(path: Path, sequence: list[Path]) -> None:
    """Frontal 2.2 s, turn over ~2.4 s, hold 2.5 s (the page captures ~1-4.75 s after the camera starts).

    The turn is slow enough that the page's 8 fps sampling sees every cross-fade, as it would a
    real head turn; skipping from one photo straight to the next loses the liveness check's
    pixel track.
    """
    stills = [frame(p) for p in sequence]
    frames = [stills[0]] * int(2.2 * FPS)
    per = max(2, int(2.4 * FPS / (len(stills) - 1)))
    for a, b in zip(stills, stills[1:], strict=False):  # cross-fade, so the footage is continuous
        frames += [cv2.addWeighted(a, 1 - k / per, b, k / per, 0) for k in range(1, per + 1)]
    frames += [stills[-1]] * int(2.5 * FPS)
    with path.open("wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for img in frames:
            f.write(b"FRAME\n")
            f.write(cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).tobytes())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(url)


def submit(page: Page, tab: str, form: str, fields: dict, shot: Path | None = None) -> str:
    page.get_by_role("tab", name=tab).click()
    for name, value in fields.items():
        el = page.locator(f"{form} [name={name}]")
        if value is True:
            el.check()
        else:
            el.fill(value)
    page.evaluate("document.querySelector('#result').replaceChildren()")
    page.locator(f"{form} button[type=submit]").first.click()
    if shot:
        page.wait_for_timeout(3300)  # mid-turn
        page.screenshot(path=str(shot))
        pixelate(shot, page.locator("#stage").bounding_box(), page.locator("#prompt").bounding_box())
    alert = page.locator("#result [role=alert]")
    alert.wait_for(timeout=30000)
    return alert.inner_text()


def pixelate(path: Path, box: dict, keep_below: dict | None) -> None:
    """Pixelate the camera area of a screenshot, above the prompt: LFW photos are not ours to redistribute."""
    img = cv2.imread(str(path))
    x0, y0 = int(box["x"]), int(box["y"])
    x1, y1 = x0 + int(box["width"]), int(keep_below["y"]) if keep_below else y0 + int(box["height"])
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, roi.shape[1] // 24), max(1, roi.shape[0] // 24)), interpolation=cv2.INTER_AREA)
    img[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), img)


def until_passes(page: Page, *args, expect: str, attempts: int = 12, **kw) -> str:
    """Repeat while liveness fails for timing or direction reasons (the feed turns one fixed way)."""
    for i in range(attempts):
        text = submit(page, *args, **kw)
        kw.pop("shot", None)
        print(f"    attempt {i + 1}: {text}")
        if not any(r in text for r in RETRYABLE):
            assert expect in text, text
            return text
    raise AssertionError("liveness never passed")


def run_browser(video: Path, url: str, steps) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={video}",
            ]
        )
        page = browser.new_page(viewport={"width": 1180, "height": 760})
        page.goto(url)
        steps(page)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screenshots", type=Path, help="directory to save screenshots into")
    args = parser.parse_args()
    shots = args.screenshots
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    engine = FaceEngine(ROOT / "models")
    work = Path(tempfile.mkdtemp(prefix="facepay-e2e-"))
    for name, person in (("alice", "George_W_Bush"), ("mallory", "Colin_Powell")):
        direction, sequence = turn_sequence(engine, person)
        print(f"{name}: {person}, {len(sequence)} photos turning {direction}")
        write_video(work / f"{name}.y4m", sequence)

    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    env = {**os.environ, "FACEPAY_DATA_DIR": str(work / "data")}
    server = subprocess.Popen([sys.executable, "-m", "facepay", "--port", str(port)], cwd=ROOT, env=env)
    try:
        wait_for(url + "api/health")
        creds = {"email": "alice@example.com", "pin": "246810"}

        def as_alice(page: Page) -> None:
            if shots:
                page.screenshot(path=str(shots / "ui-start.png"))
            capture_shot = shots / "ui-capture.png" if shots else None
            print("enrol")
            until_passes(page, "Enrol", "#enrol-form", {**creds, "consent": True}, expect="Enrolled", shot=capture_shot)
            print("pay")
            until_passes(page, "Pay", "#pay-form", creds, expect="Approved")
            if shots:
                page.screenshot(path=str(shots / "ui-approved.png"))
            print("pay with wrong PIN")
            until_passes(page, "Pay", "#pay-form", {**creds, "pin": "111111"}, expect="Authentication failed")
            print("identify")
            until_passes(page, "Who am I?", "#identify-form", {}, expect="alice@example.com")

        def as_mallory(page: Page) -> None:
            print("impostor who knows Alice's PIN")
            until_passes(page, "Pay", "#pay-form", creds, expect="Authentication failed")
            if shots:
                page.screenshot(path=str(shots / "ui-rejected.png"))

        run_browser(work / "alice.y4m", url, as_alice)
        run_browser(work / "mallory.y4m", url, as_mallory)
        print("e2e ok")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
