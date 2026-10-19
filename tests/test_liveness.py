import math

import cv2
import numpy as np
import pytest

from facepay.liveness import (
    ChallengeStore,
    Direction,
    LivenessConfig,
    Observation,
    decide,
    nose_coordinates,
    track_roll,
)

CFG = LivenessConfig()
LANDMARKS = np.array([[40, 50], [60, 50], [50, 65], [42, 80], [58, 80]], np.float64)
BOX = np.array([30, 30, 40, 60], np.float32)


def unit(seed):
    v = np.random.default_rng(seed).normal(size=128)
    return v / np.linalg.norm(v)


ME, SOMEONE_ELSE = unit(1), unit(2)


def obs(u, embedding=ME, box=BOX, second=0.0, roll=0.0, width=0.67):
    return Observation(u=u, v=0.75, roll=roll, width=width, box=box, embedding=embedding, second_face_ratio=second)


def turning(direction, peak=0.25, n=12, start=0.5):
    sign = 1 if direction is Direction.LEFT else -1
    return [obs(start)] * 3 + [obs(start + sign * peak * (i + 1) / (n - 3)) for i in range(n - 3)]


@pytest.mark.parametrize("direction", list(Direction))
def test_smooth_turn_in_requested_direction_passes(direction):
    result = decide(turning(direction), direction, CFG)
    assert result.passed, result.reason
    assert result.peak_displacement == pytest.approx(0.25)
    assert np.linalg.norm(result.frontal_embedding) == pytest.approx(1.0)


def test_turning_the_other_way_fails():
    result = decide(turning(Direction.RIGHT), Direction.LEFT, CFG)
    assert (result.passed, result.reason) == (False, "wrong_direction")


def test_still_photo_fails():
    result = decide([obs(0.5)] * 12, Direction.LEFT, CFG)
    assert (result.passed, result.reason) == (False, "turn_not_detected")


def test_small_turn_fails():
    result = decide(turning(Direction.LEFT, peak=CFG.turn_displacement * 0.8), Direction.LEFT, CFG)
    assert result.reason == "turn_not_detected"


def test_must_start_frontal():
    result = decide(turning(Direction.LEFT, start=0.5 - 2 * CFG.frontal_tolerance), Direction.LEFT, CFG)
    assert result.reason == "not_frontal_at_start"


def test_photo_swap_jump_is_rejected():
    frames = [obs(0.5)] * 4 + [obs(0.5 + CFG.turn_displacement + 0.05)] * 8
    assert decide(frames, Direction.LEFT, CFG).reason == "discontinuous_motion"


def test_face_jumping_position_is_rejected():
    moved = np.array([300, 300, 40, 60], np.float32)
    frames = turning(Direction.LEFT)
    frames[7] = obs(frames[7].u, box=moved)
    assert decide(frames, Direction.LEFT, CFG).reason == "discontinuous_motion"


def test_different_person_mid_capture_is_rejected():
    frames = turning(Direction.LEFT)
    for i in range(5, len(frames)):
        frames[i] = obs(frames[i].u, embedding=SOMEONE_ELSE)
    assert decide(frames, Direction.LEFT, CFG).reason == "identity_changed"


def test_second_face_is_rejected():
    frames = turning(Direction.LEFT)
    frames[4] = obs(frames[4].u, second=0.6)
    assert decide(frames, Direction.LEFT, CFG).reason == "multiple_faces"


def test_face_lost_too_often_is_rejected():
    frames = turning(Direction.LEFT)
    for i in (4, 6, 8):
        frames[i] = Observation()
    assert decide(frames, Direction.LEFT, CFG).reason == "face_not_found"


@pytest.mark.parametrize("n", [CFG.min_frames - 1, CFG.max_frames + 1])
def test_frame_count_bounds(n):
    assert decide([obs(0.5)] * n, Direction.LEFT, CFG).reason == "frame_count"


def test_frontal_face_has_u_one_half():
    u, v = nose_coordinates(LANDMARKS)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(0.5)


def test_nose_coordinates_ignore_eye_order():
    swapped = LANDMARKS[[1, 0, 2, 4, 3]]
    assert nose_coordinates(swapped) == pytest.approx(nose_coordinates(LANDMARKS))


def test_flat_picture_under_any_affine_transform_keeps_its_coordinates():
    """The property the liveness cue rests on: a photo cannot fake a turn by being tilted or moved."""
    turned = LANDMARKS.copy()
    turned[2, 0] += 4  # a photo of a slightly turned face
    expected = nose_coordinates(turned)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.eye(2) + rng.normal(scale=0.4, size=(2, 2))
        if abs(np.linalg.det(a)) < 0.2 or np.linalg.det(a) < 0:
            continue  # degenerate or mirrored
        moved = turned @ a.T + rng.normal(scale=50, size=2)
        assert nose_coordinates(moved) == pytest.approx(expected, abs=1e-9)


def test_three_d_head_turn_moves_the_nose_coordinate():
    """A protruding nose shifts relative to the eyes when the head yaws; the sign matches Direction."""
    # rough head geometry in cm: eyes 6.3 apart, nose tip 2.5 cm in front of the eye plane
    head = np.array([[-3.15, 0, 0], [3.15, 0, 0], [0, 3.0, -2.5], [-2.5, 5.5, -1.0], [2.5, 5.5, -1.0]])

    def project(yaw_deg):
        t = math.radians(yaw_deg)
        rot = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        p = head @ rot.T + [0, 0, 40]
        return 600 * p[:, :2] / p[:, 2:3]

    # The user faces the camera, so their left is the image's right (+x). Turning left swings the
    # nose (z < 0, nearer the camera) toward +x, which is a negative yaw in this rotation.
    u0 = nose_coordinates(project(0))[0]
    users_left = nose_coordinates(project(-30))[0]
    users_right = nose_coordinates(project(30))[0]
    assert u0 == pytest.approx(0.5)
    assert users_left - u0 > CFG.turn_displacement  # Direction.LEFT expects u to grow
    assert u0 - users_right > CFG.turn_displacement


def test_challenges_are_single_use_and_expire():
    store = ChallengeStore(ttl_seconds=60)
    c = store.issue()
    assert c.direction in set(Direction)
    assert store.redeem(c.id) is c
    assert store.redeem(c.id) is None
    expired = ChallengeStore(ttl_seconds=-1).issue()
    assert ChallengeStore(ttl_seconds=-1).redeem(expired.id) is None


def test_challenge_directions_are_random():
    store = ChallengeStore()
    assert {store.issue().direction for _ in range(64)} == set(Direction)


def test_outstanding_challenges_are_capped():
    store = ChallengeStore(max_outstanding=3)
    issued = [store.issue() for _ in range(4)]
    assert issued[-1] is None
    store.redeem(issued[0].id)
    assert store.issue() is not None


def test_rotating_a_photo_is_rejected():
    """Landmarks drift on rotated faces and can fake a turn, so the eye line must stay level."""
    frames = [obs(0.5)] * 3 + [obs(0.5 + 0.03 * i, roll=4.0 * i) for i in range(1, 10)]
    assert decide(frames, Direction.LEFT, CFG).reason == "head_tilted"


def test_modest_head_tilt_is_allowed():
    frames = turning(Direction.LEFT)
    frames = [obs(o.u, roll=(CFG.max_roll_change - 2) * i / len(frames)) for i, o in enumerate(frames)]
    assert decide(frames, Direction.LEFT, CFG).passed


def test_turn_seen_only_under_heavy_foreshortening_is_rejected():
    """A photo tilted far enough to fool the landmarks is also much narrower than at the start."""
    frames = [obs(0.5)] * 3 + [obs(0.5 + 0.025 * i, width=0.67 * (1 - 0.05 * i)) for i in range(1, 10)]
    assert decide(frames, Direction.LEFT, CFG).reason == "turn_not_detected"


def test_real_turn_registers_before_the_face_narrows_much():
    # a head at ~23 degrees yaw is still ~92 % as wide; further into the turn it may narrow more
    frames = [obs(0.5)] * 3 + [obs(0.5 + 0.03 * i, width=0.67 * math.cos(math.radians(5 * i))) for i in range(1, 10)]
    assert decide(frames, Direction.LEFT, CFG).passed


def textured_frames(angles):
    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (240, 240, 3), dtype=np.uint8), (0, 0), 3)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)  # camera-like contrast
    return [cv2.warpAffine(texture, cv2.getRotationMatrix2D((120, 120), a, 1.0), (240, 240)) for a in angles]


def test_roll_is_measured_from_pixels():
    angles = [0, 0, 3, 6, 9, 12, 15, 18, 21]
    frames = textured_frames(angles)
    tracked = track_roll(frames, [obs(0.5, box=np.array([80, 80, 80, 80], np.float32))] * len(frames))
    for angle, o in zip(angles, tracked, strict=True):
        assert abs(abs(o.roll) - angle) < 1.0


def test_lost_roll_track_counts_as_discontinuous():
    frames = turning(Direction.LEFT)
    frames[6] = obs(frames[6].u, roll=None)
    assert decide(frames, Direction.LEFT, CFG).reason == "discontinuous_motion"
