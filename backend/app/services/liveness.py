from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


class LivenessError(Exception):
    pass


@dataclass
class LivenessResult:
    passed: bool
    blinks_detected: int
    message: str


def _eye_aspect_ratio(eye: np.ndarray) -> float:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C_ = np.linalg.norm(eye[0] - eye[3])
    denom = 2.0 * (B + C_)
    if denom ==0:
        return 0.0
    return float(A / denom)


def _extract_landmarks(frame: np.ndarray, predictor, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if not faces:
        raise LivenessError("Wajah tidak terdeteksi")
    if len(faces) > 1:
        raise LivenessError("Terlalu banyak wajah di frame")
    return predictor(gray, faces[0])


def _check_blink(prev_ear: float, cur_ear: float, threshold: float =0.22) -> bool:
    return prev_ear > threshold and cur_ear < threshold


def verify_blink(
    frames: list[np.ndarray],
    predictor,
    detector,
    required_blinks: int =1,
    ear_threshold: float =0.25,
    consecutive_ratio: float =0.5,
) -> LivenessResult:
    """BAHASA: verifikasi kedipan mata (EAR) antar 2+ frame)."""
    if len(frames) < 2:
        raise LivenessError("Butuh minimal 2 frame untuk cek kedipan")
    amplitudes: list[float] = []
    prev_ear: float =0.0
    blink_count =0
    for i in range(1, len(frames)):
        left1 = _eye_aspect_ratio(_left_eye(frames[i - 1]))
        left2 = _eye_aspect_ratio(_left_eye(frames[i]))
        right1 = _eye_aspect_ratio(_right_eye(frames[i - 1]))
        right2 = _eye_aspect_ratio(_right_eye(frames[i]))
        ear1 = (left1 + right1) / 2.0
        ear2 = (left2 + right2) / 2.0
        if _check_blink(ear1, ear2, ear_threshold):
            blink_count += 1
        prev_ear = ear2

    ok = blink_count >= required_blinks
    msg = ("Liveness OK" if ok else "Tidak terdeteksi kedipan mata")
    return LivenessResult(passed=ok, blinks_detected=blink_count, message=msg)


def _left_eye(landmarks):
    eye = [landmarks.part(i) for i in [36, 37, 38, 39, 40, 41]]
    return np.array([(p.x, p.y) for p in eye])


def _right_eye(landmarks):
    eye = [landmarks.part(i) for i in [42, 43, 44, 45, 46, 47]]
    return np.array([(p.x, p.y) for p in eye])
