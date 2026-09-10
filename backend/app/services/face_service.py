"""Face encoding & matching service (128-d dlib vectors)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Optional

try:
    import face_recognition  # type: ignore
    _FR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FR_AVAILABLE = False

from PIL import Image
import numpy as np

from app.core.config import settings


class FaceServiceError(Exception):
    """Domain error mapped to HTTP 4xx."""


@dataclass
class DetectedFace:
    embedding: np.ndarray  # (128,) float32, L2-normalized
    quality_score: float
    bbox: tuple[int, int, int, int]  # (top, right, bottom, left)

def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """L2-normalize so that distances equal cosine distances."""
    norm = np.linalg.norm(vec)
    if norm ==0:
        raise FaceServiceError("Embedding bernilai nol")
    return (vec / norm).astype(np.float32)

def _image_from_b64(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise FaceServiceError("Gambar tidak valid (pastikan base64 foto])") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def load_image_b64(b64: str) -> tuple[np.ndarray, Image.Image]:
    """Decode base64 payload into an RGB numpy array."""
    if len(b64) > settings.image_max_bytes * 4 // 3:
        raise FaceServiceError("Ukuran gambar melebihi batas")
    img = _image_from_b64(b64)
    return np.asarray(img), img

def detect_faces(rgb: np.ndarray) -> list[DetectedFace]:
    """Extract embeddings + quality score for every face in the image."""
    if not _FR_AVAILABLE:
        raise FaceServiceError("Library face_recognition belum terinstal di server")

    face_locations = face_recognition.face_locations(rgb, model="hog")
    if not face_locations:
        raise FaceServiceError("Tidak ada wajah terdeteksi")

    encodings = face_recognition.face_encodings(rgb, known_face_locations=face_locations, num_jitters=2)

    faces: list[DetectedFace] = []
    for (top, right, bottom, left), enc in zip(face_locations, encodings):
        w = right - left
        h = bottom - top
        if min(w, h) < settings.min_face_size:
            continue
        area_ratio = (w * h) / (rgb.shape[0] * rgb.shape[1])
        quality = float(np.clip(min(area_ratio, 1.0) * 10, 0.3, 1.0))
        faces.append(
            DetectedFace(
                embedding=normalize_embedding(enc),
                quality_score=round(quality, 4),
                bbox=(top, right, bottom, left),
            )
        )

    if not faces:
        raise FaceServiceError(f"Wajah terdeteksi namun terlalu kecil (<{settings.min_face_size}px)")

    return faces

def encode_single_face(rgb: np.ndarray) -> DetectedFace:
    """Extract a single face; reject multi-face frames to avoid mixing vectors."""
    faces = detect_faces(rgb)
    if len(faces) > 1:
        raise FaceServiceError("Terlalu banyak wajah - pastikan hanya 1 orang di frame")
    return faces[0]

def _nearest_match(query: np.ndarray, faces: list[tuple[Any, np.ndarray]]) -> Optional[tuple[Any, float]]:
    """faces: list[((user_id, embedding_id), vector)]. Returns (key, cosine_distance)."""
    best, best_dist = None, float("inf")
    for key, vec in faces:
        sim = float(np.dot(query, vec))  # dot = cos-sim krn keduanya L2-normalized
        cos_dist = 1.0 - sim  # cosine distance in [0, 2]
        if cos_dist < best_dist:
            best, best_dist = key, cos_dist
    return (best, best_dist) if best is not None else None

def match_face(
    query: np.ndarray,
    candidate_embeddings: list[tuple[Any, np.ndarray]],
    threshold: Optional[float] = None,
) -> tuple[Optional[Any], float, bool]:
    """Find nearest embedding. Returns (key, cosine_distance, matched)."""
    threshold = settings.face_match_threshold if threshold is None else threshold
    if not candidate_embeddings:
        return None, float("inf"), False
    result = _nearest_match(query, candidate_embeddings)
    if result is None:
        return None, float("inf"), False
    key, cos_dist = result
    return key, cos_dist, bool(cos_dist <= threshold)

def image_to_webp_base64(rgb: np.ndarray, quality: int =82) -> str:
    """Encode a frame back to base64 WebP (audit evidence / source_image)."""
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
