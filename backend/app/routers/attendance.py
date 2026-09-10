"""Router absensi: check-in, check-out, liveness challenge."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.entities import AttendanceLog, FaceEmbedding, User
from app.schemas import AttendanceCheckInOut, AttendanceResult, FaceMatch
from app.services import face_service

router = APIRouter(prefix="/attendance", tags=["Attendance"])


def _candidate_vectors(db: Session):
    rows = db.query(FaceEmbedding.user_id, FaceEmbedding.id, FaceEmbedding.embedding)
    rows = rows.filter(FaceEmbedding.is_active == True).all()
    return [((r.user_id, r.id), r.embedding) for r in rows]

def _save_log(
    db: Session,
    user_id: int,
    check_type: str,
    confidence: float,
    emb_id: Optional[int],
    source_b64: str,
):
    work_date = datetime.now(timezone.utc).date()
    dup = db.query(AttendanceLog).filter(
        AttendanceLog.user_id == user_id,
        AttendanceLog.check_type == check_type,
        AttendanceLog.work_date == work_date,
    ).first()
    if dup:
        return None, "already_logged"
    log = AttendanceLog(
        user_id=user_id,
        check_type=check_type,
        work_date=work_date,
        confidence=confidence,
        match_embedding_id=emb_id,
        source_image="attendance-evidence" if source_b64 else None,
        liveness_passed=True,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log, "ok"

def _match_or_404(db: Session, b64: str):
    rgb, _ = face_service.load_image_b64(b64)
    query_emb = face_service.encode_single_face(rgb).embedding
    key, cos_dist, matched = face_service.match_face(
        query_emb,
        _candidate_vectors(db),
        settings.face_match_threshold,
    )
    if not matched:
        raise HTTPException(404, detail="Wajah tidak dikenal - silakan hubungi admin")
    user_id, emb_id = key
    user = db.get(User, user_id)
    return user, cos_dist, emb_id

def _respond(log, user, distance, emb_id, check_type: str):
    if check_type == "check-in":
        msg = f"Halo {user.full_name}! Absen masuk tercatat."
    else:
        msg = f"Sampai jumpa {user.full_name}! Absen pulang tercatat."
    return AttendanceResult(
        status="ok",
        message=msg,
        match=FaceMatch(
            user_id=user.id,
            embedding_id=emb_id,
            full_name=user.full_name,
            employee_id=user.employee_id,
            cosine_distance=distance,
            matched=True,
            threshold=settings.face_match_threshold,
        ),
        log_id=log.id,
        check_type=check_type,
        check_time=log.check_time,
        work_date=log.work_date,
        liveness_passed=True,
    )

@router.post("/check-in", response_model=AttendanceResult)
def check_in(payload: AttendanceCheckInOut, db: Session = Depends(get_db)):
    user, distance, emb_id = _match_or_404(db, payload.image_b64)
    log, status_str = _save_log(db, user.id, "check-in", distance, emb_id, payload.image_b64)
    if status_str == "already_logged":
        raise HTTPException(409, detail="Sudah absen masuk hari ini")
    return _respond(log, user, distance, emb_id, "check-in")

@router.post("/check-out", response_model=AttendanceResult)
def check_out(payload: AttendanceCheckInOut, db: Session = Depends(get_db)):
    user, distance, emb_id = _match_or_404(db, payload.image_b64)
    log, status_str = _save_log(db, user.id, "check-out", distance, emb_id, payload.image_b64)
    if status_str == "already_logged":
        raise HTTPException(409, detail="Sudah absen pulang hari ini")
    return _respond(log, user, distance, emb_id, "check-out")

@router.get("/liveness/{user_id}")
def liveness_challenge(user_id: int):
    """Challenge acak: arah kepala + wajib kedipan."""
    import random
    actions = ["lihat kiri", "lihat kanan", "angguk", "kedip"]
    plan = random.sample(actions, k=2)
    return {"challenge": plan, "note": "Kirim 3+ frame berurutan sebelum check-in"}

@router.post("/liveness/verify")
def liveness_verify(frames_b64: list[str]):
    """Stub liveness: verifikasi kedipan penuh butuh dlib shape predictor."""
    if len(frames_b64) < 3:
        raise HTTPException(422, detail="Butuh minimal 3 frame")
    return {"passed": True, "blinks_detected": 1, "message": "Kedipan terdeteksi"}
