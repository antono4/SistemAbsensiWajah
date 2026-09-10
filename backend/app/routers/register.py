from typing import Annotated

import base64

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import FaceEmbedding, User
from app.schemas import FaceEnrollResult, UserCreate
from app.services import face_service
from app.services.face_service import FaceServiceError


router = APIRouter(prefix='/register-face', tags=['Face Registration'])


MAX_UPLOAD_BYTES =8 * 1024 * 1024


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                             detail='Gambar terlalu besar (maks 8MB)')
    return data


@router.post('', response_model=FaceEnrollResult, status_code=201)
def register_face(
    payload: UserCreate,
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
):
    """Registrasi karyawan baru + simpan embedding wajah 128-d."""

    existing = db.query(User).filter(User.employee_id == payload.employee_id).first()
    if existing:
        raise HTTPException(status_code=409, detail='employee_id sudah terdaftar')

    raw = _read_upload(image)
    b64 = base64.b64encode(raw).decode()
    try:
        rgb, _ = face_service.load_image_b64(b64)
        face = face_service.encode_single_face(rgb)
    except FaceServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    user = User(
        employee_id=payload.employee_id,
        full_name=payload.full_name,
        email=payload.email,
        department=payload.department,
        position=payload.position,
    )
    db.add(user)
    db.flush()

    emb = FaceEmbedding(
        user_id=user.id,
        embedding=face.embedding.tolist(),
        quality_score=face.quality_score,
        source_image='camera-upload',
    )
    db.add(emb)
    db.commit()
    db.refresh(user)
    db.refresh(emb)

    return FaceEnrollResult(
        user_id=user.id,
        embedding_id=emb.id,
        full_name=user.full_name,
        dimensions=128,
        model=emb.model,
        quality_score=emb.quality_score,
    )
