"""Pydantic DTO schemas."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Users ----------
class UserCreate(BaseModel):
    employee_id: str = Field(..., min_length=2, max_length=32, examples=["EMP-0001"])
    full_name: str = Field(..., min_length=3, max_length=120)
    email: Optional[str] = Field(None, max_length=160)
    department: Optional[str] = Field(None, max_length=80)
    position: Optional[str] = Field(None, max_length=80)


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    employee_id: str
    full_name: str
    email: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    status: str


# ---------- Face ----------
class RegisterFaceRequest(UserCreate):
    """Payload registrasi awal wajah."""


class FaceEnrollResult(BaseModel):
    user_id: int
    embedding_id: int
    full_name: str
    dimensions: int =128
    model: str
    quality_score: float


class FaceMatch(BaseModel):
    user_id: int
    embedding_id: int
    full_name: str
    employee_id: str
    cosine_distance: float
    matched: bool
    threshold: float


# ---------- Attendance ----------
class AttendanceCheckInOut(BaseModel):
    image_b64: str = Field(..., min_length=100, description="Foto (base64) dari kamera")
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AttendanceResult(BaseModel):
    status: str
    message: str
    match: Optional[FaceMatch] = None
    log_id: Optional[int] = None
    check_type: str
    check_time: Optional[datetime] = None
    work_date: Optional[date] = None
    liveness_passed: bool = True


# ---------- Rekap harian ----------
class DailyAttendanceRow(BaseModel):
    employee_id: str
    full_name: str
    check_in: Optional[datetime] = None
    check_out: Optional[datetime] = None
    work_hours: Optional[float] = None
    status: str


class DailyReport(BaseModel):
    date: date
    total_employees: int
    present_count: int
    rows: list[DailyAttendanceRow]
