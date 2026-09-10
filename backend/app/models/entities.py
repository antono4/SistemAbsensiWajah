"""ORM models: users, face_embeddings, attendance_logs (pgvector)."""

from datetime import date, datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(160), unique=True)
    department: Mapped[Optional[str]] = mapped_column(String(80))
    position: Mapped[Optional[str]] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    faces: Mapped[list["FaceEmbedding"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    attendance_logs: Mapped[list["AttendanceLog"]] = relationship(back_populates="user")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(128), nullable=False)
    model: Mapped[str] = mapped_column(String(40), default="dlib-face_recognition", nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    source_image: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped[User] = relationship(back_populates="faces")


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "check_type", "work_date", name="uq_attendance_user_type_date"),
        CheckConstraint("check_type IN ('check-in', 'check-out')", name="ck_attendance_check_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    check_type: Mapped[str] = mapped_column(String(10), nullable=False)
    check_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    work_date: Mapped[date] = mapped_column(nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # cosine distance (0 = identical)
    match_embedding_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("face_embeddings.id", ondelete="SET NULL"))
    source_image: Mapped[Optional[str]] = mapped_column(Text)
    liveness_passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped[User] = relationship(back_populates="attendance_logs")
