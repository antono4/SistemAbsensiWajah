"""Router laporan: rekap harian dan ekspor."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import AttendanceLog, User
from app.schemas import DailyAttendanceRow, DailyReport

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/daily", response_model=DailyReport)
def daily_report(target_date: date, db: Session = Depends(get_db)):
    users = db.query(User).filter(User.status == "active").all()
    logs = db.query(AttendanceLog).filter(AttendanceLog.work_date == target_date).all()

    by_user = {}
    for log in logs:
        by_user.setdefault(log.user_id, []).append(log)
    rows = []
    for u in users:
        ul = by_user.get(u.id, [])
        ci = next((x.check_time for x in ul if x.check_type == "check-in"), None)
        co = next((x.check_time for x in ul if x.check_type == "check-out"), None)
        hours = None
        if ci and co:
            dur = co - ci
            hours = round(dur.total_seconds() / 3600, 2)
        rows.append(DailyAttendanceRow(
            employee_id=u.employee_id,
            full_name=u.full_name,
            check_in=ci,
            check_out=co,
            work_hours=hours,
            status="present" if ci else "absent",
        ))
    present = sum(1 for r in rows if r.status == "present")
    return DailyReport(date=target_date, total_employees=len(users), present_count=present, rows=rows)


@router.get("/export")
def export_report(format: str = "xlsx", target_date: date = None, db: Session = Depends(get_db)):
    day = target_date if target_date else date.today()
    report = daily_report(day, db)
    if format == "xlsx":
        return _export_xlsx(report)
    if format == "pdf":
        return _export_pdf(report)
    raise HTTPException(400, detail="Format tidak didukung. Gunakan xlsx atau pdf.")


def _export_xlsx(report: DailyReport) -> Response:
    try:
        import openpyxl
    except ImportError:
        raise HTTPException(500, detail="Instal openpyxl dulu: pip install openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = str(report.date)
    ws.append(["No", "NIP", "Nama", "Masuk", "Pulang", "Jam", "Status"])
    for i, r in enumerate(report.rows, start=1):
        ws.append([i, r.employee_id, r.full_name, r.check_in, r.check_out, r.work_hours, r.status])
    import io
    buf = io.BytesIO()
    wb.save(buf)
    name = "rekap-" + str(report.date) + ".xlsx"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=" + name},
    )


def _export_pdf(report: DailyReport) -> Response:
    raise HTTPException(501, detail="Ekspor PDF memerlukan library reportlab")
