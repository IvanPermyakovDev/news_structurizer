from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..dto.jobs import RecordNowDTO
from ..services.job_storage_service import JobStorageService
from ..services.recording_service import RecordingService


router = APIRouter()
recording_service = RecordingService()
job_storage = JobStorageService()


@router.post("/record-now")
async def record_now(dto: RecordNowDTO):
    try:
        job_id = recording_service.record_now(dto)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}")
async def get_job(job_id: str):
    try:
        return job_storage.read_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}/report")
async def get_report(job_id: str):
    path = job_storage.report_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/json")


@router.get("/{job_id}/transcript")
async def get_transcript(job_id: str):
    path = job_storage.transcript_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    return FileResponse(path, media_type="text/plain")

