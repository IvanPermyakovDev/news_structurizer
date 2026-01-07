from fastapi import APIRouter, HTTPException
from ..dto.config import CreateRecordingConfigDTO, DeleteRecordingConfigDTO
from ..services.recording_service import RecordingService

router = APIRouter()
recording_service = RecordingService()


@router.post("/configs")
async def create_recording_config(dto: CreateRecordingConfigDTO):
    try:
        config_id = recording_service.create_config(dto)
        return {"config_id": config_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/configs")
async def delete_recording_config(dto: DeleteRecordingConfigDTO):
    success = recording_service.delete_config(dto.config_id)
    if not success:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"message": "Config deleted"}
