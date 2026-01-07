from fastapi import FastAPI
from .controllers.recording_controller import router as recording_router

app = FastAPI(title="Audio Recorder Service")

app.include_router(recording_router, prefix="/api/v1/recording")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
