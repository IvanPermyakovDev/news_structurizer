from fastapi import FastAPI
from .controllers.jobs_controller import router as jobs_router
from .controllers.recording_controller import router as recording_router

app = FastAPI(title="Audio Recorder Service")

app.include_router(recording_router, prefix="/api/v1/recording")
app.include_router(jobs_router, prefix="/api/v1/jobs")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
