from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.audit import router as audit_router

app = FastAPI(
    title="AI Bias Detection API",
    description="A FastAPI backend for detecting and mitigating bias in datasets using AIF360.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(upload_router, prefix="/api")
app.include_router(audit_router, prefix="/api")


# ---------------------------------------------------------------------------
# Root health-check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Root endpoint — confirms the service is running."""
    return {"status": "ok", "message": "AI Bias Detection API is running."}


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
