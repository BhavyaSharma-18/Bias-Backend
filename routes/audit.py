"""
routes/audit.py
Bias audit and mitigation endpoints.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from engine.metrics import run_bias_audit
from engine.mitigation import apply_reweighing

router = APIRouter(tags=["Audit"])

FIXED_PATH = Path("fixed_dataset.csv")


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #

class AuditRequest(BaseModel):
    outcome_col: str
    protected_attr: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.post("/audit", summary="Run a full bias audit on the uploaded dataset")
async def audit(body: AuditRequest):
    """
    Compute disparate impact, statistical parity difference, proxy variable
    detection, group representation, and a traffic-light verdict.
    """
    try:
        results = run_bias_audit(
            outcome_col=body.outcome_col,
            protected_attr=body.protected_attr,
        )
        return JSONResponse(content=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Audit failed: {exc}"
        )


@router.post("/fix", summary="Apply Reweighing mitigation to the uploaded dataset")
async def fix(body: AuditRequest):
    """
    Apply AIF360 Reweighing to reduce bias and save a corrected dataset
    (fixed_dataset.csv) that can be downloaded via /api/download-fixed.
    """
    try:
        result = apply_reweighing(
            outcome_col=body.outcome_col,
            protected_attr=body.protected_attr,
        )
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Mitigation failed: {exc}"
        )


@router.get("/download-fixed", summary="Download the bias-mitigated dataset")
async def download_fixed():
    """
    Return the Reweighed / fixed dataset as a downloadable CSV file.
    Call /api/fix first to generate it.
    """
    if not FIXED_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No fixed dataset found. Run /api/fix first.",
        )
    return FileResponse(
        path=str(FIXED_PATH),
        media_type="text/csv",
        filename="fixed_dataset.csv",
    )
