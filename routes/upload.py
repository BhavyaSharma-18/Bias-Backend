"""
routes/upload.py
Handles CSV / JSON file uploads, validates them, and stores a temporary copy.
"""

import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Upload"])

# Temp file path (same directory as where uvicorn is launched from)
TEMP_PATH = Path("temp_dataset.csv")


@router.post("/upload", summary="Upload a CSV or JSON dataset")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a CSV or JSON file, validate it, persist it as *temp_dataset.csv*,
    and return column names, a 5-row preview, and the total row count.
    """
    try:
        # ------------------------------------------------------------------ #
        # 1. Validate file extension
        # ------------------------------------------------------------------ #
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()

        if suffix not in (".csv", ".json"):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a CSV or JSON file.",
            )

        # ------------------------------------------------------------------ #
        # 2. Read raw bytes and parse into a DataFrame
        # ------------------------------------------------------------------ #
        raw_bytes = await file.read()

        try:
            if suffix == ".csv":
                df = pd.read_csv(io.BytesIO(raw_bytes))
            else:  # .json
                df = pd.read_json(io.BytesIO(raw_bytes))
        except Exception as parse_err:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse file: {parse_err}",
            )

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="The uploaded file contains no data.",
            )

        # ------------------------------------------------------------------ #
        # 3. Persist as temp_dataset.csv for downstream endpoints
        # ------------------------------------------------------------------ #
        df.to_csv(TEMP_PATH, index=False)

        # ------------------------------------------------------------------ #
        # 4. Build response payload
        # ------------------------------------------------------------------ #
        preview = df.head(5).fillna("").to_dict(orient="records")

        return JSONResponse(
            content={
                "columns": list(df.columns),
                "preview": preview,
                "row_count": int(len(df)),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
