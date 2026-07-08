"""
api.py
------
FastAPI service exposing medicine extraction endpoints for single and
batch prescription / OCR text inference.
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import Config
from inference import MedicineExtractionError, MedicineExtractor
from utils import get_logger

logger = get_logger("api")

_extractor: Optional[MedicineExtractor] = None


class MedicineRecord(BaseModel):
    name: str = ""
    strength: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    route: str = ""
    confidence: float = 0.0


class ExtractionResponse(BaseModel):
    medicines: List[MedicineRecord] = Field(default_factory=list)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Prescription or OCR text")


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of prescription texts")


class BatchExtractionResponse(BaseModel):
    results: List[ExtractionResponse]


def get_extractor() -> MedicineExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MedicineExtractor(checkpoint_path=Config.BEST_MODEL_PATH, device=Config.DEVICE)
    return _extractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading medicine extraction model for API startup...")
    get_extractor()
    logger.info("Medicine extraction API is ready.")
    yield


app = FastAPI(
    title="MediGuard Model-2: Medicine Extraction",
    description="Extract structured medicine information from prescription / OCR text.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "medicine-extraction", "device": str(Config.DEVICE)}


@app.post("/extract", response_model=ExtractionResponse)
def extract_medicines(request: TextRequest) -> ExtractionResponse:
    try:
        result = get_extractor().extract(request.text)
        return ExtractionResponse(**result)
    except MedicineExtractionError as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/extract/batch", response_model=BatchExtractionResponse)
def extract_medicines_batch(request: BatchTextRequest) -> BatchExtractionResponse:
    if len(request.texts) > Config.API_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit of {Config.API_MAX_BATCH_SIZE}.",
        )

    try:
        results = get_extractor().extract_batch(request.texts)
        return BatchExtractionResponse(results=[ExtractionResponse(**item) for item in results])
    except MedicineExtractionError as exc:
        logger.exception("Batch extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
