from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from .orchestrator import Model7Orchestrator

logger = logging.getLogger(__name__)


class OrchestrateRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Prescription or OCR text to analyze")

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("input_text must not be empty")
        return value.strip()


class OrchestrateResponse(BaseModel):
    input_text: str
    medicines: list[dict[str, Any]]
    chat_response: str
    summary: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str


def create_app(orchestrator: Model7Orchestrator | None = None) -> FastAPI:
    app = FastAPI(
        title="MediGuard Model 7 API",
        description="Expose Model 7 orchestration through a REST API.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    orchestrator_instance = orchestrator or Model7Orchestrator()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Validation error for %s: %s", request.url.path, exc)
        return JSONResponse(status_code=422, content={"detail": _serialize_errors(exc.errors())})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning("HTTP error for %s: %s", request.url.path, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception for %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="model7")

    @app.post("/api/v1/orchestrate", response_model=OrchestrateResponse)
    async def orchestrate(payload: dict[str, Any]) -> OrchestrateResponse:
        try:
            request = OrchestrateRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_serialize_errors(exc.errors())) from exc

        try:
            result = orchestrator_instance.orchestrate(request.input_text)
            return OrchestrateResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Orchestration request failed")
            raise HTTPException(status_code=500, detail=f"error: {exc}") from exc

    return app


def _serialize_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for error in errors:
        entry = dict(error)
        if "ctx" in entry and isinstance(entry["ctx"], dict):
            entry["ctx"] = {key: _to_json_safe(value) for key, value in entry["ctx"].items()}
        serialized.append(entry)
    return serialized


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    return str(value)


app = create_app()
