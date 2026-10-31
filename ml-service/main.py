"""
EnviroTrust ML Service — FastAPI Application

Exposes endpoints for health checks, model status inspection, and real-time
telemetry anomaly analysis.
Supports:
  1. POST /analyze/telemetry: Generalized environmental and operational cross-checking
  2. POST /analyze/cod: Legacy backward-compatible endpoint
  3. Adheres strictly to trust guidelines: returns HTTP 503 if authentic model is untrained
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from anomaly_detection import ModelNotTrainedError
from predict import EnviroTrustPredictor
from preprocessing import MissingFeatureError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnviroTrust.API")

# Predictor instances: authentic production model (untrained) and isolated synthetic demonstration model
predictor: Optional[EnviroTrustPredictor] = None
demo_predictor: Optional[EnviroTrustPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model artifacts on server startup."""
    global predictor, demo_predictor
    logger.info("Initializing EnviroTrust ML Service...")
    predictor = EnviroTrustPredictor()
    demo_predictor = EnviroTrustPredictor.get_demo_predictor()

    if predictor.is_trained():
        logger.info(f"Authentic production model loaded (version: {predictor.metadata.get('model_version')}).")
    else:
        logger.info("Authentic service initialized in UNTRAINED mode. Awaiting authentic industrial dataset.")

    if demo_predictor.is_trained():
        logger.info(f"Synthetic demo model loaded (version: {demo_predictor.metadata.get('model_version')}).")
    else:
        logger.info("Synthetic demo model not loaded.")

    yield
    logger.info("Shutting down EnviroTrust ML Service.")


def get_active_predictor(mode: Optional[str] = "authentic") -> EnviroTrustPredictor:
    """
    Retrieve appropriate predictor based on explicit mode parameter.
    Does NOT silently fall back to synthetic demo when authentic mode is requested.
    """
    global predictor, demo_predictor
    if predictor is None:
        predictor = EnviroTrustPredictor()
    if demo_predictor is None:
        demo_predictor = EnviroTrustPredictor.get_demo_predictor()

    if mode and mode.lower() in ["demo", "synthetic", "synthetic_demo"]:
        if not demo_predictor.is_trained():
            demo_predictor.reload()
        return demo_predictor

    return predictor


app = FastAPI(
    title="EnviroTrust ML Anomaly Detection Service",
    description=(
        "Microservice for detecting process-consistency anomalies and sensor tampering "
        "using scikit-learn Isolation Forest. Operates exclusively with authentic data."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Schemas for Request & Response
# ==============================================================================

class OperationalContext(BaseModel):
    """Industrial operational parameters from SCADA / plant operations."""
    model_config = ConfigDict(extra="allow")

    productionRate: Optional[float] = Field(None, description="Factory throughput rate (units/hr)")
    wastewaterFlow: Optional[float] = Field(None, description="Effluent volume discharge (L/hr)")
    treatmentDose: Optional[float] = Field(None, description="Coagulant/chemical treatment dosage")
    operatingMode: Optional[str] = Field("NORMAL", description="Operating mode: NORMAL, HIGH_LOAD, CLEANING, SHUTDOWN")
    energyConsumption: Optional[float] = Field(None, description="Industrial power consumption (kWh)")


class CodAnalysisRequest(BaseModel):
    """Legacy payload sent by backend for COD process-consistency cross-checking."""
    model_config = ConfigDict(extra="allow")

    readingId: Optional[str] = Field(None, description="Unique reading identifier")
    sensorId: Optional[str] = Field(None, description="Sensor identifier (e.g. COD-OUT-001)")
    actualCod: float = Field(..., description="Chemical Oxygen Demand in mg/L reported by the sensor")
    timestamp: Optional[str] = Field(None, description="ISO 8601 UTC timestamp of measurement")
    context: Optional[OperationalContext] = Field(None, description="Independent factory operational metrics")
    recentCodReadings: Optional[List[float]] = Field(None, description="Recent chronological historical readings")


class CodAnalysisResponse(BaseModel):
    """Response returned for legacy COD analysis requests."""
    reading_id: Optional[str] = None
    sensor_id: Optional[str] = None
    actual_cod: float
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool
    isolation_forest_flag: Optional[bool] = Field(None, description="Isolation Forest score >= calibrated threshold")
    rule_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Failed physical consistency checks")
    detection_sources: List[str] = Field(default_factory=list, description="Detectors that flagged the reading")
    status: str
    model_version: str
    model_mode: str = "AUTHENTIC"
    data_provenance: str = "AUTHENTIC"
    evaluated_at_utc: str


class TelemetryAnalysisRequest(BaseModel):
    """
    Generalized telemetry analysis request supporting any combination of
    environmental measurements (COD, BOD, pH, TDS, temp, turbidity) and optional context.
    """
    model_config = ConfigDict(extra="allow")

    readingId: Optional[str] = Field(None, description="Unique telemetry record identifier")
    sensorId: Optional[str] = Field(None, description="Sensor or station identifier")
    timestamp: Optional[str] = Field(None, description="ISO 8601 UTC timestamp")
    measurements: Dict[str, float] = Field(
        ...,
        description="Key-value mapping of environmental sensor measurements (e.g., {'cod': 15.0, 'ph': 7.2})"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional industrial plant operational context (e.g., {'productionRate': 900})"
    )


class TelemetryAnalysisResponse(BaseModel):
    """Analytical response for generalized telemetry analysis."""
    reading_id: Optional[str] = None
    sensor_id: Optional[str] = None
    timestamp: Optional[str] = None
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool
    isolation_forest_flag: Optional[bool] = Field(None, description="Isolation Forest score >= calibrated threshold")
    rule_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Failed physical consistency checks")
    detection_sources: List[str] = Field(default_factory=list, description="Detectors that flagged the reading")
    status: str
    model_version: str
    model_mode: str = "AUTHENTIC"
    data_provenance: str = "AUTHENTIC"
    evaluated_features: List[str]
    evaluated_at_utc: str


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "EnviroTrust ML Anomaly Detection Service",
        "version": "1.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/status", status_code=status.HTTP_200_OK, tags=["Model"])
def model_status(
    mode: Optional[str] = Query("authentic", description="Model mode: 'authentic' or 'demo'")
) -> Dict[str, Any]:
    """
    Check model training status and inspect active schema & threshold.
    Pass ?mode=demo to inspect the isolated synthetic demonstration model.
    """
    active_pred = get_active_predictor(mode)
    return active_pred.get_status()


@app.get("/demo/model/status", status_code=status.HTTP_200_OK, tags=["Demonstration"])
def demo_model_status() -> Dict[str, Any]:
    """Inspect status of the isolated synthetic demonstration model."""
    active_pred = get_active_predictor(mode="demo")
    return active_pred.get_status()


@app.post(
    "/analyze/telemetry",
    response_model=TelemetryAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
def analyze_telemetry(
    request: TelemetryAnalysisRequest,
    mode: Optional[str] = Query("authentic", description="Model mode: 'authentic' or 'demo'"),
) -> TelemetryAnalysisResponse:
    """
    Generalized telemetry cross-checking endpoint.
    Evaluates incoming measurements against the model's exact training schema.
    By default evaluates against authentic model (returning HTTP 503 if untrained).
    Pass ?mode=demo to evaluate against the isolated synthetic demonstration model.
    """
    active_pred = get_active_predictor(mode)

    if not active_pred.is_trained():
        msg = (
            "Synthetic demonstration model is not trained."
            if active_pred.model_mode == "SYNTHETIC_DEMO"
            else "Model is not trained. Awaiting authentic industrial operational dataset."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=msg,
        )

    try:
        record_dict = request.model_dump()
        result = active_pred.predict_telemetry(record_dict)

        return TelemetryAnalysisResponse(
            reading_id=result.get("reading_id"),
            sensor_id=result.get("sensor_id"),
            timestamp=result.get("timestamp"),
            anomaly_score=result["anomaly_score"],
            anomaly_threshold=result["anomaly_threshold"],
            is_anomaly=result["is_anomaly"],
            isolation_forest_flag=result.get("isolation_forest_flag"),
            rule_violations=result.get("rule_violations", []),
            detection_sources=result.get("detection_sources", []),
            status=result["status"],
            model_version=result["model_version"],
            model_mode=result.get("model_mode", active_pred.model_mode),
            data_provenance=result.get("data_provenance", active_pred.data_provenance),
            evaluated_features=result.get("evaluated_features", []),
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    except MissingFeatureError as e:
        logger.warning(f"Inference rejected due to missing required schema features: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ModelNotTrainedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post(
    "/demo/analyze/telemetry",
    response_model=TelemetryAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Demonstration"],
)
def demo_analyze_telemetry(request: TelemetryAnalysisRequest) -> TelemetryAnalysisResponse:
    """Dedicated endpoint for synthetic demonstration telemetry analysis."""
    return analyze_telemetry(request=request, mode="demo")


@app.post(
    "/analyze/cod",
    response_model=CodAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
)
def analyze_cod(
    request: CodAnalysisRequest,
    mode: Optional[str] = Query("authentic", description="Model mode: 'authentic' or 'demo'"),
) -> CodAnalysisResponse:
    """
    Legacy endpoint for COD process-consistency cross-checking.
    Maintained for backward compatibility.
    Pass ?mode=demo to evaluate against synthetic demonstration model.
    """
    active_pred = get_active_predictor(mode)

    if not active_pred.is_trained():
        msg = (
            "Synthetic demonstration model is not trained."
            if active_pred.model_mode == "SYNTHETIC_DEMO"
            else "Model is not trained. Awaiting authentic industrial operational dataset."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=msg,
        )

    try:
        record_dict = request.model_dump()
        result = active_pred.predict_record(record_dict)

        return CodAnalysisResponse(
            reading_id=result.get("reading_id"),
            sensor_id=result.get("sensor_id"),
            actual_cod=result.get("actual_cod", request.actualCod),
            anomaly_score=result["anomaly_score"],
            anomaly_threshold=result["anomaly_threshold"],
            is_anomaly=result["is_anomaly"],
            isolation_forest_flag=result.get("isolation_forest_flag"),
            rule_violations=result.get("rule_violations", []),
            detection_sources=result.get("detection_sources", []),
            status=result["status"],
            model_version=result["model_version"],
            model_mode=result.get("model_mode", active_pred.model_mode),
            data_provenance=result.get("data_provenance", active_pred.data_provenance),
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    except MissingFeatureError as e:
        logger.warning(f"Inference rejected: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ModelNotTrainedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post(
    "/demo/analyze/cod",
    response_model=CodAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Demonstration"],
)
def demo_analyze_cod(request: CodAnalysisRequest) -> CodAnalysisResponse:
    """Dedicated endpoint for synthetic demonstration COD analysis."""
    return analyze_cod(request=request, mode="demo")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
