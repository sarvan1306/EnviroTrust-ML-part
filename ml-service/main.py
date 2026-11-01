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
import hmac
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from anomaly_detection import ModelNotTrainedError
from predict import EnviroTrustPredictor
from preprocessing import MissingFeatureError
from tamper_alarm import (
    AuditLogEntry,
    DeviceStatus,
    TamperAlert,
    TamperEvent,
    TamperEventSimulator,
    TamperSimulationRequest,
    tamper_manager,
)
from store_and_forward import (
    DeliveryStatus,
    GatewayIngestionService,
    GatewayStoreAndForwardSimulator,
    MessageCategory,
    QueueCapacityStatus,
    QueueMessage,
    ServerAckResponse,
    StoreAndForwardQueue,
    gateway_ingestion_service,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnviroTrust.API")

# Predictor instances: authentic production model (untrained) and isolated synthetic demonstration model
predictor: Optional[EnviroTrustPredictor] = None
demo_predictor: Optional[EnviroTrustPredictor] = None

# Backend authentication environment variables (checked in precedence order)
ML_AUTH_ENV_VARS = [
    "ML_SERVICE_TOKEN",
    "ML_BACKEND_SERVICE_TOKEN",
    "ENVIROTRUST_ML_TOKEN",
    "ML_SERVICE_API_KEY",
    "X_ML_SERVICE_TOKEN",
]


def get_configured_auth_token() -> Optional[str]:
    """
    Retrieve configured backend authentication token from environment.
    Returns None if no token is configured (open development mode).
    """
    for var_name in ML_AUTH_ENV_VARS:
        token = os.environ.get(var_name)
        if token and token.strip():
            return token.strip()
    return None


def verify_backend_token(
    x_ml_service_token: Optional[str] = Header(None, alias="X-ML-Service-Token", description="Backend authentication token"),
    authorization: Optional[str] = Header(None, alias="Authorization", description="Bearer token authentication"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="API key header"),
) -> Optional[str]:
    """
    Validate backend authentication credentials against configured environment variable.

    If no token is set in environment (ML_SERVICE_TOKEN / ENVIROTRUST_ML_TOKEN / ML_SERVICE_API_KEY),
    authentication is disabled (open development mode) for backward compatibility and local testing.

    If configured, requires matching token via:
      - Header 'X-ML-Service-Token: <token>'
      - Header 'Authorization: Bearer <token>'
      - Header 'X-API-Key: <token>'

    Uses constant-time comparison to prevent timing attack vulnerabilities.
    """
    expected_token = get_configured_auth_token()
    if not expected_token:
        return None

    # Extract token from available headers
    provided_token = x_ml_service_token or x_api_key
    if not provided_token and authorization:
        if authorization.lower().startswith("bearer "):
            provided_token = authorization[7:].strip()
        else:
            provided_token = authorization.strip()

    if not provided_token:
        logger.warning("Rejected unauthenticated request: missing required authentication header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Pass 'X-ML-Service-Token' header or 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(provided_token.encode("utf-8"), expected_token.encode("utf-8")):
        logger.warning("Rejected unauthenticated request: invalid token provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model artifacts on server startup."""
    global predictor, demo_predictor
    logger.info("Initializing EnviroTrust ML Service...")
    auth_token = get_configured_auth_token()
    if auth_token:
        logger.info("Authentication: ENABLED (backend token verified on incoming requests).")
    else:
        logger.info("Authentication: OPEN mode (no ML_SERVICE_TOKEN set in environment).")

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

    # Wire store-and-forward gateway ingestion service
    gateway_ingestion_service._tamper_engine = tamper_manager
    gateway_ingestion_service._predictor_fn = (
        lambda payload: demo_predictor.predict_telemetry(payload)
        if demo_predictor and demo_predictor.is_trained()
        else (predictor.predict_telemetry(payload) if predictor and predictor.is_trained() else None)
    )

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
    cusum_alarms: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sustained-shift (CUSUM) alarms for this sensor stream"
    )
    detection_sources: List[str] = Field(default_factory=list, description="Detectors that flagged the reading")
    rolling_window_filled: Optional[bool] = Field(
        None, description="Rolling-window (stuck-sensor) features had enough sensor history; null if the model has none"
    )
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
    cusum_alarms: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sustained-shift (CUSUM) alarms for this sensor stream"
    )
    detection_sources: List[str] = Field(default_factory=list, description="Detectors that flagged the reading")
    rolling_window_filled: Optional[bool] = Field(
        None, description="Rolling-window (stuck-sensor) features had enough sensor history; null if the model has none"
    )
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


@app.get("/model/status", status_code=status.HTTP_200_OK, tags=["Model"], dependencies=[Depends(verify_backend_token)])
def model_status(
    mode: Optional[str] = Query("authentic", description="Model mode: 'authentic' or 'demo'")
) -> Dict[str, Any]:
    """
    Check model training status and inspect active schema & threshold.
    Pass ?mode=demo to inspect the isolated synthetic demonstration model.
    """
    active_pred = get_active_predictor(mode)
    return active_pred.get_status()


@app.get("/demo/model/status", status_code=status.HTTP_200_OK, tags=["Demonstration"], dependencies=[Depends(verify_backend_token)])
def demo_model_status() -> Dict[str, Any]:
    """Inspect status of the isolated synthetic demonstration model."""
    active_pred = get_active_predictor(mode="demo")
    return active_pred.get_status()


@app.post(
    "/analyze/telemetry",
    response_model=TelemetryAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    dependencies=[Depends(verify_backend_token)],
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
            cusum_alarms=result.get("cusum_alarms", []),
            detection_sources=result.get("detection_sources", []),
            rolling_window_filled=result.get("rolling_window_filled"),
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
    dependencies=[Depends(verify_backend_token)],
)
def demo_analyze_telemetry(request: TelemetryAnalysisRequest) -> TelemetryAnalysisResponse:
    """Dedicated endpoint for synthetic demonstration telemetry analysis."""
    return analyze_telemetry(request=request, mode="demo")


@app.post(
    "/analyze/cod",
    response_model=CodAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    dependencies=[Depends(verify_backend_token)],
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
            cusum_alarms=result.get("cusum_alarms", []),
            detection_sources=result.get("detection_sources", []),
            rolling_window_filled=result.get("rolling_window_filled"),
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
    dependencies=[Depends(verify_backend_token)],
)
def demo_analyze_cod(request: CodAnalysisRequest) -> CodAnalysisResponse:
    """Dedicated endpoint for synthetic demonstration COD analysis."""
    return analyze_cod(request=request, mode="demo")


# ==============================================================================
# Physical Sensor Tamper Alarm Endpoints (Security Layer)
# ==============================================================================

@app.post(
    "/tamper/events",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    tags=["Security & Tamper Alarm"],
    dependencies=[Depends(verify_backend_token)],
)
def ingest_tamper_event(event: TamperEvent) -> Dict[str, Any]:
    """
    Ingest a verified physical tamper event from hardware IoT controller or Node.js backend.

    Security Guarantee:
      - Marks device status as ACTIVE.
      - Generates high/critical severity alerts immediately.
      - Records cryptographically chained entry in the tamper-evident audit ledger.
      - Prevents duplicate events from generating duplicate alerts.
      - Does NOT affect Isolation Forest ML model weights.
    """
    alert, is_new, message = tamper_manager.process_event(event)
    return {
        "status": "success",
        "message": message,
        "is_new_alert": is_new,
        "event_id": event.event_id,
        "device_id": event.device_id,
        "event_type": event.event_type.value,
        "event_state": event.event_state.value,
        "alert": alert.model_dump() if alert else None,
    }


@app.get(
    "/tamper/devices/{device_id}",
    response_model=DeviceStatus,
    status_code=status.HTTP_200_OK,
    tags=["Security & Tamper Alarm"],
    dependencies=[Depends(verify_backend_token)],
)
def get_device_tamper_status(device_id: str) -> DeviceStatus:
    """
    Inspect real-time physical security status and active alerts for an industrial monitoring unit.
    """
    dev_status = tamper_manager.get_device_status(device_id)
    if not dev_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device '{device_id}' has not been registered or reported events.",
        )
    return dev_status


@app.get(
    "/tamper/alerts",
    response_model=List[TamperAlert],
    status_code=status.HTTP_200_OK,
    tags=["Security & Tamper Alarm"],
    dependencies=[Depends(verify_backend_token)],
)
def list_tamper_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status: 'ACTIVE' or 'CLEARED'"),
    severity: Optional[str] = Query(None, description="Filter by severity: 'CRITICAL', 'HIGH', 'MEDIUM'"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    source: Optional[str] = Query(None, description="Filter by source: 'HARDWARE' or 'SIMULATED'"),
) -> List[TamperAlert]:
    """
    Regulator alert feed querying active and historical physical security alerts.
    Historical alerts are preserved even when the physical tamper condition is cleared.
    """
    return tamper_manager.get_alerts(
        status_filter=status,
        severity_filter=severity,
        device_id=device_id,
        source_filter=source,
    )


@app.get(
    "/tamper/alerts/{alert_id}",
    response_model=TamperAlert,
    status_code=status.HTTP_200_OK,
    tags=["Security & Tamper Alarm"],
    dependencies=[Depends(verify_backend_token)],
)
def get_tamper_alert_by_id(alert_id: str) -> TamperAlert:
    """
    Inspect detailed regulatory incident record for a specific alert.
    """
    alerts = tamper_manager.get_alerts()
    for a in alerts:
        if a.alert_id == alert_id:
            return a
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tamper alert '{alert_id}' not found.",
    )


@app.get(
    "/tamper/audit-log",
    status_code=status.HTTP_200_OK,
    tags=["Security & Tamper Alarm"],
    dependencies=[Depends(verify_backend_token)],
)
def get_tamper_audit_log(
    device_id: Optional[str] = Query(None, description="Filter audit log by device ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max audit entries to return"),
) -> Dict[str, Any]:
    """
    Query the cryptographically chained tamper-evident audit ledger for regulatory compliance.
    """
    is_valid, error_msg = tamper_manager.verify_audit_log_integrity()
    entries = tamper_manager.get_audit_log(device_id=device_id, limit=limit)
    return {
        "is_chain_valid": is_valid,
        "integrity_error": error_msg,
        "total_returned": len(entries),
        "entries": [e.model_dump() for e in entries],
    }


@app.post(
    "/demo/tamper/simulate",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    tags=["Demonstration"],
    dependencies=[Depends(verify_backend_token)],
)
def simulate_tamper_scenario(request: TamperSimulationRequest) -> Dict[str, Any]:
    """
    Demonstration mechanism to trigger simulated physical tamper events.
    Supports: 'enclosure_open', 'seal_broken', 'sensor_disconnected', 'sensor_removed', 'device_offline', 'clear'.
    Never represents simulated events as genuine hardware detections (stamped with source=SIMULATED).
    """
    try:
        sim_event = TamperEventSimulator.create_simulated_event(
            scenario=request.scenario,
            device_id=request.device_id,
            sensor_id=request.sensor_id,
            metadata=request.metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    alert, is_new, message = tamper_manager.process_event(sim_event)
    return {
        "status": "success",
        "demonstration_mode": True,
        "message": message,
        "is_new_alert": is_new,
        "event": sim_event.model_dump(),
        "alert": alert.model_dump() if alert else None,
    }


# ==============================================================================
# Edge Gateway Store-and-Forward Ingestion Endpoints
# ==============================================================================

@app.post(
    "/gateway/ingest",
    response_model=ServerAckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Store-and-Forward Gateway"],
    dependencies=[Depends(verify_backend_token)],
)
def ingest_gateway_message(message: QueueMessage) -> ServerAckResponse:
    """
    Ingest a store-and-forward message from edge gateway idempotently.
    Acknowledges receipt with unique receipt token.
    Duplicate messages return status DUPLICATE_ACKNOWLEDGED without duplicate alerts.
    """
    return gateway_ingestion_service.ingest_message(message)


@app.post(
    "/gateway/ingest/batch",
    response_model=List[ServerAckResponse],
    status_code=status.HTTP_200_OK,
    tags=["Store-and-Forward Gateway"],
    dependencies=[Depends(verify_backend_token)],
)
def ingest_gateway_batch(messages: List[QueueMessage]) -> List[ServerAckResponse]:
    """
    Batch ingestion endpoint for store-and-forward edge gateways.
    Iteratively processes each message idempotently and returns individual acknowledgements.
    """
    return [gateway_ingestion_service.ingest_message(msg) for msg in messages]


@app.get(
    "/gateway/status",
    status_code=status.HTTP_200_OK,
    tags=["Store-and-Forward Gateway"],
    dependencies=[Depends(verify_backend_token)],
)
def get_gateway_status() -> Dict[str, Any]:
    """
    Query server-side gateway ingestion status and total recorded message count.
    """
    return {
        "service": "EnviroTrust Store-and-Forward Ingestion Service",
        "is_online": True,
        "total_recorded_messages": len(gateway_ingestion_service._processed_messages),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/demo/gateway/simulate",
    status_code=status.HTTP_200_OK,
    tags=["Demonstration"],
    dependencies=[Depends(verify_backend_token)],
)
def simulate_gateway_scenarios(
    scenario: Optional[str] = Query("ALL", description="Scenario: 'A', 'B', 'C', 'D', 'E', 'F', or 'ALL'")
) -> Dict[str, Any]:
    """
    Execute edge gateway store-and-forward simulation scenarios.
    Demonstrates offline buffering, tamper event prioritization, network restoration,
    duplicate delivery handling, and crash recovery.
    """
    sim_db = f"sim_demo_{uuid4().hex[:8]}.db"
    simulator = GatewayStoreAndForwardSimulator(db_path=sim_db)
    try:
        sc_upper = scenario.upper().strip() if scenario else "ALL"
        if sc_upper == "A":
            results = simulator.run_scenario_a_normal_operation()
        elif sc_upper == "B":
            results = simulator.run_scenario_b_network_outage(5)
        elif sc_upper == "C":
            results = simulator.run_scenario_c_tamper_during_outage()
        elif sc_upper == "D":
            results = simulator.run_scenario_d_network_restoration()
        elif sc_upper == "E":
            results = simulator.run_scenario_e_duplicate_delivery()
        elif sc_upper == "F":
            results = simulator.run_scenario_f_application_restart_recovery()
        else:
            results = simulator.run_all_scenarios()

        return {
            "status": "success",
            "demonstration_mode": True,
            "scenario": sc_upper,
            "results": results,
        }
    finally:
        for suffix in ["", "-wal", "-shm"]:
            fpath = f"{sim_db}{suffix}"
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
