"""
Unit and Integration Tests for EnviroTrust Physical Sensor Tamper Alarm Module.

Verifies:
  1. Hardware event types: ENCLOSURE_OPEN, SEAL_BROKEN, SENSOR_DISCONNECTED, SENSOR_REMOVED, DEVICE_OFFLINE.
  2. Pydantic validation (whitespace device_id, negative sequence, future timestamp, extra fields, invalid types).
  3. Deduplication (duplicate event_id and duplicate (device_id, event_type, sequence_number)).
  4. Alert state transitions (ACTIVE -> CLEARED, preserving full history in audit ledger).
  5. Distinction between confirmed physical tamper and DEVICE_OFFLINE connectivity warning.
  6. Provenance stamping (source=SIMULATED) on all demonstration events/alerts.
  7. SHA-256 cryptographic hash-chained audit ledger integrity and corruption detection.
  8. FastAPI REST endpoints integration (/tamper/events, /tamper/devices/{id}, /tamper/alerts, /tamper/audit-log, /demo/tamper/simulate).
  9. Backend authentication enforcement on all tamper endpoints.
"""

from datetime import datetime, timezone, timedelta
import os
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tamper_alarm import (
    TamperAlarmManager,
    TamperEventSimulator,
    TamperEventType,
    EventState,
    EventSource,
    AlertSeverity,
    AlertCategory,
    DeviceTamperStatus,
    TamperEvent,
    TamperAlert,
    DeviceStatus,
    AuditLogEntry,
    tamper_manager,
)
from main import app


@pytest.fixture(autouse=True)
def reset_tamper_manager():
    """Reset the global tamper manager state before each test."""
    tamper_manager.clear_state()
    yield
    tamper_manager.clear_state()


# ==============================================================================
# 1. Pydantic Validation Tests
# ==============================================================================

def test_tamper_event_valid_instantiation():
    """Valid TamperEvent creates without error."""
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-001",
        device_id="DEV-FAC-01",
        sensor_id="PROBE-COD-1",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc,
        sequence_number=1,
        event_state=EventState.ACTIVE,
        source=EventSource.HARDWARE,
        metadata={"switch_pin": "GPIO_17"},
    )
    assert event.event_id == "EVT-001"
    assert event.device_id == "DEV-FAC-01"
    assert event.event_type == TamperEventType.ENCLOSURE_OPEN
    assert event.sequence_number == 1
    assert event.source == EventSource.HARDWARE


def test_tamper_event_rejects_empty_whitespace_fields():
    """device_id and event_id cannot be blank or whitespace-only."""
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError) as exc:
        TamperEvent(
            event_id="   ",
            device_id="DEV-001",
            event_type=TamperEventType.ENCLOSURE_OPEN,
            event_timestamp=now_utc,
            sequence_number=1,
        )
    assert "event_id" in str(exc.value)

    with pytest.raises(ValidationError) as exc2:
        TamperEvent(
            event_id="EVT-001",
            device_id="   ",
            event_type=TamperEventType.ENCLOSURE_OPEN,
            event_timestamp=now_utc,
            sequence_number=1,
        )
    assert "device_id" in str(exc2.value)


def test_tamper_event_rejects_negative_sequence():
    """Sequence number must be non-negative."""
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError) as exc:
        TamperEvent(
            event_id="EVT-001",
            device_id="DEV-001",
            event_type=TamperEventType.SEAL_BROKEN,
            event_timestamp=now_utc,
            sequence_number=-1,
        )
    assert "sequence_number" in str(exc.value)


def test_tamper_event_rejects_future_timestamp():
    """Timestamps beyond 5 minutes in the future are rejected (preventing forward time travel)."""
    future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    with pytest.raises(ValidationError) as exc:
        TamperEvent(
            event_id="EVT-001",
            device_id="DEV-001",
            event_type=TamperEventType.SENSOR_DISCONNECTED,
            event_timestamp=future_time,
            sequence_number=1,
        )
    assert "future" in str(exc.value)


def test_tamper_event_forbids_extra_fields():
    """Extra undeclared fields are strictly forbidden."""
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        TamperEvent(
            event_id="EVT-001",
            device_id="DEV-001",
            event_type=TamperEventType.SENSOR_REMOVED,
            event_timestamp=now_utc,
            sequence_number=1,
            unauthorized_field="malicious_injection",  # type: ignore
        )


def test_tamper_event_rejects_invalid_event_type():
    """Arbitrary string event types are rejected."""
    now_utc = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        TamperEvent(
            event_id="EVT-001",
            device_id="DEV-001",
            event_type="UNAUTHORIZED_TYPE",  # type: ignore
            event_timestamp=now_utc,
            sequence_number=1,
        )


# ==============================================================================
# 2. Hardware Event Types & Severity Mapping Tests
# ==============================================================================

def test_enclosure_open_triggers_critical_alert():
    """ENCLOSURE_OPEN triggers a CRITICAL, confirmed physical tamper alert."""
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-ENC-01",
        device_id="DEV-100",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc,
        sequence_number=1,
    )
    alert, is_new, msg = tamper_manager.process_event(event)
    assert is_new is True
    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.category == AlertCategory.PHYSICAL_TAMPER
    assert alert.is_confirmed_tamper is True
    assert alert.status == "ACTIVE"

    dev = tamper_manager.get_device_status("DEV-100")
    assert dev.tamper_status == DeviceTamperStatus.TAMPER_ACTIVE
    assert dev.active_alerts_count == 1
    assert "ENCLOSURE_OPEN" in dev.active_event_types


def test_seal_broken_triggers_critical_alert():
    """SEAL_BROKEN triggers a CRITICAL, confirmed physical tamper alert."""
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-SEAL-01",
        device_id="DEV-100",
        event_type=TamperEventType.SEAL_BROKEN,
        event_timestamp=now_utc,
        sequence_number=2,
    )
    alert, is_new, msg = tamper_manager.process_event(event)
    assert is_new is True
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.category == AlertCategory.PHYSICAL_TAMPER
    assert alert.is_confirmed_tamper is True


def test_sensor_removed_triggers_critical_alert():
    """SENSOR_REMOVED triggers a CRITICAL, confirmed physical tamper alert."""
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-REM-01",
        device_id="DEV-100",
        sensor_id="PROBE-BOD-1",
        event_type=TamperEventType.SENSOR_REMOVED,
        event_timestamp=now_utc,
        sequence_number=3,
    )
    alert, is_new, msg = tamper_manager.process_event(event)
    assert is_new is True
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.category == AlertCategory.PHYSICAL_TAMPER
    assert alert.is_confirmed_tamper is True


def test_sensor_disconnected_triggers_high_alert():
    """SENSOR_DISCONNECTED triggers a HIGH severity physical tamper alert."""
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-DISC-01",
        device_id="DEV-100",
        sensor_id="PROBE-TDS-1",
        event_type=TamperEventType.SENSOR_DISCONNECTED,
        event_timestamp=now_utc,
        sequence_number=4,
    )
    alert, is_new, msg = tamper_manager.process_event(event)
    assert is_new is True
    assert alert.severity == AlertSeverity.HIGH
    assert alert.category == AlertCategory.PHYSICAL_TAMPER
    assert alert.is_confirmed_tamper is True


def test_device_offline_treated_as_connectivity_warning():
    """
    DEVICE_OFFLINE is strictly categorized as CONNECTIVITY_WARNING with
    is_confirmed_tamper=False and MEDIUM severity.
    """
    now_utc = datetime.now(timezone.utc)
    event = TamperEvent(
        event_id="EVT-OFF-01",
        device_id="DEV-200",
        event_type=TamperEventType.DEVICE_OFFLINE,
        event_timestamp=now_utc,
        sequence_number=1,
    )
    alert, is_new, msg = tamper_manager.process_event(event)
    assert is_new is True
    assert alert.severity == AlertSeverity.MEDIUM
    assert alert.category == AlertCategory.CONNECTIVITY_WARNING
    assert alert.is_confirmed_tamper is False

    dev = tamper_manager.get_device_status("DEV-200")
    assert dev.tamper_status == DeviceTamperStatus.OFFLINE_WARNING
    assert "connectivity warning" in alert.details.lower()


def test_device_status_precedence_physical_over_offline():
    """
    If a device has both a physical tamper event and an offline event,
    the physical tamper status takes precedence (TAMPER_ACTIVE).
    """
    now_utc = datetime.now(timezone.utc)
    e1 = TamperEvent(
        event_id="EVT-OFF-01",
        device_id="DEV-300",
        event_type=TamperEventType.DEVICE_OFFLINE,
        event_timestamp=now_utc,
        sequence_number=1,
    )
    tamper_manager.process_event(e1)
    dev = tamper_manager.get_device_status("DEV-300")
    assert dev.tamper_status == DeviceTamperStatus.OFFLINE_WARNING

    e2 = TamperEvent(
        event_id="EVT-ENC-01",
        device_id="DEV-300",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc,
        sequence_number=2,
    )
    tamper_manager.process_event(e2)
    dev2 = tamper_manager.get_device_status("DEV-300")
    assert dev2.tamper_status == DeviceTamperStatus.TAMPER_ACTIVE
    assert dev2.active_alerts_count == 2


# ==============================================================================
# 3. Deduplication Tests
# ==============================================================================

def test_deduplication_by_event_id():
    """Resubmitting the same event_id is rejected without creating a duplicate alert."""
    now_utc = datetime.now(timezone.utc)
    event1 = TamperEvent(
        event_id="EVT-DUP-001",
        device_id="DEV-DUP",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc,
        sequence_number=1,
    )
    alert1, is_new1, msg1 = tamper_manager.process_event(event1)
    assert is_new1 is True

    # Re-submit identical event
    alert2, is_new2, msg2 = tamper_manager.process_event(event1)
    assert is_new2 is False
    assert "Duplicate" in msg2
    assert alert2.alert_id == alert1.alert_id

    # Total alerts count for device remains 1
    alerts = tamper_manager.get_alerts(device_id="DEV-DUP")
    assert len(alerts) == 1

    # Check audit log recorded the duplicate rejection
    audit_entries = tamper_manager.get_audit_log(device_id="DEV-DUP")
    actions = [e.action for e in audit_entries]
    assert "DUPLICATE_IGNORED" in actions


def test_deduplication_by_sequence_key():
    """Resubmitting different event_id with same device, type, and sequence_number is rejected."""
    now_utc = datetime.now(timezone.utc)
    e1 = TamperEvent(
        event_id="EVT-A",
        device_id="DEV-DUP-2",
        event_type=TamperEventType.SEAL_BROKEN,
        event_timestamp=now_utc,
        sequence_number=42,
    )
    tamper_manager.process_event(e1)

    e2 = TamperEvent(
        event_id="EVT-B",  # different event_id
        device_id="DEV-DUP-2",
        event_type=TamperEventType.SEAL_BROKEN,
        event_timestamp=now_utc,
        sequence_number=42,  # same sequence number
    )
    alert, is_new, msg = tamper_manager.process_event(e2)
    assert is_new is False
    assert "Duplicate" in msg

    alerts = tamper_manager.get_alerts(device_id="DEV-DUP-2")
    assert len(alerts) == 1


# ==============================================================================
# 4. State Transitions (ACTIVE -> CLEARED) & History Preservation
# ==============================================================================

def test_cleared_event_preserves_historical_records():
    """
    When an active tamper condition is marked CLEARED:
      - The alert status transitions to CLEARED.
      - The device status re-evaluates to NOMINAL.
      - The historical alert is PRESERVED, not deleted.
      - Audit ledger records an ALERT_CLEARED entry.
    """
    now_utc = datetime.now(timezone.utc)
    active_evt = TamperEvent(
        event_id="EVT-ACT-01",
        device_id="DEV-CLR-1",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc,
        sequence_number=1,
        event_state=EventState.ACTIVE,
    )
    alert_act, is_new, _ = tamper_manager.process_event(active_evt)
    assert alert_act.status == "ACTIVE"

    clear_evt = TamperEvent(
        event_id="EVT-CLR-01",
        device_id="DEV-CLR-1",
        event_type=TamperEventType.ENCLOSURE_OPEN,
        event_timestamp=now_utc + timedelta(seconds=10),
        sequence_number=2,
        event_state=EventState.CLEARED,
        metadata={"operator": "TECH-01", "note": "Enclosure re-locked and inspected."},
    )
    alert_clr, is_new_clr, msg_clr = tamper_manager.process_event(clear_evt)
    assert is_new_clr is True
    assert alert_clr.status == "CLEARED"
    assert alert_clr.cleared_at_utc is not None
    assert "preserved" in msg_clr.lower()

    # Device status should now be NOMINAL
    dev = tamper_manager.get_device_status("DEV-CLR-1")
    assert dev.tamper_status == DeviceTamperStatus.NOMINAL
    assert dev.active_alerts_count == 0
    assert dev.total_incidents_recorded == 1  # Total incidents still recorded

    # Historical alerts query still finds the cleared alert
    all_alerts = tamper_manager.get_alerts(device_id="DEV-CLR-1")
    assert len(all_alerts) == 1
    assert all_alerts[0].status == "CLEARED"

    # Filter by ACTIVE returns 0, filter by CLEARED returns 1
    active_list = tamper_manager.get_alerts(device_id="DEV-CLR-1", status_filter="ACTIVE")
    cleared_list = tamper_manager.get_alerts(device_id="DEV-CLR-1", status_filter="CLEARED")
    assert len(active_list) == 0
    assert len(cleared_list) == 1


# ==============================================================================
# 5. Cryptographic Hash Chain Audit Ledger Tests
# ==============================================================================

def test_audit_log_hash_chain_validity():
    """Audit log maintains cryptographic SHA-256 chain across multiple events."""
    now_utc = datetime.now(timezone.utc)
    for i in range(5):
        evt = TamperEvent(
            event_id=f"EVT-CHAIN-{i}",
            device_id="DEV-CHAIN",
            event_type=TamperEventType.SEAL_BROKEN,
            event_timestamp=now_utc + timedelta(seconds=i),
            sequence_number=i + 1,
            event_state=EventState.ACTIVE if i % 2 == 0 else EventState.CLEARED,
        )
        tamper_manager.process_event(evt)

    is_valid, err = tamper_manager.verify_audit_log_integrity()
    assert is_valid is True
    assert err is None


def test_audit_log_detects_tampered_entry():
    """Modifying any entry in the audit ledger breaks cryptographic chain validation."""
    now_utc = datetime.now(timezone.utc)
    for i in range(3):
        evt = TamperEvent(
            event_id=f"EVT-TAMPER-LEDGER-{i}",
            device_id="DEV-CORRUPT",
            event_type=TamperEventType.ENCLOSURE_OPEN,
            event_timestamp=now_utc + timedelta(seconds=i),
            sequence_number=i + 1,
        )
        tamper_manager.process_event(evt)

    # Illegitimately corrupt an in-memory audit log entry
    tamper_manager._audit_log[1].details = "FORGED AND ALTERED DETAILS"

    is_valid, err = tamper_manager.verify_audit_log_integrity()
    assert is_valid is False
    assert "Hash mismatch" in err or "Broken chain" in err


# ==============================================================================
# 6. Hardware Simulation Helper Tests
# ==============================================================================

def test_tamper_simulator_scenarios():
    """All 6 simulator scenarios generate properly structured, simulated events."""
    scenarios = [
        ("enclosure_open", TamperEventType.ENCLOSURE_OPEN, EventState.ACTIVE),
        ("seal_broken", TamperEventType.SEAL_BROKEN, EventState.ACTIVE),
        ("sensor_disconnected", TamperEventType.SENSOR_DISCONNECTED, EventState.ACTIVE),
        ("sensor_removed", TamperEventType.SENSOR_REMOVED, EventState.ACTIVE),
        ("device_offline", TamperEventType.DEVICE_OFFLINE, EventState.ACTIVE),
        ("clear", TamperEventType.ENCLOSURE_OPEN, EventState.CLEARED),
    ]

    for sc_name, expected_type, expected_state in scenarios:
        evt = TamperEventSimulator.create_simulated_event(
            scenario=sc_name,
            device_id="DEV-SIM-TEST",
        )
        assert evt.source == EventSource.SIMULATED
        assert evt.event_type == expected_type
        assert evt.event_state == expected_state
        assert evt.metadata.get("simulation_demonstration") is True

        alert, _, _ = tamper_manager.process_event(evt)
        if expected_state == EventState.ACTIVE:
            assert alert.source == EventSource.SIMULATED
            assert alert.is_simulated is True
            assert "[SIMULATED]" in alert.summary


def test_tamper_simulator_rejects_unknown_scenario():
    """Unknown simulator scenarios raise ValueError."""
    with pytest.raises(ValueError) as exc:
        TamperEventSimulator.create_simulated_event(
            scenario="unknown_invalid_scenario",
            device_id="DEV-SIM",
        )
    assert "Unknown simulation scenario" in str(exc.value)


# ==============================================================================
# 7. FastAPI REST Endpoints Integration Tests
# ==============================================================================

def test_rest_post_tamper_event():
    """POST /tamper/events processes incoming event and returns alert."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "event_id": "EVT-REST-01",
        "device_id": "DEV-REST-1",
        "sensor_id": "PROBE-COD-1",
        "event_type": "ENCLOSURE_OPEN",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
        "metadata": {"location": "Effluent Basin A"},
    }

    response = client.post("/tamper/events", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_new_alert"] is True
    assert data["alert"]["severity"] == "CRITICAL"
    assert data["alert"]["is_confirmed_tamper"] is True


def test_rest_get_device_status():
    """GET /tamper/devices/{device_id} returns device security status."""
    client = TestClient(app)
    # 404 for unrecorded device
    res_404 = client.get("/tamper/devices/NON_EXISTENT_DEV")
    assert res_404.status_code == 404

    # Post an event
    now_utc = datetime.now(timezone.utc).isoformat()
    client.post("/tamper/events", json={
        "event_id": "EVT-REST-02",
        "device_id": "DEV-REST-2",
        "event_type": "SEAL_BROKEN",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    })

    response = client.get("/tamper/devices/DEV-REST-2")
    assert response.status_code == 200
    data = response.json()
    assert data["device_id"] == "DEV-REST-2"
    assert data["tamper_status"] == "TAMPER_ACTIVE"
    assert data["active_alerts_count"] == 1


def test_rest_get_tamper_alerts_with_filters():
    """GET /tamper/alerts queries alerts with status, severity, and device filtering."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()

    # Ingest 2 events
    client.post("/tamper/events", json={
        "event_id": "EVT-ALERTS-1",
        "device_id": "DEV-FLT-1",
        "event_type": "ENCLOSURE_OPEN",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    })
    client.post("/tamper/events", json={
        "event_id": "EVT-ALERTS-2",
        "device_id": "DEV-FLT-2",
        "event_type": "DEVICE_OFFLINE",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    })

    # Query all
    all_res = client.get("/tamper/alerts")
    assert all_res.status_code == 200
    assert len(all_res.json()) >= 2

    # Filter by severity CRITICAL
    crit_res = client.get("/tamper/alerts?severity=CRITICAL")
    assert all(a["severity"] == "CRITICAL" for a in crit_res.json())

    # Filter by device DEV-FLT-2
    dev2_res = client.get("/tamper/alerts?device_id=DEV-FLT-2")
    assert len(dev2_res.json()) == 1
    assert dev2_res.json()[0]["event_type"] == "DEVICE_OFFLINE"


def test_rest_get_single_alert_by_id():
    """GET /tamper/alerts/{alert_id} returns the specific alert or 404."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()

    post_res = client.post("/tamper/events", json={
        "event_id": "EVT-SINGLE-01",
        "device_id": "DEV-SGL-1",
        "event_type": "SENSOR_REMOVED",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    })
    alert_id = post_res.json()["alert"]["alert_id"]

    res = client.get(f"/tamper/alerts/{alert_id}")
    assert res.status_code == 200
    assert res.json()["alert_id"] == alert_id

    res_404 = client.get("/tamper/alerts/NON_EXISTENT_ALERT")
    assert res_404.status_code == 404


def test_rest_get_audit_log():
    """GET /tamper/audit-log returns the cryptographically verified ledger."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()

    client.post("/tamper/events", json={
        "event_id": "EVT-AUD-01",
        "device_id": "DEV-AUD-1",
        "event_type": "SEAL_BROKEN",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    })

    res = client.get("/tamper/audit-log")
    assert res.status_code == 200
    data = res.json()
    assert data["is_chain_valid"] is True
    assert data["total_returned"] >= 1
    assert len(data["entries"]) >= 1


def test_rest_demo_simulate_endpoint():
    """POST /demo/tamper/simulate triggers demonstration event and returns alert."""
    client = TestClient(app)
    req = {
        "scenario": "enclosure_open",
        "device_id": "DEV-SIM-API",
        "metadata": {"test": "suite"},
    }
    res = client.post("/demo/tamper/simulate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["demonstration_mode"] is True
    assert data["is_new_alert"] is True
    assert data["event"]["source"] == "SIMULATED"
    assert data["alert"]["is_simulated"] is True

    # Bad scenario returns 422
    bad_res = client.post("/demo/tamper/simulate", json={"scenario": "nonexistent"})
    assert bad_res.status_code == 422


# ==============================================================================
# 8. Backend Authentication Tests for Tamper Endpoints
# ==============================================================================

def test_tamper_endpoints_enforce_backend_token(monkeypatch):
    """When ML_BACKEND_SERVICE_TOKEN is set, unauthenticated calls fail with 401."""
    token = "secret-regulator-token-998"
    monkeypatch.setenv("ML_BACKEND_SERVICE_TOKEN", token)
    client = TestClient(app)

    now_utc = datetime.now(timezone.utc).isoformat()
    evt_payload = {
        "event_id": "EVT-SEC-01",
        "device_id": "DEV-SEC-1",
        "event_type": "ENCLOSURE_OPEN",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "event_state": "ACTIVE",
        "source": "HARDWARE",
    }

    # 1. Unauthenticated -> 401
    assert client.post("/tamper/events", json=evt_payload).status_code == 401
    assert client.get("/tamper/devices/DEV-SEC-1").status_code == 401
    assert client.get("/tamper/alerts").status_code == 401
    assert client.get("/tamper/audit-log").status_code == 401
    assert client.post("/demo/tamper/simulate", json={"scenario": "enclosure_open"}).status_code == 401

    # 2. Authenticated with header -> 200
    headers = {"X-ML-Service-Token": token}
    res_post = client.post("/tamper/events", json=evt_payload, headers=headers)
    assert res_post.status_code == 200

    res_dev = client.get("/tamper/devices/DEV-SEC-1", headers=headers)
    assert res_dev.status_code == 200

    res_alerts = client.get("/tamper/alerts", headers=headers)
    assert res_alerts.status_code == 200

    res_audit = client.get("/tamper/audit-log", headers=headers)
    assert res_audit.status_code == 200

    res_sim = client.post("/demo/tamper/simulate", json={"scenario": "enclosure_open"}, headers=headers)
    assert res_sim.status_code == 200
