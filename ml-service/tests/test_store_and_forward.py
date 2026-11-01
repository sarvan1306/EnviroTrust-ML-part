"""
Comprehensive Unit & Integration Tests for EnviroTrust Store-and-Forward Edge Gateway.

Tests cover:
  1. SQLite persistence and schema validation.
  2. Application restart recovery and in-flight reset.
  3. Network outage buffering (no message loss).
  4. Network reconnection and automatic retry.
  5. Priority transmission (Physical Tamper > Health > Telemetry).
  6. Anti-starvation mechanism for routine telemetry.
  7. Exponential backoff with randomized jitter.
  8. Explicit server acknowledgement and lost ACK recovery.
  9. Idempotent server ingestion & duplicate delivery prevention.
  10. Separate monotonic sequence counters for telemetry vs security.
  11. Storage capacity monitoring (75% warning, 90% reserved security, 100% exhaustion).
  12. Telemetry retention policies (DROP_OLDEST_TELEMETRY with audit log vs REJECT_NEW).
  13. Local payload checksum validation and disk corruption detection.
  14. Complete 6-scenario simulator execution (A through F).
  15. FastAPI REST API endpoints (/gateway/ingest, /gateway/status, /demo/gateway/simulate).
  16. Backend authentication token enforcement on gateway endpoints.
"""

from datetime import datetime, timezone, timedelta
import os
import random
import tempfile
import pytest
from fastapi.testclient import TestClient

from store_and_forward import (
    CorruptedPayloadError,
    DeliveryStatus,
    EdgeGatewaySyncAgent,
    ExponentialBackoffWithJitter,
    GatewayIngestionService,
    GatewayStoreAndForwardSimulator,
    MessageCategory,
    QueueCapacityStatus,
    QueueMessage,
    ServerAckResponse,
    StorageExhaustedError,
    StorageWarningState,
    StoreAndForwardQueue,
    TelemetryRetentionPolicy,
    TransmissionPriority,
    gateway_ingestion_service,
)
from tamper_alarm import tamper_manager, TamperEventType
from main import app


@pytest.fixture
def temp_db_path():
    """Provides a temporary SQLite database file path cleaned up after test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset server-side ingestion state and tamper manager before each test."""
    gateway_ingestion_service.clear()
    tamper_manager.clear_state()
    yield
    gateway_ingestion_service.clear()
    tamper_manager.clear_state()


# ==============================================================================
# 1. SQLite Persistence & Schema Validation
# ==============================================================================

def test_sqlite_persistence_and_fields(temp_db_path):
    """Enqueued messages are durably committed with all required schema fields."""
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    now_utc = datetime.now(timezone.utc)

    msg = queue.enqueue(
        device_id="DEV-FIELD-01",
        category=MessageCategory.TELEMETRY,
        message_type="COD_READING",
        payload={"COD_mg_L": 480.0, "pH": 7.4},
        event_timestamp=now_utc,
    )

    assert msg.message_id.startswith("MSG-")
    assert msg.device_id == "DEV-FIELD-01"
    assert msg.message_category == MessageCategory.TELEMETRY
    assert msg.message_type == "COD_READING"
    assert msg.sequence_number == 1
    assert msg.priority == TransmissionPriority.ROUTINE_TELEMETRY.value
    assert msg.delivery_status == DeliveryStatus.PENDING
    assert msg.payload_checksum is not None

    # Fetch from SQLite
    fetched = queue.get_message(msg.message_id)
    assert fetched is not None
    assert fetched.message_id == msg.message_id
    assert fetched.payload == {"COD_mg_L": 480.0, "pH": 7.4}
    assert fetched.payload_checksum == msg.payload_checksum


def test_enqueue_validates_device_id(temp_db_path):
    """Empty or whitespace device_id raises ValueError."""
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    with pytest.raises(ValueError):
        queue.enqueue(
            device_id="   ",
            category=MessageCategory.TELEMETRY,
            message_type="TEST",
            payload={"x": 1},
        )


# ==============================================================================
# 2. Application Restart Recovery
# ==============================================================================

def test_application_restart_recovers_unacknowledged_messages(temp_db_path):
    """Messages survive application termination and are recovered on restart."""
    # 1. First session
    q1 = StoreAndForwardQueue(db_path=temp_db_path)
    msg1 = q1.enqueue(
        device_id="DEV-RESTART-01",
        category=MessageCategory.TELEMETRY,
        message_type="T1",
        payload={"val": 100},
    )
    msg2 = q1.enqueue(
        device_id="DEV-RESTART-01",
        category=MessageCategory.SECURITY,
        message_type="ENCLOSURE_OPEN",
        payload={"switch": "OPEN"},
    )
    # Mark msg1 in flight
    q1.mark_in_flight([msg1.message_id])

    # Terminate q1
    del q1

    # 2. Second session (simulated restart)
    q2 = StoreAndForwardQueue(db_path=temp_db_path)
    q2.reset_in_flight_to_pending()

    # msg1 should have reverted from IN_FLIGHT to PENDING
    rec_msg1 = q2.get_message(msg1.message_id)
    assert rec_msg1.delivery_status == DeliveryStatus.PENDING

    # msg2 should still be PENDING
    rec_msg2 = q2.get_message(msg2.message_id)
    assert rec_msg2.delivery_status == DeliveryStatus.PENDING

    # Sequence numbers continue monotonically from previous session
    msg3 = q2.enqueue(
        device_id="DEV-RESTART-01",
        category=MessageCategory.TELEMETRY,
        message_type="T2",
        payload={"val": 200},
    )
    assert msg3.sequence_number == 2  # Telemetry seq 1 was msg1, so next is 2!


# ==============================================================================
# 3. Separate Monotonic Sequence Counters
# ==============================================================================

def test_separate_sequence_counters_for_telemetry_and_security(temp_db_path):
    """Telemetry and Security maintain completely isolated sequence counters."""
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    dev_id = "DEV-SEQ-TEST"

    t1 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "COD", {"cod": 100})
    s1 = queue.enqueue(dev_id, MessageCategory.SECURITY, "ENCLOSURE_OPEN", {"pin": "GPIO_17"})
    t2 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "BOD", {"bod": 50})
    s2 = queue.enqueue(dev_id, MessageCategory.SECURITY, "SEAL_BROKEN", {"wire": "BROKEN"})
    t3 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "TDS", {"tds": 400})

    assert t1.sequence_number == 1
    assert s1.sequence_number == 1  # Security starts at 1 independently!
    assert t2.sequence_number == 2
    assert s2.sequence_number == 2  # Security increments to 2
    assert t3.sequence_number == 3


# ==============================================================================
# 4. Priority Transmission (Tamper > Health > Telemetry)
# ==============================================================================

def test_priority_transmission_ordering(temp_db_path):
    """
    Messages are queued and dispatched in strict priority order:
      1. Physical Tamper (ENCLOSURE_OPEN, SEAL_BROKEN, SENSOR_REMOVED, SENSOR_DISCONNECTED)
      2. Connectivity Health (DEVICE_OFFLINE)
      3. Routine Telemetry
    """
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    dev_id = "DEV-PRIORITY"

    # Enqueue in reverse priority order
    t1 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "ROUTINE", {"cod": 100})
    h1 = queue.enqueue(dev_id, MessageCategory.SECURITY, "DEVICE_OFFLINE", {"reason": "heartbeat_timeout"})
    p1 = queue.enqueue(dev_id, MessageCategory.SECURITY, "ENCLOSURE_OPEN", {"door": "OPEN"})
    p2 = queue.enqueue(dev_id, MessageCategory.SECURITY, "SEAL_BROKEN", {"wire": "SEVERED"})

    # Fetch batch
    messages, _ = queue.get_pending_messages(limit=10)
    ordered_types = [m.message_type for m in messages]

    # Priority 1 (p1, p2) before Priority 2 (h1) before Priority 3 (t1)
    assert ordered_types[0] in ["ENCLOSURE_OPEN", "SEAL_BROKEN"]
    assert ordered_types[1] in ["ENCLOSURE_OPEN", "SEAL_BROKEN"]
    assert ordered_types[2] == "DEVICE_OFFLINE"
    assert ordered_types[3] == "ROUTINE"


# ==============================================================================
# 5. Anti-Starvation Mechanism
# ==============================================================================

def test_anti_starvation_yields_telemetry_under_security_storm(temp_db_path):
    """
    Under sustained security traffic, anti-starvation yields transmission
    slots to pending telemetry after max_consecutive_security transmissions.
    """
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    dev_id = "DEV-STARVE-TEST"

    # Enqueue 10 security events
    sec_ids = []
    for i in range(10):
        m = queue.enqueue(dev_id, MessageCategory.SECURITY, "ENCLOSURE_OPEN", {"idx": i})
        sec_ids.append(m.message_id)

    # Enqueue 2 telemetry readings
    t1 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "COD", {"cod": 500})
    t2 = queue.enqueue(dev_id, MessageCategory.TELEMETRY, "BOD", {"bod": 200})

    # With max_consecutive_security=3, after 3 security messages, telemetry must yield
    batch1, sec_count1 = queue.get_pending_messages(limit=3, max_consecutive_security=3, consecutive_security_count=0)
    assert len(batch1) == 3
    assert all(m.message_category == MessageCategory.SECURITY for m in batch1)
    assert sec_count1 == 3

    # Next call has consecutive_security_count=3 >= max_consecutive_security=3
    # It MUST yield the first slot to telemetry!
    batch2, sec_count2 = queue.get_pending_messages(limit=2, max_consecutive_security=3, consecutive_security_count=sec_count1)
    assert batch2[0].message_category == MessageCategory.TELEMETRY
    assert batch2[0].message_id == t1.message_id


# ==============================================================================
# 6. Network Outage & Reconnection Sync
# ==============================================================================

def test_network_outage_accumulates_messages_and_reconnection_syncs(temp_db_path):
    """
    Simulates complete network outage:
      - Messages accumulate in SQLite.
      - None are lost or deleted.
      - Upon network restoration, all messages are delivered and acknowledged.
    """
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    server = GatewayIngestionService()
    agent = EdgeGatewaySyncAgent(queue=queue, ingestion_service=server)

    # 1. Outage begins
    agent.is_online = False
    enqueued = []
    for i in range(5):
        m = queue.enqueue("DEV-OUTAGE", MessageCategory.TELEMETRY, "COD", {"reading": i})
        enqueued.append(m)

    # Attempt sync during outage
    res_offline = agent.transmit_pending_batch()
    assert res_offline["is_online"] is False
    assert res_offline["attempted"] == 0

    # Verify all 5 messages remain PENDING in SQLite
    status = queue.get_capacity_status()
    assert status.pending_messages == 5
    assert status.acknowledged_messages == 0

    # 2. Network restored
    agent.is_online = True
    flush_res = agent.flush_all()
    assert flush_res["total_acknowledged"] == 5
    assert flush_res["remaining_pending"] == 0

    # Verify all messages marked ACKNOWLEDGED in SQLite
    post_status = queue.get_capacity_status()
    assert post_status.pending_messages == 0
    assert post_status.acknowledged_messages == 5

    for m in enqueued:
        db_m = queue.get_message(m.message_id)
        assert db_m.delivery_status == DeliveryStatus.ACKNOWLEDGED
        assert db_m.server_ack_receipt is not None


# ==============================================================================
# 7. Idempotent Duplicate Delivery & Lost Acknowledgement
# ==============================================================================

def test_idempotent_duplicate_delivery_and_lost_ack(temp_db_path):
    """
    If a message is successfully processed by the server but the acknowledgement
    is lost in transit, retransmitting it returns DUPLICATE_ACKNOWLEDGED without
    creating duplicate records or triggering duplicate alarms.
    """
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    server = GatewayIngestionService(tamper_engine=tamper_manager)
    agent = EdgeGatewaySyncAgent(queue=queue, ingestion_service=server)

    # Enqueue a physical tamper event
    msg = queue.enqueue(
        "DEV-DUP-IDEM",
        MessageCategory.SECURITY,
        "ENCLOSURE_OPEN",
        {"event_state": "ACTIVE", "source": "HARDWARE"},
    )

    # 1. First attempt: simulate network dropping ACK
    agent.simulate_lost_ack_once = True
    agent.transmit_pending_batch()

    # Server recorded it!
    assert server.is_message_recorded(msg.message_id) is True
    # Tamper alarm was generated
    alerts = tamper_manager.get_alerts(device_id="DEV-DUP-IDEM")
    assert len(alerts) == 1

    # But gateway marked message FAILED because ACK never arrived
    db_msg1 = queue.get_message(msg.message_id)
    assert db_msg1.delivery_status == DeliveryStatus.FAILED

    # 2. Gateway retries transmission
    res2 = agent.transmit_pending_batch()
    assert res2["acknowledged"] == 1

    # Gateway is now ACKNOWLEDGED
    db_msg2 = queue.get_message(msg.message_id)
    assert db_msg2.delivery_status == DeliveryStatus.ACKNOWLEDGED
    assert db_msg2.server_ack_receipt is not None

    # Crucially: NO duplicate alert was created in tamper_manager!
    alerts_after = tamper_manager.get_alerts(device_id="DEV-DUP-IDEM")
    assert len(alerts_after) == 1


# ==============================================================================
# 8. Exponential Backoff with Jitter
# ==============================================================================

def test_exponential_backoff_with_jitter():
    """Verify exponential backoff scaling and randomized jitter."""
    # Deterministic test with fixed RNG seed
    rng = random.Random(42)

    wait_0 = ExponentialBackoffWithJitter.calculate_wait_seconds(0, base_seconds=1.0, jitter_pct=0.2, rng=rng)
    wait_1 = ExponentialBackoffWithJitter.calculate_wait_seconds(1, base_seconds=1.0, jitter_pct=0.2, rng=rng)
    wait_2 = ExponentialBackoffWithJitter.calculate_wait_seconds(2, base_seconds=1.0, jitter_pct=0.2, rng=rng)
    wait_capped = ExponentialBackoffWithJitter.calculate_wait_seconds(10, base_seconds=1.0, max_seconds=30.0, jitter_pct=0.1, rng=rng)

    assert 0.8 <= wait_0 <= 1.2
    assert 1.6 <= wait_1 <= 2.4
    assert 3.2 <= wait_2 <= 4.8
    assert wait_capped <= 33.0  # Capped at max_seconds + jitter


# ==============================================================================
# 9. Storage Limits, Capacity Warning & Retention Policies
# ==============================================================================

def test_storage_capacity_warning_levels(temp_db_path):
    """Queue reports NOMINAL, WARNING_75, CRITICAL_90_SECURITY_RESERVED, and STORAGE_EXHAUSTED."""
    # Capacity = 10 messages
    queue = StoreAndForwardQueue(
        db_path=temp_db_path,
        max_capacity_messages=10,
        telemetry_retention=TelemetryRetentionPolicy.REJECT_NEW_TELEMETRY,
    )

    # 1. 0 to 7 messages -> NOMINAL
    for i in range(7):
        queue.enqueue("DEV-CAP", MessageCategory.TELEMETRY, "M", {"i": i})
    assert queue.get_capacity_status().warning_state == StorageWarningState.NOMINAL

    # 2. 8th message (80%) -> WARNING_75
    queue.enqueue("DEV-CAP", MessageCategory.TELEMETRY, "M", {"i": 7})
    assert queue.get_capacity_status().warning_state == StorageWarningState.WARNING_75

    # 3. 9th message (90%) -> CRITICAL_90_SECURITY_RESERVED
    queue.enqueue("DEV-CAP", MessageCategory.TELEMETRY, "M", {"i": 8})
    assert queue.get_capacity_status().warning_state == StorageWarningState.CRITICAL_90_SECURITY_RESERVED

    # 4. In REJECT_NEW_TELEMETRY mode, attempting to enqueue telemetry at 90% raises StorageExhaustedError
    with pytest.raises(StorageExhaustedError) as exc:
        queue.enqueue("DEV-CAP", MessageCategory.TELEMETRY, "M", {"i": 9})
    assert "reserved for security" in str(exc.value).lower()

    # But SECURITY event IS allowed!
    sec_msg = queue.enqueue("DEV-CAP", MessageCategory.SECURITY, "ENCLOSURE_OPEN", {"door": "OPEN"})
    assert sec_msg is not None

    # 5. At 100% (10/10), even security raises StorageExhaustedError
    with pytest.raises(StorageExhaustedError) as exc2:
        queue.enqueue("DEV-CAP", MessageCategory.SECURITY, "SEAL_BROKEN", {"wire": "BROKEN"})
    assert "storage exhausted" in str(exc2.value).lower()


def test_drop_oldest_telemetry_policy_preserves_headroom(temp_db_path):
    """
    In DROP_OLDEST_TELEMETRY mode, enqueuing telemetry beyond 90% drops the oldest
    pending telemetry message and logs an auditable entry, preserving security space.
    """
    queue = StoreAndForwardQueue(
        db_path=temp_db_path,
        max_capacity_messages=10,
        telemetry_retention=TelemetryRetentionPolicy.DROP_OLDEST_TELEMETRY,
    )

    # Enqueue 9 telemetry messages (reaching 90%)
    first_msg = queue.enqueue("DEV-DROP", MessageCategory.TELEMETRY, "M", {"i": 0})
    for i in range(1, 9):
        queue.enqueue("DEV-DROP", MessageCategory.TELEMETRY, "M", {"i": i})

    # Enqueue 10th message: should drop oldest message (first_msg)
    new_msg = queue.enqueue("DEV-DROP", MessageCategory.TELEMETRY, "M", {"i": 9})
    assert new_msg is not None

    # first_msg was dropped
    assert queue.get_message(first_msg.message_id) is None

    # Audit log recorded the drop
    audit_entries = queue.get_audit_log()
    drop_entries = [e for e in audit_entries if e["action"] == "TELEMETRY_DROPPED_DUE_TO_STORAGE_LIMIT"]
    assert len(drop_entries) == 1
    assert drop_entries[0]["message_id"] == first_msg.message_id


# ==============================================================================
# 10. Local Disk Checksum Validation & Corruption Detection
# ==============================================================================

def test_local_payload_checksum_detects_disk_corruption(temp_db_path):
    """Tampering with SQLite payload text causes CorruptedPayloadError on read."""
    queue = StoreAndForwardQueue(db_path=temp_db_path)
    msg = queue.enqueue("DEV-CORRUPT", MessageCategory.TELEMETRY, "COD", {"cod": 500.0})

    # Illegitimately corrupt the payload JSON in the SQLite table
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE edge_messages SET payload_json = ? WHERE message_id = ?",
            ('{"cod": 10.0}', msg.message_id),  # changed 500 to 10
        )
        conn.commit()

    # Reading should trigger checksum failure
    with pytest.raises(CorruptedPayloadError) as exc:
        queue.get_message(msg.message_id)
    assert "checksum mismatch" in str(exc.value).lower()


# ==============================================================================
# 11. Complete Gateway Simulator Scenarios (A through F)
# ==============================================================================

def test_gateway_simulator_all_scenarios(temp_db_path):
    """Simulator successfully executes all 6 required store-and-forward scenarios."""
    simulator = GatewayStoreAndForwardSimulator(db_path=temp_db_path)
    results = simulator.run_all_scenarios()

    # Scenario A: Normal
    assert results["A"]["enqueued_status"] == "PENDING"
    assert results["A"]["final_status"] == "ACKNOWLEDGED"
    assert results["A"]["server_acknowledged"] is True

    # Scenario B: Outage
    assert results["B"]["is_online"] is False
    assert results["B"]["pending_in_sqlite"] == 5

    # Scenario C: Tamper during outage
    assert results["C"]["priority"] == 1  # Physical tamper is Priority 1
    assert results["C"]["security_pending_count"] == 1

    # Scenario D: Reconnection & restoration
    assert results["D"]["first_dispatched_category"] == "SECURITY"
    assert results["D"]["first_dispatched_type"] == "ENCLOSURE_OPEN"
    assert results["D"]["remaining_pending"] == 0

    # Scenario E: Duplicate delivery / lost ack
    assert results["E"]["first_attempt_server_recorded"] is True
    assert results["E"]["second_attempt_gateway_status"] == "ACKNOWLEDGED"

    # Scenario F: Application restart
    assert results["F"]["all_messages_recovered"] is True
    assert results["F"]["pre_restart_pending"] == 3
    assert results["F"]["post_restart_pending"] == 3


# ==============================================================================
# 12. FastAPI Endpoints & Authentication Integration
# ==============================================================================

def test_fastapi_gateway_ingest_endpoint():
    """POST /gateway/ingest processes incoming edge message and returns ServerAckResponse."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()

    msg_payload = {
        "message_id": "MSG-API-001",
        "device_id": "DEV-API-01",
        "message_category": "TELEMETRY",
        "message_type": "COD_READING",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "priority": 3,
        "payload": {"COD_mg_L": 480.0, "pH": 7.2},
        "payload_checksum": StoreAndForwardQueue.compute_checksum({"COD_mg_L": 480.0, "pH": 7.2}),
        "delivery_status": "PENDING",
        "retry_count": 0,
        "created_at": now_utc,
    }

    # First transmission -> ACKNOWLEDGED
    res1 = client.post("/gateway/ingest", json=msg_payload)
    assert res1.status_code == 200
    ack1 = res1.json()
    assert ack1["message_id"] == "MSG-API-001"
    assert ack1["status"] == "ACKNOWLEDGED"
    assert ack1["receipt_token"].startswith("RCT-")

    # Resend identical message -> DUPLICATE_ACKNOWLEDGED
    res2 = client.post("/gateway/ingest", json=msg_payload)
    assert res2.status_code == 200
    ack2 = res2.json()
    assert ack2["message_id"] == "MSG-API-001"
    assert ack2["status"] == "DUPLICATE_ACKNOWLEDGED"
    assert ack2["receipt_token"] == ack1["receipt_token"]


def test_fastapi_gateway_batch_endpoint():
    """POST /gateway/ingest/batch processes array of messages."""
    client = TestClient(app)
    now_utc = datetime.now(timezone.utc).isoformat()

    messages = [
        {
            "message_id": f"MSG-BATCH-{i}",
            "device_id": "DEV-BATCH",
            "message_category": "TELEMETRY",
            "message_type": "READING",
            "event_timestamp": now_utc,
            "sequence_number": i + 1,
            "priority": 3,
            "payload": {"idx": i},
            "payload_checksum": StoreAndForwardQueue.compute_checksum({"idx": i}),
            "delivery_status": "PENDING",
            "retry_count": 0,
            "created_at": now_utc,
        }
        for i in range(3)
    ]

    res = client.post("/gateway/ingest/batch", json=messages)
    assert res.status_code == 200
    acks = res.json()
    assert len(acks) == 3
    assert all(a["status"] == "ACKNOWLEDGED" for a in acks)


def test_fastapi_gateway_status_endpoint():
    """GET /gateway/status returns current ingestion metrics."""
    client = TestClient(app)
    res = client.get("/gateway/status")
    assert res.status_code == 200
    data = res.json()
    assert data["is_online"] is True
    assert "total_recorded_messages" in data


def test_fastapi_demo_simulate_endpoint():
    """POST /demo/gateway/simulate triggers simulator and returns scenario results."""
    client = TestClient(app)
    res = client.post("/demo/gateway/simulate?scenario=ALL")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["demonstration_mode"] is True
    assert "A" in data["results"]
    assert "D" in data["results"]


def test_fastapi_gateway_endpoints_enforce_auth(monkeypatch):
    """When backend token is configured, gateway endpoints require X-ML-Service-Token."""
    token = "secret-gateway-token-123"
    monkeypatch.setenv("ML_BACKEND_SERVICE_TOKEN", token)
    client = TestClient(app)

    now_utc = datetime.now(timezone.utc).isoformat()
    msg_payload = {
        "message_id": "MSG-AUTH-01",
        "device_id": "DEV-AUTH",
        "message_category": "TELEMETRY",
        "message_type": "COD",
        "event_timestamp": now_utc,
        "sequence_number": 1,
        "priority": 3,
        "payload": {"cod": 100},
        "payload_checksum": StoreAndForwardQueue.compute_checksum({"cod": 100}),
        "delivery_status": "PENDING",
        "retry_count": 0,
        "created_at": now_utc,
    }

    # Unauthenticated -> 401
    assert client.post("/gateway/ingest", json=msg_payload).status_code == 401
    assert client.post("/gateway/ingest/batch", json=[msg_payload]).status_code == 401
    assert client.get("/gateway/status").status_code == 401
    assert client.post("/demo/gateway/simulate").status_code == 401

    # Authenticated -> 200
    headers = {"X-ML-Service-Token": token}
    assert client.post("/gateway/ingest", json=msg_payload, headers=headers).status_code == 200
    assert client.get("/gateway/status", headers=headers).status_code == 200

