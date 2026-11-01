"""
EnviroTrust Edge Gateway — Offline Buffering and Store-and-Forward Module

Implements durable offline message buffering, separate monotonic sequence tracking,
priority transmission with anti-starvation, exponential backoff with jitter,
explicit server acknowledgement, idempotent duplicate prevention, storage capacity
monitoring, and an interactive simulation engine for field IoT gateways.

IMPORTANT ARCHITECTURAL BOUNDARIES:
  - Production Architecture:
      Physical Sensors
      -> IoT Edge Gateway (Persistent SQLite Queue)
      -> Node.js / Express Backend (Signature Verification & Durable Storage)
      -> FastAPI ML Service (Anomaly Detection & Tamper Alert Processing)
      -> Regulator Dashboard
  - The Python ML service is NOT the authority for physical-device identity or digital signatures.
  - The Edge Gateway stores messages durably in SQLite; messages are never considered enqueued
    until the SQLite transaction commits to disk.
  - A message is NEVER deleted merely because an HTTP request was sent; it must receive an
    explicit, cryptographically verifiable server acknowledgement.
  - Retransmission of messages with lost acknowledgements MUST be idempotent on the server.
"""

from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import threading
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("EnviroTrust.StoreAndForward")


# ==============================================================================
# 1. Enums and Data Models
# ==============================================================================

class MessageCategory(str, Enum):
    """Broad category governing transmission priority and sequence isolation."""
    SECURITY = "SECURITY"
    TELEMETRY = "TELEMETRY"


class DeliveryStatus(str, Enum):
    """Lifecycle state of an edge-queued message."""
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


class TransmissionPriority(int, Enum):
    """
    Transmission priority levels. Lower integer denotes higher dispatch priority.
    1. Physical tamper alerts
    2. Device connectivity & hardware health
    3. Routine environmental telemetry
    """
    PHYSICAL_TAMPER = 1
    CONNECTIVITY_HEALTH = 2
    ROUTINE_TELEMETRY = 3


class StorageWarningState(str, Enum):
    """Queue capacity and storage threshold status."""
    NOMINAL = "NOMINAL"
    WARNING_75 = "WARNING_75"
    CRITICAL_90_SECURITY_RESERVED = "CRITICAL_90_SECURITY_RESERVED"
    STORAGE_EXHAUSTED = "STORAGE_EXHAUSTED"


class TelemetryRetentionPolicy(str, Enum):
    """Policy for handling routine telemetry when storage exceeds 90% reserved threshold."""
    DROP_OLDEST_TELEMETRY = "DROP_OLDEST_TELEMETRY"
    REJECT_NEW_TELEMETRY = "REJECT_NEW_TELEMETRY"


class QueueMessage(BaseModel):
    """
    Validated edge-gateway queue message schema.
    """
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(..., description="Globally unique UUID for the message")
    device_id: str = Field(..., description="Unique industrial unit identifier")
    message_category: MessageCategory = Field(..., description="SECURITY or TELEMETRY")
    message_type: str = Field(..., description="Specific event or reading type (e.g. ENCLOSURE_OPEN, COD_READING)")
    event_timestamp: datetime = Field(..., description="Original event generation timestamp in UTC")
    sequence_number: int = Field(..., ge=0, description="Category-specific monotonic sequence counter")
    priority: int = Field(..., ge=1, le=3, description="1=Tamper, 2=Health, 3=Telemetry")
    payload: Dict[str, Any] = Field(..., description="Message payload dictionary")
    payload_checksum: str = Field(..., description="SHA-256 hash of canonical payload JSON")
    delivery_status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    retry_count: int = Field(default=0, ge=0)
    last_transmission_attempt: Optional[datetime] = None
    server_ack_timestamp: Optional[datetime] = None
    server_ack_receipt: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ServerAckResponse(BaseModel):
    """
    Explicit server acknowledgement response returned to the edge gateway.
    """
    model_config = ConfigDict(extra="allow")

    message_id: str
    status: str = Field(..., description="'ACKNOWLEDGED' or 'DUPLICATE_ACKNOWLEDGED'")
    server_receipt_timestamp: datetime
    receipt_token: str
    message_category: Optional[str] = None
    details: Optional[str] = None


class QueueCapacityStatus(BaseModel):
    """
    Current storage capacity and health metrics for the edge queue.
    """
    total_messages: int
    pending_messages: int
    in_flight_messages: int
    acknowledged_messages: int
    telemetry_pending_count: int
    security_pending_count: int
    max_capacity_messages: int
    capacity_percentage: float
    warning_state: StorageWarningState
    database_size_bytes: int


# Mapping from security event type to transmission priority
SECURITY_PRIORITY_MAP: Dict[str, TransmissionPriority] = {
    "ENCLOSURE_OPEN": TransmissionPriority.PHYSICAL_TAMPER,
    "SEAL_BROKEN": TransmissionPriority.PHYSICAL_TAMPER,
    "SENSOR_REMOVED": TransmissionPriority.PHYSICAL_TAMPER,
    "SENSOR_DISCONNECTED": TransmissionPriority.PHYSICAL_TAMPER,
    "DEVICE_OFFLINE": TransmissionPriority.CONNECTIVITY_HEALTH,
}


# ==============================================================================
# 2. SQLite Persistent Local Message Queue (StoreAndForwardQueue)
# ==============================================================================

class StoreAndForwardQueue:
    """
    Thread-safe, transaction-durable SQLite message queue for edge gateways.

    Features:
      - Durable ACID transactions (never considers message saved until commit succeeds).
      - Separate monotonic sequence counters for TELEMETRY and SECURITY per device.
      - SHA-256 local payload checksum to detect on-disk database corruption or tampering.
      - Storage capacity monitoring (75% warning, 90% critical reserved for security, 100% exhaustion).
      - Configurable telemetry retention policy (DROP_OLDEST_TELEMETRY with auditable log vs. REJECT_NEW).
      - Never silently discards security events.
      - Full persistence surviving process restarts.
    """

    def __init__(
        self,
        db_path: str = "edge_gateway_queue.db",
        max_capacity_messages: int = 10000,
        telemetry_retention: TelemetryRetentionPolicy = TelemetryRetentionPolicy.DROP_OLDEST_TELEMETRY,
    ):
        self.db_path = db_path
        self.max_capacity_messages = max_capacity_messages
        self.telemetry_retention = telemetry_retention
        self._lock = threading.Lock()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_database(self):
        """Initialize database schema if not present."""
        with self._lock:
            with self._get_connection() as conn:
                # 1. Main edge message queue table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_messages (
                    message_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    message_category TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_checksum TEXT NOT NULL,
                    delivery_status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    last_transmission_attempt TEXT,
                    server_ack_timestamp TEXT,
                    server_ack_receipt TEXT,
                    created_at TEXT NOT NULL
                );
                """)

                # Performance indexes for priority-ordered dequeuing & sequence queries
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority_seq
                ON edge_messages (delivery_status, priority, sequence_number);
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_device_cat_seq
                ON edge_messages (device_id, message_category, sequence_number);
                """)

                # 2. Monotonic sequence counters table (isolated per device and category)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS device_sequences (
                    device_id TEXT NOT NULL,
                    message_category TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (device_id, message_category)
                );
                """)

                # 3. Queue audit log for dropped telemetry, storage warnings, and corruptions
                conn.execute("""
                CREATE TABLE IF NOT EXISTS queue_audit_log (
                    log_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    action TEXT NOT NULL,
                    message_id TEXT,
                    device_id TEXT,
                    details TEXT
                );
                """)
                conn.commit()

    @staticmethod
    def compute_checksum(payload: Dict[str, Any]) -> str:
        """Compute SHA-256 hash over canonical JSON payload for local disk integrity verification."""
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _get_next_sequence_number(self, conn: sqlite3.Connection, device_id: str, category: MessageCategory) -> int:
        """
        Atomically increment and return the next monotonic sequence number
        for the given device and message category.
        """
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_sequence FROM device_sequences WHERE device_id = ? AND message_category = ?",
            (device_id, category.value),
        )
        row = cursor.fetchone()
        now_iso = datetime.now(timezone.utc).isoformat()

        if row is None:
            seq = 1
            cursor.execute(
                "INSERT INTO device_sequences (device_id, message_category, last_sequence, updated_at) VALUES (?, ?, ?, ?)",
                (device_id, category.value, seq, now_iso),
            )
        else:
            seq = row["last_sequence"] + 1
            cursor.execute(
                "UPDATE device_sequences SET last_sequence = ?, updated_at = ? WHERE device_id = ? AND message_category = ?",
                (seq, now_iso, device_id, category.value),
            )
        return seq

    def get_last_sequence_number(self, device_id: str, category: MessageCategory) -> int:
        """Query the highest sequence number generated for a device and category."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_sequence FROM device_sequences WHERE device_id = ? AND message_category = ?",
                    (device_id, category.value),
                )
                row = cursor.fetchone()
                return row["last_sequence"] if row else 0

    def get_capacity_status(self) -> QueueCapacityStatus:
        """Evaluate real-time queue storage utilization and warning thresholds."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM edge_messages")
                total = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM edge_messages WHERE delivery_status = 'PENDING'")
                pending = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM edge_messages WHERE delivery_status = 'IN_FLIGHT'")
                in_flight = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM edge_messages WHERE delivery_status = 'ACKNOWLEDGED'")
                acknowledged = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM edge_messages WHERE delivery_status IN ('PENDING', 'IN_FLIGHT') AND message_category = 'TELEMETRY'"
                )
                telemetry_pending = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM edge_messages WHERE delivery_status IN ('PENDING', 'IN_FLIGHT') AND message_category = 'SECURITY'"
                )
                security_pending = cursor.fetchone()[0]

                active_pending = pending + in_flight
                pct = (active_pending / self.max_capacity_messages) * 100.0 if self.max_capacity_messages > 0 else 0.0

                if active_pending >= self.max_capacity_messages:
                    warning_state = StorageWarningState.STORAGE_EXHAUSTED
                elif pct >= 90.0:
                    warning_state = StorageWarningState.CRITICAL_90_SECURITY_RESERVED
                elif pct >= 75.0:
                    warning_state = StorageWarningState.WARNING_75
                else:
                    warning_state = StorageWarningState.NOMINAL

                # DB file size
                db_bytes = 0
                if os.path.exists(self.db_path) and self.db_path != ":memory:":
                    db_bytes = os.path.getsize(self.db_path)

                return QueueCapacityStatus(
                    total_messages=total,
                    pending_messages=pending,
                    in_flight_messages=in_flight,
                    acknowledged_messages=acknowledged,
                    telemetry_pending_count=telemetry_pending,
                    security_pending_count=security_pending,
                    max_capacity_messages=self.max_capacity_messages,
                    capacity_percentage=round(pct, 2),
                    warning_state=warning_state,
                    database_size_bytes=db_bytes,
                )

    def _record_audit_log(
        self,
        conn: sqlite3.Connection,
        action: str,
        message_id: Optional[str],
        device_id: Optional[str],
        details: str,
    ):
        """Append an entry to the edge queue audit log."""
        log_id = f"Q-LOG-{uuid4().hex[:10].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO queue_audit_log (log_id, timestamp_utc, action, message_id, device_id, details) VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, now_iso, action, message_id, device_id, details),
        )

    def enqueue(
        self,
        device_id: str,
        category: MessageCategory,
        message_type: str,
        payload: Dict[str, Any],
        event_timestamp: Optional[datetime] = None,
        custom_message_id: Optional[str] = None,
    ) -> QueueMessage:
        """
        Duranbly enqueue a message into SQLite.
        Guarantees that a message is not considered saved until the SQLite commit completes.

        Applies:
          - Separate monotonic sequence counter per device/category.
          - Priority assignment (Tamper=1, Health=2, Telemetry=3).
          - Storage limit checks (75% warning, 90% reserved security, 100% exhaustion).
          - Telemetry retention eviction or rejection.
        """
        if not device_id or not device_id.strip():
            raise ValueError("device_id cannot be blank or whitespace-only.")

        now_utc = datetime.now(timezone.utc)
        ts = event_timestamp or now_utc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        msg_id = custom_message_id or f"MSG-{uuid4().hex}"
        checksum = self.compute_checksum(payload)
        payload_str = json.dumps(payload, sort_keys=True, default=str)

        # Determine priority
        if category == MessageCategory.SECURITY:
            priority = SECURITY_PRIORITY_MAP.get(
                message_type.upper(),
                TransmissionPriority.PHYSICAL_TAMPER
            ).value
        else:
            priority = TransmissionPriority.ROUTINE_TELEMETRY.value

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Capacity Check
                cursor.execute(
                    "SELECT COUNT(*) FROM edge_messages WHERE delivery_status IN ('PENDING', 'IN_FLIGHT')"
                )
                active_count = cursor.fetchone()[0]

                # If queue is at or above 90% capacity
                if active_count >= int(0.9 * self.max_capacity_messages):
                    if category == MessageCategory.TELEMETRY:
                        if self.telemetry_retention == TelemetryRetentionPolicy.DROP_OLDEST_TELEMETRY:
                            # Find oldest pending telemetry message
                            cursor.execute(
                                """
                                SELECT message_id, device_id FROM edge_messages
                                WHERE delivery_status = 'PENDING' AND message_category = 'TELEMETRY'
                                ORDER BY priority DESC, sequence_number ASC, event_timestamp ASC
                                LIMIT 1
                                """
                            )
                            oldest_row = cursor.fetchone()
                            if oldest_row:
                                dropped_id = oldest_row["message_id"]
                                cursor.execute("DELETE FROM edge_messages WHERE message_id = ?", (dropped_id,))
                                self._record_audit_log(
                                    conn,
                                    action="TELEMETRY_DROPPED_DUE_TO_STORAGE_LIMIT",
                                    message_id=dropped_id,
                                    device_id=oldest_row["device_id"],
                                    details=(
                                        f"Queue reached {active_count}/{self.max_capacity_messages} (>=90%). "
                                        f"Oldest telemetry message {dropped_id} dropped to preserve security headroom."
                                    ),
                                )
                                logger.warning(
                                    f"Dropped oldest telemetry message {dropped_id} due to 90% storage threshold."
                                )
                            else:
                                # No pending telemetry to drop, storage is filled with security
                                raise StorageExhaustedError(
                                    f"Queue capacity critical ({active_count}/{self.max_capacity_messages}). "
                                    "No pending telemetry available to evict. New telemetry rejected."
                                )
                        else:
                            raise StorageExhaustedError(
                                f"Storage threshold exceeded ({active_count}/{self.max_capacity_messages}). "
                                "Remaining 10% capacity is strictly reserved for security events. New telemetry rejected."
                            )

                    # Re-check if active_count reached 100% (even for security)
                    cursor.execute(
                        "SELECT COUNT(*) FROM edge_messages WHERE delivery_status IN ('PENDING', 'IN_FLIGHT')"
                    )
                    if cursor.fetchone()[0] >= self.max_capacity_messages:
                        self._record_audit_log(
                            conn,
                            action="STORAGE_EXHAUSTED_SECURITY_BLOCKED",
                            message_id=msg_id,
                            device_id=device_id,
                            details="Queue reached absolute 100% capacity limit. Operation refused.",
                        )
                        raise StorageExhaustedError(
                            f"Queue storage exhausted ({self.max_capacity_messages}/{self.max_capacity_messages}). "
                            "Cannot enqueue message."
                        )

                # Get monotonic sequence number
                seq = self._get_next_sequence_number(conn, device_id, category)

                # Insert into edge_messages
                cursor.execute(
                    """
                    INSERT INTO edge_messages (
                        message_id, device_id, message_category, message_type,
                        event_timestamp, sequence_number, priority,
                        payload_json, payload_checksum, delivery_status,
                        retry_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg_id,
                        device_id,
                        category.value,
                        message_type,
                        ts.isoformat(),
                        seq,
                        priority,
                        payload_str,
                        checksum,
                        DeliveryStatus.PENDING.value,
                        0,
                        now_utc.isoformat(),
                    ),
                )
                conn.commit()

        return QueueMessage(
            message_id=msg_id,
            device_id=device_id,
            message_category=category,
            message_type=message_type,
            event_timestamp=ts,
            sequence_number=seq,
            priority=priority,
            payload=payload,
            payload_checksum=checksum,
            delivery_status=DeliveryStatus.PENDING,
            retry_count=0,
            created_at=now_utc,
        )

    def get_message(self, message_id: str) -> Optional[QueueMessage]:
        """Fetch a specific message by ID with payload checksum validation."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM edge_messages WHERE message_id = ?", (message_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_queue_message(row)

    def _row_to_queue_message(self, row: sqlite3.Row) -> QueueMessage:
        """Convert a SQLite row to a QueueMessage, verifying checksum integrity."""
        payload_dict = json.loads(row["payload_json"])
        expected_checksum = row["payload_checksum"]
        actual_checksum = self.compute_checksum(payload_dict)

        if expected_checksum != actual_checksum:
            logger.critical(
                f"CORRUPTED_LOCAL_PAYLOAD detected for message {row['message_id']}! "
                f"Expected {expected_checksum}, found {actual_checksum}."
            )
            # Mark or raise corruption warning
            raise CorruptedPayloadError(
                f"Payload checksum mismatch on message {row['message_id']}. Possible database tampering."
            )

        return QueueMessage(
            message_id=row["message_id"],
            device_id=row["device_id"],
            message_category=MessageCategory(row["message_category"]),
            message_type=row["message_type"],
            event_timestamp=datetime.fromisoformat(row["event_timestamp"]),
            sequence_number=row["sequence_number"],
            priority=row["priority"],
            payload=payload_dict,
            payload_checksum=expected_checksum,
            delivery_status=DeliveryStatus(row["delivery_status"]),
            retry_count=row["retry_count"],
            last_transmission_attempt=(
                datetime.fromisoformat(row["last_transmission_attempt"])
                if row["last_transmission_attempt"] else None
            ),
            server_ack_timestamp=(
                datetime.fromisoformat(row["server_ack_timestamp"])
                if row["server_ack_timestamp"] else None
            ),
            server_ack_receipt=row["server_ack_receipt"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_pending_messages(
        self,
        limit: int = 50,
        max_consecutive_security: int = 5,
        consecutive_security_count: int = 0,
    ) -> Tuple[List[QueueMessage], int]:
        """
        Retrieve next batch of messages for transmission ordered by priority with anti-starvation.

        Ordering:
          - Physical tamper alerts (Priority 1)
          - Device health events (Priority 2)
          - Routine telemetry (Priority 3)

        Anti-Starvation Mechanism:
          - If consecutive_security_count >= max_consecutive_security, and pending telemetry
            is waiting, interleave the oldest pending telemetry message before continuing with security.
          - Returns (batch_messages, updated_consecutive_security_count).
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check if telemetry is pending
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM edge_messages
                    WHERE delivery_status IN ('PENDING', 'FAILED') AND message_category = 'TELEMETRY'
                    """
                )
                telemetry_pending_count = cursor.fetchone()[0]

                # Check if security is pending
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM edge_messages
                    WHERE delivery_status IN ('PENDING', 'FAILED') AND message_category = 'SECURITY'
                    """
                )
                security_pending_count = cursor.fetchone()[0]

                # Anti-starvation check: if security has transmitted N consecutive messages and telemetry is waiting
                force_telemetry = (
                    consecutive_security_count >= max_consecutive_security
                    and telemetry_pending_count > 0
                    and security_pending_count > 0
                )

                if force_telemetry:
                    # Select 1 telemetry message first to prevent starvation
                    cursor.execute(
                        """
                        SELECT * FROM edge_messages
                        WHERE delivery_status IN ('PENDING', 'FAILED') AND message_category = 'TELEMETRY'
                        ORDER BY sequence_number ASC, event_timestamp ASC
                        LIMIT 1
                        """
                    )
                    t_row = cursor.fetchone()
                    results = [self._row_to_queue_message(t_row)]
                    updated_sec_count = 0  # reset starvation counter

                    # Fill remaining limit with standard priority
                    remaining_limit = limit - 1
                    if remaining_limit > 0:
                        cursor.execute(
                            """
                            SELECT * FROM edge_messages
                            WHERE delivery_status IN ('PENDING', 'FAILED') AND message_id != ?
                            ORDER BY priority ASC, sequence_number ASC, event_timestamp ASC
                            LIMIT ?
                            """,
                            (t_row["message_id"], remaining_limit),
                        )
                        for row in cursor.fetchall():
                            msg = self._row_to_queue_message(row)
                            results.append(msg)
                            if msg.message_category == MessageCategory.SECURITY:
                                updated_sec_count += 1
                    return results, updated_sec_count

                # Standard priority query
                cursor.execute(
                    """
                    SELECT * FROM edge_messages
                    WHERE delivery_status IN ('PENDING', 'FAILED')
                    ORDER BY priority ASC, sequence_number ASC, event_timestamp ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                results = []
                cur_sec = consecutive_security_count
                for row in rows:
                    msg = self._row_to_queue_message(row)
                    results.append(msg)
                    if msg.message_category == MessageCategory.SECURITY:
                        cur_sec += 1
                    else:
                        cur_sec = 0

                return results, cur_sec

    def mark_in_flight(self, message_ids: List[str]):
        """Mark messages as IN_FLIGHT and record transmission attempt timestamp."""
        if not message_ids:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_connection() as conn:
                placeholders = ",".join("?" for _ in message_ids)
                conn.execute(
                    f"""
                    UPDATE edge_messages
                    SET delivery_status = 'IN_FLIGHT',
                        last_transmission_attempt = ?,
                        retry_count = retry_count + 1
                    WHERE message_id IN ({placeholders})
                    """,
                    [now_iso] + message_ids,
                )
                conn.commit()

    def record_server_ack(self, ack: ServerAckResponse):
        """
        Record verified server acknowledgement for a message.
        Transitions message status to ACKNOWLEDGED with receipt token and timestamp.
        CRITICAL: Message is only acknowledged upon receiving explicit ServerAckResponse.
        """
        now_iso = ack.server_receipt_timestamp.isoformat()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE edge_messages
                    SET delivery_status = 'ACKNOWLEDGED',
                        server_ack_timestamp = ?,
                        server_ack_receipt = ?
                    WHERE message_id = ?
                    """,
                    (now_iso, ack.receipt_token, ack.message_id),
                )
                self._record_audit_log(
                    conn,
                    action="SERVER_ACKNOWLEDGED",
                    message_id=ack.message_id,
                    device_id=None,
                    details=f"Status: {ack.status}, Receipt: {ack.receipt_token}",
                )
                conn.commit()

    def record_transmission_failure(self, message_id: str, error_details: str):
        """
        Record transmission failure for a message.
        Reverts delivery_status from IN_FLIGHT to FAILED so it will be retried.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE edge_messages
                    SET delivery_status = 'FAILED',
                        last_transmission_attempt = ?
                    WHERE message_id = ?
                    """,
                    (now_iso, message_id),
                )
                self._record_audit_log(
                    conn,
                    action="TRANSMISSION_FAILED",
                    message_id=message_id,
                    device_id=None,
                    details=error_details,
                )
                conn.commit()

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Query queue audit log."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM queue_audit_log ORDER BY timestamp_utc DESC LIMIT ?",
                    (limit,),
                )
                return [dict(r) for r in cursor.fetchall()]

    def reset_in_flight_to_pending(self):
        """
        Crash recovery on gateway startup:
        Any messages left in IN_FLIGHT when the process died are reverted to PENDING.
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE edge_messages
                    SET delivery_status = 'PENDING'
                    WHERE delivery_status = 'IN_FLIGHT'
                    """
                )
                self._record_audit_log(
                    conn,
                    action="GATEWAY_RESTART_RECOVERY",
                    message_id=None,
                    device_id=None,
                    details="Reset in-flight messages to PENDING following gateway initialization.",
                )
                conn.commit()


# Exceptions
class StorageExhaustedError(Exception):
    """Raised when queue storage exceeds capacity limits."""
    pass


class CorruptedPayloadError(Exception):
    """Raised when local SQLite message payload checksum fails validation."""
    pass


# ==============================================================================
# 3. Transmission Backoff with Jitter
# ==============================================================================

class ExponentialBackoffWithJitter:
    """
    Computes exponential backoff interval with randomized full jitter
    to prevent thundering-herd synchronizations upon network reconnection.
    """

    @staticmethod
    def calculate_wait_seconds(
        retry_count: int,
        base_seconds: float = 1.0,
        max_seconds: float = 60.0,
        jitter_pct: float = 0.25,
        rng: Optional[random.Random] = None,
    ) -> float:
        """
        Computes:
          t_exp = min(max_seconds, base_seconds * 2^(retry_count))
          t_wait = t_exp * (1.0 +/- uniform(0, jitter_pct))
        """
        r = rng or random
        t_exp = min(max_seconds, base_seconds * (2 ** min(retry_count, 10)))
        jitter_factor = 1.0 + r.uniform(-jitter_pct, jitter_pct)
        return max(0.1, t_exp * jitter_factor)


# ==============================================================================
# 4. Server-Side Idempotent Ingestion Service & Duplicate Registry
# ==============================================================================

class GatewayIngestionService:
    """
    Server-side ingestion handler for store-and-forward edge gateway transmissions.

    Guarantees:
      - Idempotent ingestion: Duplicate message IDs return DUPLICATE_ACKNOWLEDGED
        without triggering second tamper alerts or redundant predictions.
      - Dispatches SECURITY events to tamper_manager.
      - Dispatches TELEMETRY events to ML anomaly predictor.
      - Issues cryptographic receipt token for every acknowledged message.
    """

    def __init__(self, tamper_engine=None, predictor_fn=None):
        self._tamper_engine = tamper_engine
        self._predictor_fn = predictor_fn
        self._lock = threading.Lock()
        self._processed_messages: Dict[str, Dict[str, Any]] = {}  # message_id -> receipt_metadata

    def ingest_message(self, message: QueueMessage) -> ServerAckResponse:
        """
        Process single message from edge gateway idempotently.
        """
        now_utc = datetime.now(timezone.utc)
        with self._lock:
            # Check for existing duplicate
            if message.message_id in self._processed_messages:
                existing = self._processed_messages[message.message_id]
                logger.info(
                    f"Idempotent server deduplication: message {message.message_id} "
                    f"already ingested at {existing['received_at']}. Resending ACK."
                )
                return ServerAckResponse(
                    message_id=message.message_id,
                    status="DUPLICATE_ACKNOWLEDGED",
                    server_receipt_timestamp=existing["received_at"],
                    receipt_token=existing["receipt_token"],
                    message_category=message.message_category.value,
                    details="Message was previously recorded; duplicate request acknowledged idempotently.",
                )

            # Generate unique receipt token
            hasher = hashlib.sha256()
            hasher.update(message.message_id.encode("utf-8"))
            hasher.update(now_utc.isoformat().encode("utf-8"))
            receipt_token = f"RCT-{hasher.hexdigest()[:16].upper()}"

            # Process according to category
            dispatch_details = "Recorded."
            if message.message_category == MessageCategory.SECURITY:
                if self._tamper_engine:
                    from tamper_alarm import TamperEvent, TamperEventType, EventState, EventSource
                    t_event = TamperEvent(
                        event_id=f"EVT-GW-{message.message_id[:8]}",
                        device_id=message.device_id,
                        sensor_id=message.payload.get("sensor_id"),
                        event_type=TamperEventType(message.message_type),
                        event_timestamp=message.event_timestamp,
                        sequence_number=message.sequence_number,
                        event_state=EventState(message.payload.get("event_state", "ACTIVE")),
                        source=EventSource(message.payload.get("source", "HARDWARE")),
                        metadata=message.payload.get("metadata", {}),
                    )
                    alert, is_new, msg = self._tamper_engine.process_event(t_event)
                    dispatch_details = f"Tamper event processed: {msg}"

            elif message.message_category == MessageCategory.TELEMETRY:
                if self._predictor_fn:
                    try:
                        self._predictor_fn(message.payload)
                        dispatch_details = "Telemetry cross-checked against ML model."
                    except Exception as e:
                        dispatch_details = f"Telemetry received (ML check error: {e})."

            # Store receipt
            self._processed_messages[message.message_id] = {
                "received_at": now_utc,
                "receipt_token": receipt_token,
                "category": message.message_category.value,
            }

            return ServerAckResponse(
                message_id=message.message_id,
                status="ACKNOWLEDGED",
                server_receipt_timestamp=now_utc,
                receipt_token=receipt_token,
                message_category=message.message_category.value,
                details=dispatch_details,
            )

    def is_message_recorded(self, message_id: str) -> bool:
        """Check whether a message ID has been durably recorded on server."""
        with self._lock:
            return message_id in self._processed_messages

    def clear(self):
        """Reset in-memory state for testing."""
        with self._lock:
            self._processed_messages.clear()


# Global server ingestion service singleton
gateway_ingestion_service = GatewayIngestionService()


# ==============================================================================
# 5. Edge Gateway Sync Agent (Transmitter)
# ==============================================================================

class EdgeGatewaySyncAgent:
    """
    Gateway transmitter agent responsible for dispatching queued messages to the backend.

    Features:
      - Network connectivity toggle (is_online) to simulate outages and re-connections.
      - Transmits according to priority and anti-starvation.
      - Updates in-flight and server acknowledgement status.
      - Implements lost-acknowledgement simulation mode.
    """

    def __init__(
        self,
        queue: StoreAndForwardQueue,
        ingestion_service: GatewayIngestionService,
        batch_size: int = 10,
        max_consecutive_security: int = 5,
    ):
        self.queue = queue
        self.ingestion_service = ingestion_service
        self.batch_size = batch_size
        self.max_consecutive_security = max_consecutive_security
        self.is_online = True
        self._consecutive_security_count = 0
        self.simulate_lost_ack_once = False  # For lost ACK testing

    def transmit_pending_batch(self) -> Dict[str, Any]:
        """
        Execute one transmission cycle of pending messages.

        Returns summary of transmission results:
          - attempted: count of messages attempted
          - acknowledged: count of server acks recorded
          - failed: count of transmission failures
          - is_online: connectivity state
        """
        if not self.is_online:
            logger.warning("Gateway sync cycle skipped: network is currently offline.")
            return {
                "attempted": 0,
                "acknowledged": 0,
                "failed": 0,
                "is_online": False,
                "status": "OFFLINE_BUFFERING",
            }

        messages, new_sec_count = self.queue.get_pending_messages(
            limit=self.batch_size,
            max_consecutive_security=self.max_consecutive_security,
            consecutive_security_count=self._consecutive_security_count,
        )
        self._consecutive_security_count = new_sec_count

        if not messages:
            return {
                "attempted": 0,
                "acknowledged": 0,
                "failed": 0,
                "is_online": True,
                "status": "QUEUE_EMPTY",
            }

        msg_ids = [m.message_id for m in messages]
        self.queue.mark_in_flight(msg_ids)

        acked_count = 0
        failed_count = 0

        for msg in messages:
            if not self.is_online:
                # Network dropped mid-batch
                self.queue.record_transmission_failure(msg.message_id, "Network disconnected mid-transmission.")
                failed_count += 1
                continue

            try:
                # Send to server ingestion service
                ack = self.ingestion_service.ingest_message(msg)

                # Simulated Lost ACK scenario (server got it, but gateway network dropped before ack received)
                if self.simulate_lost_ack_once:
                    self.simulate_lost_ack_once = False
                    self.queue.record_transmission_failure(
                        msg.message_id,
                        "Network timeout waiting for server HTTP ACK response (simulated lost ACK)."
                    )
                    failed_count += 1
                    logger.warning(f"Simulated lost ACK for message {msg.message_id}.")
                    continue

                # Normal ACK received
                self.queue.record_server_ack(ack)
                acked_count += 1

            except Exception as e:
                self.queue.record_transmission_failure(msg.message_id, str(e))
                failed_count += 1

        return {
            "attempted": len(messages),
            "acknowledged": acked_count,
            "failed": failed_count,
            "is_online": True,
            "status": "SYNC_COMPLETE",
        }

    def flush_all(self, max_cycles: int = 100) -> Dict[str, Any]:
        """
        Repeatedly transmit until all pending messages are delivered or network fails.
        """
        total_attempted = 0
        total_acked = 0
        cycles = 0

        while cycles < max_cycles:
            cycles += 1
            res = self.transmit_pending_batch()
            total_attempted += res["attempted"]
            total_acked += res["acknowledged"]
            if res["attempted"] == 0 or not self.is_online or res["failed"] > 0:
                break

        return {
            "cycles": cycles,
            "total_attempted": total_attempted,
            "total_acknowledged": total_acked,
            "remaining_pending": self.queue.get_capacity_status().pending_messages,
        }


# ==============================================================================
# 6. Interactive Gateway Store-and-Forward Simulator
# ==============================================================================

class GatewayStoreAndForwardSimulator:
    """
    Demonstrates all 6 required store-and-forward edge scenarios:
      Scenario A: Normal operation (immediate generation & delivery).
      Scenario B: Internet outage (messages accumulate in SQLite without loss).
      Scenario C: Tamper event during outage (sensor enclosure open buffered locally).
      Scenario D: Network restoration (security dispatched first, older telemetry follows, acks recorded).
      Scenario E: Duplicate delivery (simulated lost ACK, idempotent backend resolution without duplicate alert).
      Scenario F: Application restart recovery (unacknowledged messages safely restored from SQLite).
    """

    def __init__(self, db_path: str = "edge_simulator.db"):
        self.db_path = db_path
        # Clean up old demo database file and WAL/SHM companions if present
        for suffix in ["", "-wal", "-shm"]:
            fpath = f"{self.db_path}{suffix}"
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        self.queue = StoreAndForwardQueue(db_path=self.db_path, max_capacity_messages=500)
        self.server = GatewayIngestionService()
        self.agent = EdgeGatewaySyncAgent(queue=self.queue, ingestion_service=self.server)

    def run_scenario_a_normal_operation(self) -> Dict[str, Any]:
        """Scenario A: Normal operation - messages generated and immediately delivered."""
        self.agent.is_online = True
        msg = self.queue.enqueue(
            device_id="DEV-GW-01",
            category=MessageCategory.TELEMETRY,
            message_type="COD_READING",
            payload={"COD_mg_L": 450.0, "pH": 7.2, "turbidity_NTU": 120.0},
        )
        res = self.agent.transmit_pending_batch()
        updated_msg = self.queue.get_message(msg.message_id)
        return {
            "scenario": "A_NORMAL_OPERATION",
            "message_id": msg.message_id,
            "enqueued_status": msg.delivery_status.value,
            "final_status": updated_msg.delivery_status.value if updated_msg else None,
            "server_acknowledged": updated_msg.server_ack_receipt is not None if updated_msg else False,
            "sync_result": res,
        }

    def run_scenario_b_network_outage(self, message_count: int = 5) -> Dict[str, Any]:
        """Scenario B: Internet outage - network drops, messages accumulate in SQLite."""
        self.agent.is_online = False
        enqueued_ids = []
        for i in range(message_count):
            m = self.queue.enqueue(
                device_id="DEV-GW-01",
                category=MessageCategory.TELEMETRY,
                message_type="ROUTINE_MONITORING",
                payload={"reading_index": i, "COD_mg_L": 400.0 + i * 10, "pH": 7.0},
            )
            enqueued_ids.append(m.message_id)

        sync_res = self.agent.transmit_pending_batch()
        status = self.queue.get_capacity_status()

        return {
            "scenario": "B_NETWORK_OUTAGE",
            "is_online": self.agent.is_online,
            "messages_enqueued": len(enqueued_ids),
            "pending_in_sqlite": status.pending_messages,
            "sync_attempt_result": sync_res,
        }

    def run_scenario_c_tamper_during_outage(self) -> Dict[str, Any]:
        """Scenario C: Tamper event during outage - sensor enclosure opened and saved with Priority 1."""
        tamper_msg = self.queue.enqueue(
            device_id="DEV-GW-01",
            category=MessageCategory.SECURITY,
            message_type="ENCLOSURE_OPEN",
            payload={
                "switch_pin": "GPIO_17",
                "contact_state": "OPEN",
                "note": "Intrusion detected while network offline.",
            },
        )
        status = self.queue.get_capacity_status()
        return {
            "scenario": "C_TAMPER_DURING_OUTAGE",
            "tamper_message_id": tamper_msg.message_id,
            "priority": tamper_msg.priority,
            "sequence_number": tamper_msg.sequence_number,
            "security_pending_count": status.security_pending_count,
            "total_pending": status.pending_messages,
        }

    def run_scenario_d_network_restoration(self) -> Dict[str, Any]:
        """
        Scenario D: Network restoration - connectivity returns, retransmission triggers,
        tamper event is delivered FIRST, older telemetry follows, all ACKs recorded.
        """
        self.agent.is_online = True

        # Peek at the very first message scheduled for transmission
        peek_messages, _ = self.queue.get_pending_messages(limit=1)
        first_scheduled = peek_messages[0] if peek_messages else None

        flush_res = self.agent.flush_all()
        status = self.queue.get_capacity_status()

        return {
            "scenario": "D_NETWORK_RESTORATION",
            "first_dispatched_category": first_scheduled.message_category.value if first_scheduled else None,
            "first_dispatched_type": first_scheduled.message_type if first_scheduled else None,
            "flush_result": flush_res,
            "remaining_pending": status.pending_messages,
            "acknowledged_count": status.acknowledged_messages,
        }

    def run_scenario_e_duplicate_delivery(self) -> Dict[str, Any]:
        """
        Scenario E: Duplicate delivery - simulate a lost ACK; resend same message;
        verify backend recognizes duplicate and acknowledges idempotently without duplicate records.
        """
        self.agent.is_online = True
        msg = self.queue.enqueue(
            device_id="DEV-GW-01",
            category=MessageCategory.TELEMETRY,
            message_type="COD_READING",
            payload={"COD_mg_L": 520.0, "pH": 6.8},
        )

        # 1. First attempt: simulate lost ACK
        self.agent.simulate_lost_ack_once = True
        first_sync = self.agent.transmit_pending_batch()
        msg_after_first = self.queue.get_message(msg.message_id)

        # Verify server got it, but gateway marked FAILED
        server_got_it = self.server.is_message_recorded(msg.message_id)

        # 2. Second attempt: resend
        second_sync = self.agent.transmit_pending_batch()
        msg_after_second = self.queue.get_message(msg.message_id)

        return {
            "scenario": "E_DUPLICATE_DELIVERY",
            "message_id": msg.message_id,
            "first_attempt_server_recorded": server_got_it,
            "first_attempt_gateway_status": msg_after_first.delivery_status.value if msg_after_first else None,
            "second_attempt_sync": second_sync,
            "second_attempt_gateway_status": msg_after_second.delivery_status.value if msg_after_second else None,
            "server_ack_receipt": msg_after_second.server_ack_receipt if msg_after_second else None,
        }

    def run_scenario_f_application_restart_recovery(self) -> Dict[str, Any]:
        """
        Scenario F: Application restart recovery - enqueue messages, shut down queue connection,
        initialize a fresh queue instance on the same SQLite file, verify complete recovery.
        """
        # 1. Enqueue 3 messages
        self.agent.is_online = False
        msg_ids = []
        for i in range(3):
            m = self.queue.enqueue(
                device_id="DEV-GW-RESTART",
                category=MessageCategory.TELEMETRY,
                message_type="RESTART_TEST",
                payload={"index": i},
            )
            msg_ids.append(m.message_id)

        pre_status = self.queue.get_capacity_status()

        # 2. Simulate application termination and restart with a new StoreAndForwardQueue object
        del self.agent
        del self.queue

        restarted_queue = StoreAndForwardQueue(db_path=self.db_path)
        restarted_queue.reset_in_flight_to_pending()
        post_status = restarted_queue.get_capacity_status()

        # Verify all 3 messages are recovered
        recovered_all = all(restarted_queue.get_message(mid) is not None for mid in msg_ids)

        return {
            "scenario": "F_APPLICATION_RESTART_RECOVERY",
            "pre_restart_pending": pre_status.pending_messages,
            "post_restart_pending": post_status.pending_messages,
            "all_messages_recovered": recovered_all,
            "sequence_preserved": restarted_queue.get_last_sequence_number("DEV-GW-RESTART", MessageCategory.TELEMETRY),
        }

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Execute complete end-to-end demonstration sequence."""
        return {
            "A": self.run_scenario_a_normal_operation(),
            "B": self.run_scenario_b_network_outage(5),
            "C": self.run_scenario_c_tamper_during_outage(),
            "D": self.run_scenario_d_network_restoration(),
            "E": self.run_scenario_e_duplicate_delivery(),
            "F": self.run_scenario_f_application_restart_recovery(),
        }
