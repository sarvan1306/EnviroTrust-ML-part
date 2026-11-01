"""
EnviroTrust ML Service — Physical Sensor Tamper Alarm Module

Provides security monitoring, physical tamper event validation, alert generation,
audit logging, and hardware simulation for industrial wastewater monitoring units.

IMPORTANT ARCHITECTURAL BOUNDARIES:
  - This is a SECURITY FEATURE, completely distinct from ML anomaly detection (Isolation Forest).
  - Physical tamper detection operates independently of numerical pollution models.
  - Digital signature verification remains with the backend / cryptographic component.
  - An event is only treated as authentic once received through verified backend channels.
  - DEVICE_OFFLINE is treated as a connectivity warning, NOT confirmed physical tampering.
  - A CLEARED event preserves complete historical incident records in the tamper-evident audit ledger.
  - Alerts denote possible physical interference with monitoring hardware, not automated declarations
    of pollution fraud.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("EnviroTrust.TamperAlarm")


# ==============================================================================
# 1. Hardware Event Enums & Constants
# ==============================================================================

class TamperEventType(str, Enum):
    """
    Supported physical tamper and hardware security event types.
    """
    ENCLOSURE_OPEN = "ENCLOSURE_OPEN"
    SEAL_BROKEN = "SEAL_BROKEN"
    SENSOR_DISCONNECTED = "SENSOR_DISCONNECTED"
    SENSOR_REMOVED = "SENSOR_REMOVED"
    DEVICE_OFFLINE = "DEVICE_OFFLINE"


class EventState(str, Enum):
    """Active or cleared state of a physical tamper condition."""
    ACTIVE = "ACTIVE"
    CLEARED = "CLEARED"


class EventSource(str, Enum):
    """Provenance of the event: genuine hardware signal vs. software demonstration."""
    HARDWARE = "HARDWARE"
    SIMULATED = "SIMULATED"


class AlertSeverity(str, Enum):
    """Severity classification for security alerts."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    INFO = "INFO"


class AlertCategory(str, Enum):
    """Category distinguishing confirmed physical tampering from connectivity warnings."""
    PHYSICAL_TAMPER = "PHYSICAL_TAMPER"
    CONNECTIVITY_WARNING = "CONNECTIVITY_WARNING"


class DeviceTamperStatus(str, Enum):
    """Current real-time tamper status of an industrial monitoring device."""
    NOMINAL = "NOMINAL"
    TAMPER_ACTIVE = "TAMPER_ACTIVE"
    OFFLINE_WARNING = "OFFLINE_WARNING"


# Severity and categorization mapping per hardware event type
EVENT_SEVERITY_MAP: Dict[TamperEventType, Tuple[AlertSeverity, AlertCategory, bool]] = {
    TamperEventType.ENCLOSURE_OPEN: (
        AlertSeverity.CRITICAL,
        AlertCategory.PHYSICAL_TAMPER,
        True,   # is_confirmed_tamper
    ),
    TamperEventType.SEAL_BROKEN: (
        AlertSeverity.CRITICAL,
        AlertCategory.PHYSICAL_TAMPER,
        True,
    ),
    TamperEventType.SENSOR_REMOVED: (
        AlertSeverity.CRITICAL,
        AlertCategory.PHYSICAL_TAMPER,
        True,
    ),
    TamperEventType.SENSOR_DISCONNECTED: (
        AlertSeverity.HIGH,
        AlertCategory.PHYSICAL_TAMPER,
        True,
    ),
    TamperEventType.DEVICE_OFFLINE: (
        AlertSeverity.MEDIUM,
        AlertCategory.CONNECTIVITY_WARNING,
        False,  # NOT confirmed physical tampering
    ),
}

EVENT_DESCRIPTIONS: Dict[TamperEventType, str] = {
    TamperEventType.ENCLOSURE_OPEN: "Protective sensor enclosure door or casing opened.",
    TamperEventType.SEAL_BROKEN: "Physical tamper-seal loop or security seal circuit broken.",
    TamperEventType.SENSOR_DISCONNECTED: "Sensor probe or wiring connection disconnected from controller.",
    TamperEventType.SENSOR_REMOVED: "Sensor probe physically extracted or dismounted from effluent channel.",
    TamperEventType.DEVICE_OFFLINE: "Device communication lost. Connectivity warning (possible tampering, unconfirmed).",
}


# ==============================================================================
# 2. Pydantic Schemas for Event, Alert, and Audit Log
# ==============================================================================

class TamperEvent(BaseModel):
    """
    Validated physical tamper event schema transmitted from IoT controller or backend.
    """
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., description="Unique event identifier (UUID)", min_length=3)
    device_id: str = Field(..., description="Industrial monitoring unit device ID", min_length=3)
    sensor_id: Optional[str] = Field(None, description="Affected sensor ID, where applicable")
    event_type: TamperEventType = Field(..., description="Hardware event classification")
    event_timestamp: datetime = Field(..., description="Event generation timestamp in UTC (ISO 8601)")
    received_timestamp: Optional[datetime] = Field(None, description="Server receipt timestamp in UTC")
    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number from device")
    event_state: EventState = Field(default=EventState.ACTIVE, description="ACTIVE or CLEARED condition")
    source: EventSource = Field(default=EventSource.HARDWARE, description="HARDWARE or SIMULATED provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional diagnostic or electrical telemetry")

    @field_validator("event_id", "device_id")
    @classmethod
    def validate_non_whitespace(cls, v: str, info) -> str:
        s = v.strip()
        if not s:
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace only.")
        return s

    @field_validator("event_timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        # Normalize to UTC
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        # Allow max 5 minutes clock skew for field IoT devices
        if v > now_utc + timedelta(minutes=5):
            raise ValueError(f"Event timestamp {v.isoformat()} cannot be in the future (exceeds 5 min skew tolerance).")
        return v


class TamperAlert(BaseModel):
    """
    High-priority security alert generated for government regulators.
    """
    alert_id: str = Field(..., description="Unique alert identifier")
    device_id: str = Field(..., description="Affected industrial unit device ID")
    sensor_id: Optional[str] = Field(None, description="Affected sensor probe ID")
    originating_event_id: str = Field(..., description="Triggering tamper event ID")
    event_type: TamperEventType = Field(..., description="Hardware event classification")
    severity: AlertSeverity = Field(..., description="CRITICAL, HIGH, MEDIUM, or INFO")
    category: AlertCategory = Field(..., description="PHYSICAL_TAMPER or CONNECTIVITY_WARNING")
    is_confirmed_tamper: bool = Field(..., description="True for physical switches/circuits; False for connectivity")
    status: str = Field(..., description="ACTIVE or CLEARED")
    summary: str = Field(..., description="Human-readable alert summary")
    details: str = Field(..., description="Regulatory advice and incident context")
    created_at_utc: datetime = Field(..., description="Alert generation timestamp in UTC")
    cleared_at_utc: Optional[datetime] = Field(None, description="Clearance timestamp in UTC")
    source: EventSource = Field(..., description="HARDWARE or SIMULATED")
    is_simulated: bool = Field(..., description="Flag indicating demonstration simulation")
    audit_hash: str = Field(..., description="Cryptographic hash linking alert to audit ledger")


class DeviceStatus(BaseModel):
    """
    Real-time physical security status of an industrial monitoring device.
    """
    device_id: str
    tamper_status: DeviceTamperStatus
    active_alerts_count: int
    active_event_types: List[str]
    last_event_timestamp: Optional[datetime] = None
    last_heartbeat_timestamp: Optional[datetime] = None
    highest_sequence_seen: int = 0
    total_incidents_recorded: int = 0
    is_simulated: bool = False
    updated_at_utc: datetime


class AuditLogEntry(BaseModel):
    """
    Tamper-evident audit ledger record with cryptographic hash chaining.
    """
    entry_id: str
    timestamp_utc: datetime
    action: str  # EVENT_RECORDED, ALERT_GENERATED, ALERT_CLEARED, DUPLICATE_IGNORED
    event_id: str
    device_id: str
    event_type: str
    event_state: str
    source: str
    details: str
    prev_hash: str
    entry_hash: str


class TamperSimulationRequest(BaseModel):
    """
    Request payload to trigger a simulated tamper scenario for demonstration.
    """
    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(
        ...,
        description="Scenario: 'enclosure_open', 'seal_broken', 'sensor_disconnected', 'sensor_removed', 'device_offline', 'clear'"
    )
    device_id: str = Field(default="DEV-SIM-001", min_length=3)
    sensor_id: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


# ==============================================================================
# 3. Core Engine: TamperAlarmManager
# ==============================================================================

class TamperAlarmManager:
    """
    Central manager for physical sensor tamper alarms and tamper-evident audit logging.
    Thread-safe implementation with monotonic sequence tracking, deduplication,
    cryptographic ledger chaining, and simulation support.
    """

    GENESIS_HASH: str = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self):
        self._lock = threading.Lock()
        self._events: Dict[str, TamperEvent] = {}
        self._device_events: Dict[str, List[TamperEvent]] = {}
        self._alerts: Dict[str, TamperAlert] = {}
        self._devices: Dict[str, DeviceStatus] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._seen_keys: Set[Tuple[str, str, int]] = set()  # (device_id, event_type, sequence_number)
        self._last_hash: str = self.GENESIS_HASH

    def _compute_audit_hash(self, prev_hash: str, payload: Dict[str, Any]) -> str:
        """Compute SHA-256 hash chaining to ensure tamper-evident audit history."""
        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode("utf-8"))
        canonical_json = json.dumps(payload, sort_keys=True, default=str)
        hasher.update(canonical_json.encode("utf-8"))
        return hasher.hexdigest()

    def _record_audit_entry(
        self,
        action: str,
        event: TamperEvent,
        details: str,
    ) -> AuditLogEntry:
        """Append a cryptographically chained entry to the audit log."""
        entry_id = f"AUDIT-{uuid4().hex[:12].upper()}"
        ts_utc = datetime.now(timezone.utc)

        payload = {
            "entry_id": entry_id,
            "timestamp_utc": ts_utc.isoformat(),
            "action": action,
            "event_id": event.event_id,
            "device_id": event.device_id,
            "event_type": event.event_type.value,
            "event_state": event.event_state.value,
            "source": event.source.value,
            "details": details,
        }

        entry_hash = self._compute_audit_hash(self._last_hash, payload)
        entry = AuditLogEntry(
            entry_id=entry_id,
            timestamp_utc=ts_utc,
            action=action,
            event_id=event.event_id,
            device_id=event.device_id,
            event_type=event.event_type.value,
            event_state=event.event_state.value,
            source=event.source.value,
            details=details,
            prev_hash=self._last_hash,
            entry_hash=entry_hash,
        )

        self._audit_log.append(entry)
        self._last_hash = entry_hash
        return entry

    def process_event(self, event: TamperEvent) -> Tuple[Optional[TamperAlert], bool, str]:
        """
        Process incoming physical tamper event.

        Returns:
            Tuple of:
              - TamperAlert: Created or updated alert (or None if unassociated)
              - bool: True if a new alert was created or updated; False if duplicate
              - str: Processing status message
        """
        with self._lock:
            now_utc = datetime.now(timezone.utc)
            if event.received_timestamp is None:
                event.received_timestamp = now_utc

            # 1. Deduplication check
            event_key = (event.device_id, event.event_type.value, event.sequence_number)
            if event.event_id in self._events or event_key in self._seen_keys:
                logger.warning(
                    f"Duplicate tamper event rejected: event_id={event.event_id}, "
                    f"device={event.device_id}, type={event.event_type.value}, seq={event.sequence_number}"
                )
                self._record_audit_entry(
                    action="DUPLICATE_IGNORED",
                    event=event,
                    details=f"Duplicate event rejected (event_id={event.event_id}, seq={event.sequence_number}).",
                )
                # Find existing matching alert if any
                matching_alert = next(
                    (a for a in self._alerts.values()
                     if a.device_id == event.device_id and a.event_type == event.event_type),
                    None
                )
                return matching_alert, False, "Duplicate event ignored. Original incident preserved."

            # Store event
            self._events[event.event_id] = event
            self._seen_keys.add(event_key)
            if event.device_id not in self._device_events:
                self._device_events[event.device_id] = []
            self._device_events[event.device_id].append(event)

            # Record event in audit ledger
            audit_entry = self._record_audit_entry(
                action="EVENT_RECORDED",
                event=event,
                details=f"Received {event.event_state.value} {event.event_type.value} from {event.source.value}.",
            )

            # Retrieve or initialize device status
            dev_status = self._devices.get(event.device_id)
            if dev_status is None:
                dev_status = DeviceStatus(
                    device_id=event.device_id,
                    tamper_status=DeviceTamperStatus.NOMINAL,
                    active_alerts_count=0,
                    active_event_types=[],
                    last_event_timestamp=event.event_timestamp,
                    last_heartbeat_timestamp=event.event_timestamp,
                    highest_sequence_seen=event.sequence_number,
                    total_incidents_recorded=0,
                    is_simulated=(event.source == EventSource.SIMULATED),
                    updated_at_utc=now_utc,
                )
                self._devices[event.device_id] = dev_status

            # Update monotonic sequence and timestamp
            dev_status.highest_sequence_seen = max(dev_status.highest_sequence_seen, event.sequence_number)
            dev_status.last_event_timestamp = event.event_timestamp
            dev_status.updated_at_utc = now_utc
            if event.source == EventSource.SIMULATED:
                dev_status.is_simulated = True

            # 2. Handle Event State: ACTIVE vs CLEARED
            if event.event_state == EventState.ACTIVE:
                dev_status.total_incidents_recorded += 1
                severity, category, is_confirmed = EVENT_SEVERITY_MAP[event.event_type]

                # Check if an ACTIVE alert of this type already exists for this device
                existing_alert = next(
                    (a for a in self._alerts.values()
                     if a.device_id == event.device_id and a.event_type == event.event_type and a.status == "ACTIVE"),
                    None
                )

                if existing_alert:
                    # Update without duplicate alert creation
                    existing_alert.originating_event_id = event.event_id
                    existing_alert.audit_hash = audit_entry.entry_hash
                    logger.info(f"Updated existing active alert {existing_alert.alert_id} for device {event.device_id}")
                    return existing_alert, False, "Existing active alert updated with new event sequence."

                # Create new high-priority regulator alert
                alert_id = f"ALT-{event.device_id}-{event.event_type.value}-{uuid4().hex[:6].upper()}"
                sim_tag = "[SIMULATED] " if event.source == EventSource.SIMULATED else ""
                summary = f"{sim_tag}Physical Tamper Alert: {EVENT_DESCRIPTIONS[event.event_type]} on device {event.device_id}"

                details = (
                    "Security Notice: Hardware switch or tamper circuit indicated potential physical interference. "
                    "Notice to Regulator: Inspect unit mounting and physical seals. "
                    "(Note: This alert denotes physical sensor interference, not automated fraud or discharge violation)."
                    if is_confirmed else
                    "Connectivity Warning: Unexpected communication loss. Inspect network or power supply. "
                    "Unconfirmed physical tampering."
                )

                alert = TamperAlert(
                    alert_id=alert_id,
                    device_id=event.device_id,
                    sensor_id=event.sensor_id,
                    originating_event_id=event.event_id,
                    event_type=event.event_type,
                    severity=severity,
                    category=category,
                    is_confirmed_tamper=is_confirmed,
                    status="ACTIVE",
                    summary=summary,
                    details=details,
                    created_at_utc=now_utc,
                    cleared_at_utc=None,
                    source=event.source,
                    is_simulated=(event.source == EventSource.SIMULATED),
                    audit_hash=audit_entry.entry_hash,
                )
                self._alerts[alert_id] = alert

                # Update device tamper status
                if is_confirmed:
                    dev_status.tamper_status = DeviceTamperStatus.TAMPER_ACTIVE
                elif dev_status.tamper_status != DeviceTamperStatus.TAMPER_ACTIVE:
                    dev_status.tamper_status = DeviceTamperStatus.OFFLINE_WARNING

                if event.event_type.value not in dev_status.active_event_types:
                    dev_status.active_event_types.append(event.event_type.value)
                dev_status.active_alerts_count = len(dev_status.active_event_types)

                self._record_audit_entry(
                    action="ALERT_GENERATED",
                    event=event,
                    details=f"Generated {severity.value} alert {alert_id} ({summary}).",
                )
                logger.warning(f"Generated physical tamper alert: {alert_id} ({summary})")
                return alert, True, "Physical tamper alert successfully generated and logged."

            elif event.event_state == EventState.CLEARED:
                # Find and clear active alerts for this event type
                cleared_alerts = [
                    a for a in self._alerts.values()
                    if a.device_id == event.device_id and a.event_type == event.event_type and a.status == "ACTIVE"
                ]

                last_cleared_alert = None
                for alert in cleared_alerts:
                    alert.status = "CLEARED"
                    alert.cleared_at_utc = now_utc
                    last_cleared_alert = alert

                # Update device active events list
                if event.event_type.value in dev_status.active_event_types:
                    dev_status.active_event_types.remove(event.event_type.value)
                dev_status.active_alerts_count = len(dev_status.active_event_types)

                # Re-evaluate device status
                remaining_types = [TamperEventType(t) for t in dev_status.active_event_types]
                has_active_physical = any(
                    EVENT_SEVERITY_MAP[t][2] for t in remaining_types
                )
                has_active_offline = any(
                    t == TamperEventType.DEVICE_OFFLINE for t in remaining_types
                )

                if has_active_physical:
                    dev_status.tamper_status = DeviceTamperStatus.TAMPER_ACTIVE
                elif has_active_offline:
                    dev_status.tamper_status = DeviceTamperStatus.OFFLINE_WARNING
                else:
                    dev_status.tamper_status = DeviceTamperStatus.NOMINAL

                self._record_audit_entry(
                    action="ALERT_CLEARED",
                    event=event,
                    details=(
                        f"Cleared condition for {event.event_type.value}. "
                        f"Historical alert {last_cleared_alert.alert_id if last_cleared_alert else 'N/A'} preserved in audit ledger."
                    ),
                )
                logger.info(f"Cleared tamper condition for {event.event_type.value} on {event.device_id}")
                return last_cleared_alert, True, "Tamper condition marked CLEARED. Complete incident history preserved."

            return None, False, "Unknown event state."

    def get_device_status(self, device_id: str) -> Optional[DeviceStatus]:
        """Retrieve real-time security status for a specific monitoring unit."""
        with self._lock:
            status = self._devices.get(device_id)
            return status.model_copy() if status else None

    def get_alerts(
        self,
        status_filter: Optional[str] = None,
        severity_filter: Optional[str] = None,
        device_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[TamperAlert]:
        """
        Query regulator security alerts with optional filtering.
        Preserves all historical alerts (both ACTIVE and CLEARED).
        """
        with self._lock:
            results = list(self._alerts.values())

            if status_filter:
                s_norm = status_filter.upper()
                results = [a for a in results if a.status == s_norm]

            if severity_filter:
                sev_norm = severity_filter.upper()
                results = [a for a in results if a.severity.value == sev_norm]

            if device_id:
                results = [a for a in results if a.device_id == device_id]

            if source_filter:
                src_norm = source_filter.upper()
                results = [a for a in results if a.source.value == src_norm]

            # Return sorted by creation timestamp descending (newest first)
            results.sort(key=lambda a: a.created_at_utc, reverse=True)
            return [a.model_copy() for a in results]

    def get_audit_log(
        self,
        device_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Retrieve cryptographically chained audit history for regulatory inspection.
        """
        with self._lock:
            logs = self._audit_log
            if device_id:
                logs = [entry for entry in logs if entry.device_id == device_id]
            # Return newest entries first
            return [entry.model_copy() for entry in reversed(logs[-limit:])]

    def verify_audit_log_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Cryptographically verify the entire audit log hash chain.
        Returns (True, None) if completely valid, or (False, error_reason) if broken.
        """
        with self._lock:
            expected_prev = self.GENESIS_HASH
            for idx, entry in enumerate(self._audit_log):
                if entry.prev_hash != expected_prev:
                    return False, f"Broken chain at entry index {idx}: expected prev_hash {expected_prev}, got {entry.prev_hash}"

                payload = {
                    "entry_id": entry.entry_id,
                    "timestamp_utc": entry.timestamp_utc.isoformat(),
                    "action": entry.action,
                    "event_id": entry.event_id,
                    "device_id": entry.device_id,
                    "event_type": entry.event_type,
                    "event_state": entry.event_state,
                    "source": entry.source,
                    "details": entry.details,
                }
                computed = self._compute_audit_hash(expected_prev, payload)
                if computed != entry.entry_hash:
                    return False, f"Hash mismatch at entry {entry.entry_id}: expected {computed}, got {entry.entry_hash}"

                expected_prev = entry.entry_hash

            return True, None

    def clear_state(self):
        """Reset in-memory state. Used strictly for test isolation."""
        with self._lock:
            self._events.clear()
            self._device_events.clear()
            self._alerts.clear()
            self._devices.clear()
            self._audit_log.clear()
            self._seen_keys.clear()
            self._last_hash = self.GENESIS_HASH


# Global singleton instance
tamper_manager = TamperAlarmManager()


# ==============================================================================
# 4. Hardware Simulation Helper
# ==============================================================================

class TamperEventSimulator:
    """
    Clearly labelled simulation mechanism for hardware demonstration.
    Generates realistic tamper events stamped with source=SIMULATED.
    """

    _sim_sequence: int = 1

    @classmethod
    def create_simulated_event(
        cls,
        scenario: str,
        device_id: str = "DEV-SIM-001",
        sensor_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TamperEvent:
        """
        Create a simulated tamper event for demonstration.

        Supported scenarios:
          - 'enclosure_open'
          - 'seal_broken'
          - 'sensor_disconnected'
          - 'sensor_removed'
          - 'device_offline'
          - 'clear'
        """
        now_utc = datetime.now(timezone.utc)
        seq = cls._sim_sequence
        cls._sim_sequence += 1
        meta = metadata.copy() if metadata else {}
        meta["simulation_demonstration"] = True

        sc_norm = scenario.strip().lower()

        if sc_norm == "enclosure_open":
            meta["microswitch_pin"] = "GPIO_17"
            meta["contact_state"] = "OPEN"
            return TamperEvent(
                event_id=f"EVT-SIM-ENC-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=sensor_id,
                event_type=TamperEventType.ENCLOSURE_OPEN,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.ACTIVE,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        elif sc_norm == "seal_broken":
            meta["seal_loop_resistance_ohms"] = "INF_OPEN_CIRCUIT"
            meta["seal_serial_tag"] = "SEAL-ONT-2024-998"
            return TamperEvent(
                event_id=f"EVT-SIM-SEAL-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=sensor_id,
                event_type=TamperEventType.SEAL_BROKEN,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.ACTIVE,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        elif sc_norm == "sensor_disconnected":
            s_id = sensor_id or "COD-PROBE-01"
            meta["loop_current_ma"] = 0.0
            meta["namur_fault"] = "NE43_UNDERRANGE_BREAK"
            return TamperEvent(
                event_id=f"EVT-SIM-DISC-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=s_id,
                event_type=TamperEventType.SENSOR_DISCONNECTED,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.ACTIVE,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        elif sc_norm == "sensor_removed":
            s_id = sensor_id or "COD-PROBE-01"
            meta["mounting_bracket_interlock"] = "DISENGAGED"
            meta["immersion_depth_cm"] = 0.0
            return TamperEvent(
                event_id=f"EVT-SIM-REM-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=s_id,
                event_type=TamperEventType.SENSOR_REMOVED,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.ACTIVE,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        elif sc_norm == "device_offline":
            meta["missed_heartbeats"] = 3
            meta["watchdog_timer_seconds"] = 180
            meta["note"] = "Connectivity warning; possible tampering (unconfirmed)."
            return TamperEvent(
                event_id=f"EVT-SIM-OFF-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=None,
                event_type=TamperEventType.DEVICE_OFFLINE,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.ACTIVE,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        elif sc_norm in ["clear", "cleared"]:
            # Default to clearing enclosure if not specified
            clear_type = TamperEventType.ENCLOSURE_OPEN
            if metadata and "target_event_type" in metadata:
                clear_type = TamperEventType(metadata["target_event_type"])
            meta["operator_id"] = "REGULATOR-INSPECTOR-007"
            meta["resolution_note"] = "Physical unit inspected and resealed."
            return TamperEvent(
                event_id=f"EVT-SIM-CLR-{uuid4().hex[:8]}",
                device_id=device_id,
                sensor_id=sensor_id,
                event_type=clear_type,
                event_timestamp=now_utc,
                received_timestamp=now_utc,
                sequence_number=seq,
                event_state=EventState.CLEARED,
                source=EventSource.SIMULATED,
                metadata=meta,
            )

        else:
            raise ValueError(
                f"Unknown simulation scenario '{scenario}'. Supported: "
                "'enclosure_open', 'seal_broken', 'sensor_disconnected', 'sensor_removed', 'device_offline', 'clear'"
            )

