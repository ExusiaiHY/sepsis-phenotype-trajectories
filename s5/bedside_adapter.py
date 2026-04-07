"""
bedside_adapter.py - Prospective bedside feed replay, audit, and stage-data adapters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from s4.treatment_aware_model import _load_or_build_splits
from s5.bedside_service import BedsideMonitoringService
from s5.realtime_model import _align_stage5_inputs
from s5.silent_deployment import _derive_active_hours

FEED_CONTEXT_FIELDS = (
    "label",
    "center_id",
    "icu_type",
    "data_source",
    "set_name",
    "anchor_time_type",
)

_CORE_EVENT_FIELDS = {
    "action",
    "patient_id",
    "timestamp",
    "step_index",
    "hour_index",
    "values",
    "mask",
    "treatments",
    "treatment_mask",
    "note_embedding",
    "context",
}


class BedsideFeedAdapter:
    """Replay bedside updates through the runtime service with audit logging."""

    def __init__(
        self,
        *,
        service: BedsideMonitoringService,
        audit_jsonl_path: Path | None = None,
        dashboard_dir: Path | None = None,
        render_dashboard_on_alert: bool = False,
        render_final_dashboards: bool = False,
    ):
        self.service = service
        self.audit_jsonl_path = None if audit_jsonl_path is None else Path(audit_jsonl_path)
        if self.audit_jsonl_path is not None:
            self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_jsonl_path.write_text("", encoding="utf-8")
        self.dashboard_dir = None if dashboard_dir is None else Path(dashboard_dir)
        if self.dashboard_dir is not None:
            self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.render_dashboard_on_alert = bool(render_dashboard_on_alert)
        self.render_final_dashboards = bool(render_final_dashboards)
        self.policy_name = None
        if isinstance(self.service.policy_artifact, dict):
            policy = self.service.policy_artifact.get("policy", {})
            if isinstance(policy, dict):
                self.policy_name = policy.get("policy_name")

        self.audit_rows: list[dict] = []
        self.alert_rows: list[dict] = []
        self.dashboard_paths: list[str] = []
        self.patient_state: dict[str, dict] = {}
        self.patient_context: dict[str, dict] = {}
        self._audit_handle = None

    def replay_events(
        self,
        *,
        events: Iterable[dict],
        output_dir: Path | None = None,
    ) -> dict:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        if self.audit_jsonl_path is not None:
            self._audit_handle = open(self.audit_jsonl_path, "a", encoding="utf-8")
        try:
            for event_index, event in enumerate(events, start=1):
                self.process_event(event=event, event_index=event_index)
        finally:
            if self._audit_handle is not None:
                self._audit_handle.close()
                self._audit_handle = None

        if self.render_final_dashboards and self.dashboard_dir is not None:
            for patient in self.service.list_patients():
                patient_id = str(patient["patient_id"])
                output_path = self.dashboard_dir / f"{_safe_token(patient_id)}_final.html"
                rendered = self.service.render_patient_dashboard(patient_id, output_path=output_path)
                self.dashboard_paths.append(rendered["output_path"])

        summary = self._build_summary()
        artifacts = {}
        if output_dir is not None:
            artifacts = self._write_artifacts(output_dir=output_dir, summary=summary)
        summary["artifacts"] = artifacts
        if output_dir is not None:
            summary_path = output_dir / "bedside_replay_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
            summary["artifacts"]["summary_json"] = str(summary_path)
        return _json_safe(summary)

    def process_event(self, *, event: dict, event_index: int) -> dict:
        payload = _normalize_event_payload(event)
        patient_id = str(payload["patient_id"])
        action = str(payload["action"]).lower()
        context = dict(payload.get("context") or {})
        if context:
            merged = dict(self.patient_context.get(patient_id, {}))
            merged.update({str(key): _json_safe(value) for key, value in context.items() if value is not None})
            self.patient_context[patient_id] = merged

        if patient_id not in self.patient_state:
            self.patient_state[patient_id] = {
                "patient_id": patient_id,
                "n_events": 0,
                "n_updates": 0,
                "n_resets": 0,
                "n_alert_events": 0,
                "ever_alert_event": False,
                "active_session": False,
                "last_event_index": 0,
                "last_action": None,
                "last_summary": {},
                "last_snapshot": {},
                "last_removed": False,
            }
        patient_state = self.patient_state[patient_id]
        patient_state["n_events"] += 1
        patient_state["last_event_index"] = int(event_index)
        patient_state["last_action"] = action

        audit_row = {
            "event_index": int(event_index),
            "action": action,
            "patient_id": patient_id,
            "timestamp": payload.get("timestamp"),
            "step_index": payload.get("step_index"),
            "hour_index": payload.get("hour_index"),
            "policy_name": self.policy_name,
            "context": context,
        }
        passthrough = {
            str(key): _json_safe(value)
            for key, value in payload.items()
            if key not in _CORE_EVENT_FIELDS and value is not None
        }
        if passthrough:
            audit_row["event_meta"] = passthrough

        if action == "update":
            response = self.service.ingest_update(
                patient_id=patient_id,
                values=payload["values"],
                mask=payload["mask"],
                treatments=payload["treatments"],
                treatment_mask=payload["treatment_mask"],
                note_embedding=payload.get("note_embedding"),
            )
            snapshot = dict(response["snapshot"])
            summary = dict(response["summary"])
            dashboard_path = None
            if self.render_dashboard_on_alert and snapshot.get("alert_event") and self.dashboard_dir is not None:
                output_path = self.dashboard_dir / (
                    f"{_safe_token(patient_id)}_event{int(event_index):06d}_h{int(snapshot['hours_seen']):03d}.html"
                )
                rendered = self.service.render_patient_dashboard(patient_id, output_path=output_path)
                dashboard_path = rendered["output_path"]
                self.dashboard_paths.append(dashboard_path)
            audit_row.update(
                {
                    "snapshot": snapshot,
                    "summary": summary,
                    "dashboard_path": dashboard_path,
                }
            )
            patient_state["n_updates"] += 1
            patient_state["active_session"] = True
            patient_state["last_removed"] = False
            patient_state["last_snapshot"] = snapshot
            patient_state["last_summary"] = summary
            if snapshot.get("alert_event"):
                patient_state["n_alert_events"] += 1
                patient_state["ever_alert_event"] = True
                alert_row = self._alert_row(event_index=event_index, patient_id=patient_id, snapshot=snapshot, dashboard_path=dashboard_path)
                self.alert_rows.append(alert_row)
        elif action == "reset":
            response = self.service.reset_patient(patient_id)
            audit_row["reset"] = response
            patient_state["n_resets"] += 1
            patient_state["active_session"] = False
            patient_state["last_removed"] = bool(response.get("removed"))
            patient_state["last_summary"] = {
                **patient_state.get("last_summary", {}),
                "patient_id": patient_id,
                "hours_seen": int(patient_state.get("last_summary", {}).get("hours_seen", 0)),
            }
        else:
            raise ValueError(f"Unsupported action: {action}")

        self.audit_rows.append(_json_safe(audit_row))
        self._write_audit_row(audit_row)
        return _json_safe(audit_row)

    def _alert_row(
        self,
        *,
        event_index: int,
        patient_id: str,
        snapshot: dict,
        dashboard_path: str | None,
    ) -> dict:
        row = {
            "event_index": int(event_index),
            "patient_id": patient_id,
            "hours_seen": int(snapshot.get("hours_seen", 0)),
            "risk_probability": float(snapshot.get("risk_probability", 0.0)),
            "alerts_emitted": int(snapshot.get("alerts_emitted", 0)),
            "first_alert_hour": snapshot.get("first_alert_hour"),
            "top_treatment_signal": snapshot.get("top_treatment_signal"),
            "deployment_ready": bool(snapshot.get("deployment_ready", False)),
            "policy_name": snapshot.get("policy_name", self.policy_name),
            "dashboard_path": dashboard_path,
        }
        row.update({f"context_{key}": value for key, value in self.patient_context.get(patient_id, {}).items()})
        return _json_safe(row)

    def _build_summary(self) -> dict:
        patient_summary = self._patient_summary_frame()
        return {
            "policy_name": self.policy_name,
            "n_events": int(len(self.audit_rows)),
            "n_updates": int(sum(int(row.get("action") == "update") for row in self.audit_rows)),
            "n_resets": int(sum(int(row.get("action") == "reset") for row in self.audit_rows)),
            "n_patients_seen": int(len(self.patient_state)),
            "n_active_patients_final": int(sum(int(bool(row.get("active_session"))) for row in self.patient_state.values())),
            "n_alert_events": int(len(self.alert_rows)),
            "patients_with_alert_events": int(sum(int(bool(row.get("ever_alert_event"))) for row in self.patient_state.values())),
            "dashboards_rendered": int(len(self.dashboard_paths)),
            "active_patients": [row["patient_id"] for row in self.service.list_patients()],
            "patient_preview": patient_summary.head(10).to_dict(orient="records"),
        }

    def _write_artifacts(self, *, output_dir: Path, summary: dict) -> dict:
        audit_frame = self._audit_frame()
        patient_frame = self._patient_summary_frame()
        alert_frame = pd.DataFrame(self.alert_rows)

        audit_csv = output_dir / "bedside_audit.csv"
        patient_csv = output_dir / "bedside_patient_summary.csv"
        alert_csv = output_dir / "bedside_alert_events.csv"
        audit_frame.to_csv(audit_csv, index=False)
        patient_frame.to_csv(patient_csv, index=False)
        if alert_frame.empty:
            alert_frame = pd.DataFrame(
                columns=[
                    "event_index",
                    "patient_id",
                    "hours_seen",
                    "risk_probability",
                    "alerts_emitted",
                    "first_alert_hour",
                    "top_treatment_signal",
                    "deployment_ready",
                    "policy_name",
                    "dashboard_path",
                ]
            )
        alert_frame.to_csv(alert_csv, index=False)
        return {
            "audit_jsonl": None if self.audit_jsonl_path is None else str(self.audit_jsonl_path),
            "audit_csv": str(audit_csv),
            "patient_summary_csv": str(patient_csv),
            "alert_events_csv": str(alert_csv),
            "dashboard_dir": None if self.dashboard_dir is None else str(self.dashboard_dir),
            "n_patient_rows": int(len(patient_frame)),
            "n_alert_rows": int(len(alert_frame)),
            "n_audit_rows": int(len(audit_frame)),
            "n_dashboards": int(summary["dashboards_rendered"]),
        }

    def _audit_frame(self) -> pd.DataFrame:
        rows = []
        for row in self.audit_rows:
            flat = {
                "event_index": row.get("event_index"),
                "action": row.get("action"),
                "patient_id": row.get("patient_id"),
                "timestamp": row.get("timestamp"),
                "step_index": row.get("step_index"),
                "hour_index": row.get("hour_index"),
                "policy_name": row.get("policy_name"),
                "dashboard_path": row.get("dashboard_path"),
            }
            snapshot = row.get("snapshot") or {}
            summary = row.get("summary") or {}
            reset = row.get("reset") or {}
            flat.update(
                {
                    "risk_probability": snapshot.get("risk_probability"),
                    "risk_alert": snapshot.get("risk_alert"),
                    "alert_event": snapshot.get("alert_event"),
                    "alerts_emitted": snapshot.get("alerts_emitted"),
                    "deployment_ready": snapshot.get("deployment_ready"),
                    "top_treatment_signal": snapshot.get("top_treatment_signal"),
                    "hours_seen": snapshot.get("hours_seen", summary.get("hours_seen")),
                    "summary_hours_seen": summary.get("hours_seen"),
                    "summary_n_snapshots": summary.get("n_snapshots"),
                    "reset_removed": reset.get("removed"),
                }
            )
            for key, value in (row.get("context") or {}).items():
                flat[f"context_{key}"] = value
            for key, value in (row.get("event_meta") or {}).items():
                flat[f"meta_{key}"] = value
            rows.append(flat)
        return pd.DataFrame(rows)

    def _patient_summary_frame(self) -> pd.DataFrame:
        rows = []
        for patient_id, state in sorted(self.patient_state.items()):
            last_summary = state.get("last_summary") or {}
            last_snapshot = state.get("last_snapshot") or {}
            row = {
                "patient_id": patient_id,
                "n_events": int(state["n_events"]),
                "n_updates": int(state["n_updates"]),
                "n_resets": int(state["n_resets"]),
                "n_alert_events": int(state["n_alert_events"]),
                "ever_alert_event": bool(state["ever_alert_event"]),
                "active_session": bool(state["active_session"]),
                "last_removed": bool(state["last_removed"]),
                "last_action": state.get("last_action"),
                "last_event_index": int(state["last_event_index"]),
                "hours_seen": int(last_summary.get("hours_seen", last_snapshot.get("hours_seen", 0))),
                "risk_probability": last_snapshot.get("risk_probability", last_summary.get("risk_probability")),
                "risk_alert": bool(last_snapshot.get("risk_alert", last_summary.get("risk_alert", False))),
                "alerts_emitted": int(last_snapshot.get("alerts_emitted", last_summary.get("alerts_emitted", 0))),
                "deployment_ready": bool(last_snapshot.get("deployment_ready", last_summary.get("deployment_ready", False))),
                "top_treatment_signal": last_summary.get("top_treatment_signal", last_snapshot.get("top_treatment_signal")),
            }
            row.update({f"context_{key}": value for key, value in self.patient_context.get(patient_id, {}).items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def _write_audit_row(self, audit_row: dict) -> None:
        if self.audit_jsonl_path is None:
            return
        line = json.dumps(_json_safe(audit_row), ensure_ascii=False)
        if self._audit_handle is not None:
            self._audit_handle.write(line)
            self._audit_handle.write("\n")
            self._audit_handle.flush()
            return
        with open(self.audit_jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def replay_bedside_feed(
    *,
    service: BedsideMonitoringService,
    events: Iterable[dict],
    output_dir: Path,
    audit_jsonl_path: Path | None = None,
    dashboard_dir: Path | None = None,
    render_dashboard_on_alert: bool = False,
    render_final_dashboards: bool = False,
) -> dict:
    adapter = BedsideFeedAdapter(
        service=service,
        audit_jsonl_path=audit_jsonl_path,
        dashboard_dir=dashboard_dir,
        render_dashboard_on_alert=render_dashboard_on_alert,
        render_final_dashboards=render_final_dashboards,
    )
    return adapter.replay_events(events=events, output_dir=output_dir)


def load_bedside_feed_jsonl(feed_path: Path) -> Iterator[dict]:
    with open(Path(feed_path), encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Feed line {line_number} is not a JSON object")
            yield payload


def tee_bedside_feed_jsonl(feed_path: Path, events: Iterable[dict]) -> Iterator[dict]:
    path = Path(feed_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(_json_safe(event), ensure_ascii=False))
            f.write("\n")
            yield event


def write_bedside_feed_jsonl(feed_path: Path, events: Iterable[dict]) -> int:
    count = 0
    for count, _ in enumerate(tee_bedside_feed_jsonl(feed_path, events), start=1):
        pass
    return int(count)


def iter_stage5_bedside_feed(
    *,
    s0_dir: Path,
    treatment_dir: Path,
    note_embeddings_path: Path | None = None,
    label_col: str = "mortality_inhospital",
    split: str = "test",
    max_patients: int | None = None,
    max_events: int | None = None,
    order: str = "round_robin",
) -> Iterator[dict]:
    s0_dir = Path(s0_dir)
    treatment_dir = Path(treatment_dir)
    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks_cont = np.load(s0_dir / "processed" / "masks_continuous.npy")
    treatments = np.load(treatment_dir / "treatments.npy")
    masks_treat = np.load(treatment_dir / "masks_treatments.npy")
    s0_static = pd.read_csv(s0_dir / "static.csv")
    treatment_static_path = treatment_dir / "cohort_static.csv"
    if treatment_static_path.exists():
        treatment_static = pd.read_csv(treatment_static_path)
    else:
        treatment_static = s0_static[["patient_id"]].copy()

    notes = None
    if note_embeddings_path is not None and Path(note_embeddings_path).exists():
        notes = np.load(note_embeddings_path)

    aligned = _align_stage5_inputs(
        s0_static=s0_static,
        treatment_static=treatment_static,
        continuous=continuous,
        masks_continuous=masks_cont,
        treatments=treatments,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )
    continuous = aligned["continuous"]
    masks_cont = aligned["masks_continuous"]
    treatments = aligned["treatments"]
    masks_treat = aligned["masks_treatments"]
    static = aligned["s0_static"]
    notes = aligned["note_embeddings"]

    label_frame = static if label_col in static.columns else aligned["treatment_static"]
    labels = label_frame[label_col].fillna(0).astype(int).to_numpy()
    indices = _load_requested_split_indices(
        split=str(split),
        n_samples=len(labels),
        labels=labels,
        splits_path=s0_dir / "splits.json",
        seed=42,
    )
    if max_patients is not None:
        indices = indices[: max(1, int(max_patients))]

    continuous = continuous[indices].astype(np.float32, copy=False)
    masks_cont = masks_cont[indices].astype(np.float32, copy=False)
    treatments = treatments[indices].astype(np.float32, copy=False)
    masks_treat = masks_treat[indices].astype(np.float32, copy=False)
    static = static.iloc[indices].reset_index(drop=True)
    labels = labels[indices]
    if notes is not None:
        notes = np.asarray(notes[indices], dtype=np.float32)
        if notes.ndim == 2:
            notes = np.repeat(notes[:, None, :], continuous.shape[1], axis=1)

    active_hours = _derive_active_hours(
        static=static,
        masks_continuous=masks_cont,
        masks_treatments=masks_treat,
        note_embeddings=notes,
    )

    if str(order) not in {"round_robin", "patient_major"}:
        raise ValueError("order must be one of {'round_robin', 'patient_major'}")

    event_count = 0
    if str(order) == "patient_major":
        for patient_idx in range(len(static)):
            for hour_idx in range(int(active_hours[patient_idx])):
                event_count += 1
                if max_events is not None and event_count > int(max_events):
                    return
                yield _build_stage_event(
                    static=static,
                    labels=labels,
                    continuous=continuous,
                    masks_cont=masks_cont,
                    treatments=treatments,
                    masks_treat=masks_treat,
                    notes=notes,
                    patient_idx=patient_idx,
                    hour_idx=hour_idx,
                    event_index=event_count,
                    order=str(order),
                )
        return

    seq_len = int(continuous.shape[1])
    for hour_idx in range(seq_len):
        for patient_idx in range(len(static)):
            if int(active_hours[patient_idx]) <= hour_idx:
                continue
            event_count += 1
            if max_events is not None and event_count > int(max_events):
                return
            yield _build_stage_event(
                static=static,
                labels=labels,
                continuous=continuous,
                masks_cont=masks_cont,
                treatments=treatments,
                masks_treat=masks_treat,
                notes=notes,
                patient_idx=patient_idx,
                hour_idx=hour_idx,
                event_index=event_count,
                order=str(order),
            )


def _build_stage_event(
    *,
    static: pd.DataFrame,
    labels: np.ndarray,
    continuous: np.ndarray,
    masks_cont: np.ndarray,
    treatments: np.ndarray,
    masks_treat: np.ndarray,
    notes: np.ndarray | None,
    patient_idx: int,
    hour_idx: int,
    event_index: int,
    order: str,
) -> dict:
    patient_row = static.iloc[patient_idx]
    patient_id = str(patient_row["patient_id"])
    context = {"label": int(labels[patient_idx])}
    for field in FEED_CONTEXT_FIELDS[1:]:
        if field not in static.columns:
            continue
        value = patient_row[field]
        context[field] = None if pd.isna(value) else value
    event = {
        "action": "update",
        "patient_id": patient_id,
        "step_index": int(event_index),
        "hour_index": int(hour_idx + 1),
        "feed_order": str(order),
        "patient_position": int(patient_idx),
        "values": continuous[patient_idx, hour_idx].tolist(),
        "mask": masks_cont[patient_idx, hour_idx].tolist(),
        "treatments": treatments[patient_idx, hour_idx].tolist(),
        "treatment_mask": masks_treat[patient_idx, hour_idx].tolist(),
        "context": _json_safe(context),
    }
    if notes is not None:
        event["note_embedding"] = notes[patient_idx, hour_idx].tolist()
    return event


def _load_requested_split_indices(
    *,
    split: str,
    n_samples: int,
    labels: np.ndarray,
    splits_path: Path,
    seed: int,
) -> np.ndarray:
    splits_path = Path(splits_path)
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and split in raw:
            arr = np.asarray(raw[split], dtype=int)
            arr = arr[(arr >= 0) & (arr < int(n_samples))]
            if len(arr) > 0:
                return arr
    rebuilt = _load_or_build_splits(
        n_samples=n_samples,
        labels=labels,
        splits_path=splits_path,
        seed=seed,
    )
    return np.asarray(rebuilt[split], dtype=int)


def _normalize_event_payload(event: dict) -> dict:
    if not isinstance(event, dict):
        raise ValueError("bedside feed event must be a JSON object")
    payload = dict(event)
    payload["patient_id"] = str(payload.get("patient_id", "")).strip()
    if not payload["patient_id"]:
        raise ValueError("bedside feed event must include patient_id")
    payload["action"] = str(payload.get("action", "update")).lower()
    if payload["action"] not in {"update", "reset"}:
        raise ValueError("bedside feed event action must be 'update' or 'reset'")
    if payload["action"] == "update":
        for field in ("values", "mask", "treatments", "treatment_mask"):
            if field not in payload:
                raise ValueError(f"bedside feed update missing required field: {field}")
    context = payload.get("context", {})
    if context is None:
        payload["context"] = {}
    elif not isinstance(context, dict):
        raise ValueError("bedside feed event context must be a JSON object when provided")
    return payload


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return [_json_safe(value) for value in payload.tolist()]
    if isinstance(payload, (np.bool_, bool)):
        return bool(payload)
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating, float)):
        value = float(payload)
        return None if np.isnan(value) else value
    return payload
