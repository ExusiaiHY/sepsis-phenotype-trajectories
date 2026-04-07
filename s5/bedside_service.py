"""
bedside_service.py - Minimal bedside HTTP service and patient-session manager.
"""
from __future__ import annotations

import json
import logging
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from s5.dashboard import render_clinical_dashboard_html
from s5.deployment_policy import load_policy_artifact
from s5.realtime_model import (
    RealtimePatientBuffer,
    RealtimePhenotypeMonitor,
    load_realtime_student_artifact,
)


@dataclass
class PatientRuntimeState:
    patient_id: str
    buffer: RealtimePatientBuffer
    monitor: RealtimePhenotypeMonitor
    snapshots: deque[dict] = field(default_factory=deque)


class BedsideMonitoringService:
    """Stateful bedside wrapper around the realtime student and deployment policy."""

    def __init__(
        self,
        *,
        model,
        model_config: dict,
        threshold: float,
        temperature: float,
        policy_artifact: dict | None = None,
        phenotype_centroids: np.ndarray | None = None,
        device: str = "cpu",
        max_snapshots: int = 72,
        dashboard_dir: Path | None = None,
        treatment_feature_names: list[str] | None = None,
        model_artifact_path: Path | None = None,
        policy_path: Path | None = None,
    ):
        self.model = model.to(device).eval()
        self.model_config = dict(model_config)
        self.threshold = float(threshold)
        self.temperature = max(float(temperature), 1.0e-3)
        self.policy_artifact = None if policy_artifact is None else dict(policy_artifact)
        self.phenotype_centroids = None if phenotype_centroids is None else np.asarray(phenotype_centroids, dtype=np.float32)
        self.device = str(device)
        self.max_snapshots = max(1, int(max_snapshots))
        self.dashboard_dir = None if dashboard_dir is None else Path(dashboard_dir)
        if self.dashboard_dir is not None:
            self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifact_path = None if model_artifact_path is None else str(model_artifact_path)
        self.policy_path = None if policy_path is None else str(policy_path)

        self.seq_len = int(self.model_config["max_seq_len"])
        self.n_cont_features = int(self.model_config["n_cont_features"])
        self.n_treat_features = int(self.model_config["n_treat_features"])
        self.note_dim = int(self.model_config.get("note_dim", 0))
        self.treatment_feature_names = treatment_feature_names or [f"treatment_{idx}" for idx in range(self.n_treat_features)]
        self._patients: dict[str, PatientRuntimeState] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_artifact_path: Path,
        policy_path: Path | None = None,
        phenotype_centroids_path: Path | None = None,
        treatment_feature_names_path: Path | None = None,
        device: str = "cpu",
        max_snapshots: int = 72,
        dashboard_dir: Path | None = None,
    ) -> "BedsideMonitoringService":
        artifact = load_realtime_student_artifact(model_artifact_path, device=device)
        policy_artifact = None if policy_path is None else load_policy_artifact(policy_path)
        phenotype_centroids = None
        if phenotype_centroids_path is not None:
            phenotype_centroids = np.load(Path(phenotype_centroids_path))
        treatment_feature_names = None
        if treatment_feature_names_path is not None and Path(treatment_feature_names_path).exists():
            with open(Path(treatment_feature_names_path), encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                treatment_feature_names = [str(item) for item in payload]
        return cls(
            model=artifact["model"],
            model_config=artifact["config"],
            threshold=float(artifact["threshold"]),
            temperature=float(artifact["temperature"]),
            policy_artifact=policy_artifact,
            phenotype_centroids=phenotype_centroids,
            device=device,
            max_snapshots=max_snapshots,
            dashboard_dir=dashboard_dir,
            treatment_feature_names=treatment_feature_names,
            model_artifact_path=model_artifact_path,
            policy_path=policy_path,
        )

    def describe(self) -> dict:
        with self._lock:
            return {
                "device": self.device,
                "model_artifact_path": self.model_artifact_path,
                "policy_path": self.policy_path,
                "threshold": round(self.threshold, 4),
                "temperature": round(self.temperature, 4),
                "max_snapshots": self.max_snapshots,
                "n_active_patients": len(self._patients),
                "model": _json_safe(self.model_config),
                "policy": None if self.policy_artifact is None else _json_safe(self.policy_artifact.get("policy", {})),
            }

    def list_patients(self) -> list[dict]:
        with self._lock:
            return sorted((self._patient_summary(state) for state in self._patients.values()), key=lambda item: item["patient_id"])

    def get_patient_record(self, patient_id: str) -> dict:
        with self._lock:
            state = self._require_patient(patient_id)
            return {
                "patient_id": state.patient_id,
                "summary": self._patient_summary(state),
                "snapshots": [_json_safe(snapshot) for snapshot in list(state.snapshots)],
            }

    def ingest_update(
        self,
        *,
        patient_id: str,
        values: np.ndarray | list[float],
        mask: np.ndarray | list[float],
        treatments: np.ndarray | list[float],
        treatment_mask: np.ndarray | list[float],
        note_embedding: np.ndarray | list[float] | None = None,
    ) -> dict:
        with self._lock:
            state = self._patients.get(str(patient_id))
            if state is None:
                state = self._create_patient_state(str(patient_id))
                self._patients[str(patient_id)] = state

            values_arr = self._validate_vector(values, expected=self.n_cont_features, field_name="values")
            mask_arr = self._validate_vector(mask, expected=self.n_cont_features, field_name="mask")
            treatments_arr = self._validate_vector(treatments, expected=self.n_treat_features, field_name="treatments")
            treatment_mask_arr = self._validate_vector(treatment_mask, expected=self.n_treat_features, field_name="treatment_mask")
            note_arr = None
            if self.note_dim > 0:
                note_arr = self._validate_vector(note_embedding, expected=self.note_dim, field_name="note_embedding")
            elif note_embedding is not None:
                raise ValueError("note_embedding was provided but the loaded artifact has note_dim=0")

            state.buffer.update(
                values=values_arr,
                mask=mask_arr,
                treatments=treatments_arr,
                treatment_mask=treatment_mask_arr,
                note_embedding=note_arr,
            )
            snapshot = state.monitor.predict(state.buffer)
            if snapshot is None:
                raise RuntimeError("monitor returned no snapshot after an update")
            snapshot = {
                **snapshot,
                "patient_id": state.patient_id,
                "top_treatment_signal": _dominant_treatment_signal(
                    treatment_values=treatments_arr,
                    treatment_mask=treatment_mask_arr,
                    treatment_feature_names=self.treatment_feature_names,
                ),
            }
            state.snapshots.append(snapshot)
            return {
                "patient_id": state.patient_id,
                "snapshot": _json_safe(snapshot),
                "summary": self._patient_summary(state),
            }

    def reset_patient(self, patient_id: str) -> dict:
        with self._lock:
            removed = self._patients.pop(str(patient_id), None) is not None
            return {"patient_id": str(patient_id), "removed": bool(removed)}

    def render_patient_dashboard(self, patient_id: str, *, output_path: Path | None = None) -> dict:
        with self._lock:
            state = self._require_patient(patient_id)
            if not state.snapshots:
                raise ValueError(f"patient {patient_id} has no snapshots to render")
            resolved_path = self._resolve_dashboard_path(patient_id, output_path=output_path)
            html = render_clinical_dashboard_html(
                patient_id=state.patient_id,
                snapshots=list(state.snapshots),
                output_path=resolved_path,
                model_meta={
                    "model_artifact_path": self.model_artifact_path,
                    "policy_path": self.policy_path,
                    "device": self.device,
                    "threshold": round(self.threshold, 4),
                    "temperature": round(self.temperature, 4),
                    "policy": None if self.policy_artifact is None else _json_safe(self.policy_artifact.get("policy", {})),
                    "model": _json_safe(self.model_config),
                },
            )
            return {
                "patient_id": state.patient_id,
                "output_path": str(resolved_path),
                "html": html,
            }

    def _create_patient_state(self, patient_id: str) -> PatientRuntimeState:
        buffer = RealtimePatientBuffer(
            seq_len=self.seq_len,
            n_cont_features=self.n_cont_features,
            n_treat_features=self.n_treat_features,
            note_dim=self.note_dim,
        )
        monitor = RealtimePhenotypeMonitor(
            model=self.model,
            threshold=self.threshold,
            phenotype_centroids=self.phenotype_centroids,
            device=self.device,
            temperature=self.temperature,
            deployment_policy=self.policy_artifact,
        )
        return PatientRuntimeState(
            patient_id=patient_id,
            buffer=buffer,
            monitor=monitor,
            snapshots=deque(maxlen=self.max_snapshots),
        )

    def _patient_summary(self, state: PatientRuntimeState) -> dict:
        latest = dict(state.snapshots[-1]) if state.snapshots else {
            "hours_seen": int(state.buffer.n_updates),
            "risk_alert": False,
            "alert_event": False,
            "alerts_emitted": 0,
            "deployment_ready": False,
        }
        return {
            "patient_id": state.patient_id,
            "hours_seen": int(state.buffer.n_updates),
            "n_snapshots": int(len(state.snapshots)),
            "risk_probability": round(float(latest.get("risk_probability", 0.0)), 4),
            "risk_alert": bool(latest.get("risk_alert", False)),
            "alert_event": bool(latest.get("alert_event", False)),
            "alerts_emitted": int(latest.get("alerts_emitted", 0)),
            "deployment_ready": bool(latest.get("deployment_ready", False)),
            "top_treatment_signal": str(latest.get("top_treatment_signal", "none")),
        }

    def _require_patient(self, patient_id: str) -> PatientRuntimeState:
        state = self._patients.get(str(patient_id))
        if state is None:
            raise KeyError(f"unknown patient_id={patient_id}")
        return state

    def _resolve_dashboard_path(self, patient_id: str, *, output_path: Path | None) -> Path:
        if output_path is not None:
            resolved = Path(output_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved
        safe_patient_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(patient_id))
        if self.dashboard_dir is not None:
            return self.dashboard_dir / f"{safe_patient_id}.html"
        return Path(tempfile.gettempdir()) / f"s5_bedside_{safe_patient_id}.html"

    @staticmethod
    def _validate_vector(value: np.ndarray | list[float] | None, *, expected: int, field_name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{field_name} is required")
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] != int(expected):
            raise ValueError(f"{field_name} must have length {expected}, got {array.shape[0]}")
        return array


def make_bedside_request_handler(service: BedsideMonitoringService):
    """Create a request handler bound to a concrete bedside service instance."""

    class BedsideRequestHandler(BaseHTTPRequestHandler):
        server_version = "S5BedsideHTTP/0.1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                response = dispatch_bedside_request(service=service, method="GET", path=self.path, body=None)
                self._write_response(response)
            except Exception as exc:  # pragma: no cover
                logging.getLogger("s5.bedside").exception("Unhandled GET error")
                self._write_response(_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)))

        def do_POST(self) -> None:  # noqa: N802
            try:
                response = dispatch_bedside_request(
                    service=service,
                    method="POST",
                    path=self.path,
                    body=self._read_body(),
                )
                self._write_response(response)
            except Exception as exc:  # pragma: no cover
                logging.getLogger("s5.bedside").exception("Unhandled POST error")
                self._write_response(_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)))

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            logging.getLogger("s5.bedside").info("%s - %s", self.address_string(), format % args)

        def _read_body(self) -> bytes:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return b""
            return self.rfile.read(content_length)

        def _write_response(self, response: dict) -> None:
            status = int(response["status"])
            body = response["body"]
            content_type = str(response.get("content_type", "application/json; charset=utf-8"))
            headers = response.get("headers", {})
            if isinstance(body, str):
                encoded = body.encode("utf-8")
            else:
                encoded = json.dumps(_json_safe(body), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for key, value in headers.items():
                self.send_header(str(key), str(value))
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return BedsideRequestHandler


def run_bedside_service(
    *,
    service: BedsideMonitoringService,
    host: str = "127.0.0.1",
    port: int = 8085,
) -> ThreadingHTTPServer:
    """Start a threaded bedside HTTP service."""
    server = ThreadingHTTPServer((host, int(port)), make_bedside_request_handler(service))
    logging.getLogger("s5.bedside").info("Serving bedside runtime on http://%s:%s", host, port)
    return server


def dispatch_bedside_request(
    *,
    service: BedsideMonitoringService,
    method: str,
    path: str,
    body: bytes | dict | None,
) -> dict:
    """Pure request dispatcher for HTTP handlers and tests."""
    method = str(method).upper()
    try:
        parsed = urlparse(path)
        if method == "GET" and parsed.path == "/health":
            return _json_response(HTTPStatus.OK, {"status": "ok", "n_active_patients": len(service.list_patients())})
        if method == "GET" and parsed.path == "/config":
            return _json_response(HTTPStatus.OK, service.describe())
        if method == "GET" and parsed.path == "/patients":
            return _json_response(HTTPStatus.OK, {"patients": service.list_patients()})

        patient_id, action = _parse_patient_route(parsed.path)
        if method == "GET" and action is None:
            return _json_response(HTTPStatus.OK, service.get_patient_record(patient_id))
        if method == "GET" and action == "dashboard":
            query = parse_qs(parsed.query)
            output_path = query.get("output_path", [None])[0]
            rendered = service.render_patient_dashboard(
                patient_id,
                output_path=None if output_path is None else Path(output_path),
            )
            return {
                "status": int(HTTPStatus.OK),
                "content_type": "text/html; charset=utf-8",
                "headers": {"X-Dashboard-Path": rendered["output_path"]},
                "body": rendered["html"],
            }
        if method == "POST" and action == "reset":
            return _json_response(HTTPStatus.OK, service.reset_patient(patient_id))
        if method == "POST" and action == "update":
            payload = _decode_json_body(body)
            response = service.ingest_update(
                patient_id=patient_id,
                values=payload.get("values"),
                mask=payload.get("mask"),
                treatments=payload.get("treatments"),
                treatment_mask=payload.get("treatment_mask"),
                note_embedding=payload.get("note_embedding"),
            )
            return _json_response(HTTPStatus.OK, response)
        raise KeyError(f"unsupported route {path}")
    except KeyError as exc:
        return _error_response(HTTPStatus.NOT_FOUND, str(exc))
    except ValueError as exc:
        return _error_response(HTTPStatus.BAD_REQUEST, str(exc))


def _parse_patient_route(path: str) -> tuple[str, str | None]:
    parts = [unquote(part) for part in path.split("/") if part]
    if len(parts) < 2 or parts[0] != "patients":
        raise KeyError(f"unknown route {path}")
    patient_id = parts[1]
    action = None if len(parts) == 2 else parts[2]
    return patient_id, action


def _dominant_treatment_signal(
    *,
    treatment_values: np.ndarray,
    treatment_mask: np.ndarray,
    treatment_feature_names: list[str],
) -> str:
    observed = np.asarray(treatment_mask, dtype=float) > 0
    if not np.any(observed):
        return "none"
    for idx, name in enumerate(treatment_feature_names):
        if idx < len(observed) and observed[idx] and name.endswith("_on") and float(treatment_values[idx]) > 0:
            return str(name)
    masked_values = np.abs(np.asarray(treatment_values, dtype=float)) * observed.astype(float)
    best_idx = int(np.argmax(masked_values))
    if masked_values[best_idx] <= 0:
        return "none"
    return str(treatment_feature_names[best_idx])


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


def _decode_json_body(body: bytes | dict | None) -> dict:
    if body is None or body == b"":
        return {}
    if isinstance(body, dict):
        return dict(body)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _json_response(status: HTTPStatus, body: dict | list) -> dict:
    return {
        "status": int(status),
        "content_type": "application/json; charset=utf-8",
        "headers": {},
        "body": body,
    }


def _error_response(status: HTTPStatus, message: str) -> dict:
    return _json_response(status, {"error": message})
