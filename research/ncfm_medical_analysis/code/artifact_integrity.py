"""Verify immutable hashes declared by NCFM and HoP run manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def verify_record(root: Path, record: object, label: str) -> dict:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"{label} must declare path and sha256")
    path = resolve(root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256(path)
    if record["sha256"] != actual:
        raise ValueError(f"{label} hash mismatch: {path}")
    return {"path": str(path), "sha256": actual}


def verify_run_manifest_integrity(root: Path, payload: dict) -> dict:
    """Verify all mutable inputs and outputs bound by a run manifest."""
    checked = {
        key: verify_record(root, payload.get(key), key)
        for key in ("prepared_manifest", "statistics", "config", "synthetic")
    }
    if payload.get("method") == "HoP-TM":
        selection_record = verify_record(
            root, payload.get("lr_selection"), "lr_selection"
        )
        selection_payload = json.loads(Path(selection_record["path"]).read_text(encoding="utf-8"))
        if selection_payload.get("status") != "complete":
            raise ValueError("lr_selection status must be complete")
        if selection_payload.get("uses_validation_or_test_accuracy") is not False:
            raise ValueError("lr_selection must not use validation or test accuracy")
        selected = selection_payload.get("selected_lr_img")
        contract = payload.get("method_contract", {})
        if selected != contract.get("lr_img"):
            raise ValueError("lr_selection selected_lr_img does not match method_contract")
        if selection_payload.get("selection_rule") != contract.get("lr_selection"):
            raise ValueError("lr_selection rule does not match method_contract")
        attempts = selection_payload.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("lr_selection must contain at least one attempt")
        checked_attempts = []
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise ValueError(f"lr_selection attempt {index} must be an object")
            checked_attempts.append({
                "lr_img": attempt.get("lr_img"),
                "status": attempt.get("status"),
                "stdout": verify_record(root, attempt.get("stdout"), f"lr_selection.attempts[{index}].stdout"),
                "stderr": verify_record(root, attempt.get("stderr"), f"lr_selection.attempts[{index}].stderr"),
            })
        finite = [attempt for attempt in attempts if attempt.get("status") in {"finite_complete", "config_fixed"}]
        if len(finite) != 1 or finite[0].get("lr_img") != selected:
            raise ValueError("lr_selection must identify exactly one selected finite attempt")
        checked["lr_selection"] = {**selection_record, "attempts": checked_attempts}

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("run manifest provenance must be an object")
    checked["provenance"] = {
        key: verify_record(root, provenance.get(key), f"provenance.{key}")
        for key in ("command", "stdout", "stderr")
    }

    source = payload.get("source_provenance")
    source_hashes = source.get("files_sha256") if isinstance(source, dict) else None
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("source_provenance.files_sha256 must be non-empty")
    checked["source_provenance"] = {}
    for relative, expected in source_hashes.items():
        path = resolve(root, str(relative))
        if not path.is_file():
            raise FileNotFoundError(f"source file is missing: {path}")
        actual = sha256(path)
        if expected != actual:
            raise ValueError(f"source file hash mismatch: {path}")
        checked["source_provenance"][str(relative)] = actual

    method = payload.get("method")
    if method == "NCFM":
        pretrained = payload.get("pretrained_dir")
        if not isinstance(pretrained, dict) or not pretrained.get("path"):
            raise ValueError("NCFM manifest must declare pretrained_dir")
        directory = resolve(root, str(pretrained["path"]))
        checked_teachers = {}
        for kind in ("init_sha256", "trained_sha256"):
            hashes = pretrained.get(kind)
            if not isinstance(hashes, dict) or len(hashes) != 20:
                raise ValueError(f"pretrained_dir.{kind} must contain exactly 20 files")
            checked_teachers[kind] = {}
            for name, expected in hashes.items():
                path = directory / name
                if not path.is_file():
                    raise FileNotFoundError(f"teacher checkpoint is missing: {path}")
                actual = sha256(path)
                if expected != actual:
                    raise ValueError(f"teacher checkpoint hash mismatch: {path}")
                checked_teachers[kind][name] = actual
        checked["pretrained_dir"] = checked_teachers
    elif method == "HoP-TM":
        buffer = payload.get("buffer")
        if not isinstance(buffer, dict) or not buffer.get("path"):
            raise ValueError("HoP-TM manifest must declare buffer")
        directory = resolve(root, str(buffer["path"]))
        hashes = buffer.get("trajectory_files")
        if not isinstance(hashes, dict) or len(hashes) != 10:
            raise ValueError("buffer.trajectory_files must contain exactly 10 files")
        checked_buffers = {}
        for name, expected in hashes.items():
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"HoP buffer is missing: {path}")
            actual = sha256(path)
            if expected != actual:
                raise ValueError(f"HoP buffer hash mismatch: {path}")
            checked_buffers[name] = actual
        checked["buffer"] = checked_buffers
    else:
        raise ValueError(f"unsupported run manifest method: {method}")
    return checked
