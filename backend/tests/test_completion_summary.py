import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import router
from app.database import Base, get_db


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # JSONB not available on SQLite — compile as JSON (same shim as test_state_machine)
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_sqlite(_type, compiler, **kw):
        return "JSON"

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


def auth_headers(client, username, password):
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def make_run_with_data(client, headers, metrics=2, artifacts=1):
    dataset_sha = sha("dataset")
    code_sha = "abc1234"
    res = client.post(
        "/api/runs",
        json={
            "project": "p1",
            "name": "n1",
            "dataset_content_sha256": dataset_sha,
            "code_commit_sha": code_sha,
            "expected_version": 0,
        },
        headers=headers,
    )
    assert res.status_code == 201
    run = res.json()
    version = run["version"]
    for i in range(metrics):
        res = client.post(
            f"/api/runs/{run['id']}/metrics",
            json={"name": "loss", "value": 0.5, "step": i, "expected_version": version},
            headers=headers,
        )
        assert res.status_code == 200
        version = res.json()["version"]
    for i in range(artifacts):
        res = client.post(
            f"/api/runs/{run['id']}/artifacts",
            json={
                "name": f"ckpt-{i}.pt",
                "uri": f"s3://lab/ckpt-{i}.pt",
                "content_sha256": sha(f"artifact-{i}"),
                "expected_version": version,
            },
            headers=headers,
        )
        assert res.status_code == 200
        version = res.json()["version"]
    return run["id"], dataset_sha, code_sha, version


def test_completion_summary_counts_and_fingerprints(client):
    headers = auth_headers(client, "researcher", "lab123456")
    run_id, dataset_sha, code_sha, version = make_run_with_data(client, headers)

    res = client.get(f"/api/runs/{run_id}/completion-summary", headers=headers)
    assert res.status_code == 200
    summary = res.json()
    assert summary["run_id"] == run_id
    assert summary["status"] == "running"
    assert summary["version"] == version
    assert summary["dataset_content_sha256"] == dataset_sha
    assert summary["code_commit_sha"] == code_sha
    assert summary["metrics_count"] == 2
    assert summary["artifacts_count"] == 1

    # 条数要和详情对得上
    detail = client.get(f"/api/runs/{run_id}", headers=headers).json()
    assert summary["metrics_count"] == len(detail["metrics_json"])
    assert summary["artifacts_count"] == len(detail["artifacts_json"])
    assert summary["version"] == detail["version"]


def test_summary_fetch_is_read_only_cancel_keeps_running(client):
    headers = auth_headers(client, "researcher", "lab123456")
    run_id, _ds, _code, version = make_run_with_data(client, headers, metrics=1, artifacts=0)

    events_before = client.get(f"/api/runs/{run_id}/events", headers=headers).json()

    # 打开确认框（取摘要）后取消：不得写入任何事件
    res = client.get(f"/api/runs/{run_id}/completion-summary", headers=headers)
    assert res.status_code == 200

    events_after = client.get(f"/api/runs/{run_id}/events", headers=headers).json()
    assert len(events_after) == len(events_before)
    assert all(e["event_type"] != "RunCompleted" for e in events_after)

    detail = client.get(f"/api/runs/{run_id}", headers=headers).json()
    assert detail["status"] == "running"
    assert detail["version"] == version


def test_confirm_complete_then_counts_match(client):
    headers = auth_headers(client, "researcher", "lab123456")
    run_id, _ds, _code, _version = make_run_with_data(client, headers)

    summary = client.get(f"/api/runs/{run_id}/completion-summary", headers=headers).json()
    res = client.post(
        f"/api/runs/{run_id}/complete",
        json={"result_summary": "done", "expected_version": summary["version"]},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"

    events = client.get(f"/api/runs/{run_id}/events", headers=headers).json()
    assert events[-1]["event_type"] == "RunCompleted"
    assert events[-1]["version"] == summary["version"] + 1

    detail = client.get(f"/api/runs/{run_id}", headers=headers).json()
    assert detail["status"] == "completed"
    assert len(detail["metrics_json"]) == summary["metrics_count"]
    assert len(detail["artifacts_json"]) == summary["artifacts_count"]


def test_auditor_read_only(client):
    headers = auth_headers(client, "researcher", "lab123456")
    run_id, _ds, _code, version = make_run_with_data(client, headers, metrics=1, artifacts=0)

    auditor = auth_headers(client, "auditor", "audit123456")
    res = client.get(f"/api/runs/{run_id}/completion-summary", headers=auditor)
    assert res.status_code == 200

    res = client.post(
        f"/api/runs/{run_id}/complete",
        json={"result_summary": "nope", "expected_version": version},
        headers=auditor,
    )
    assert res.status_code == 403

    res = client.get(f"/api/runs/{run_id}/completion-summary")
    assert res.status_code == 401
