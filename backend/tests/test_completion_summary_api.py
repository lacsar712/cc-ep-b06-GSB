"""完成确认流程的 API 级核对：

1. 确认框摘要（指纹/度量条数/附件条数/版本）必须来自服务端接口；
2. 取消 = 只取摘要、不发命令 —— 状态仍为进行中，时间线无完成事件；
3. 确认后才变为已完成，时间线出现 RunCompleted，条数与详情一致；
4. 只读账号（auditor）不能执行 CompleteRun。
"""

import hashlib
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles

from app.main import app


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_type, compiler, **kw):  # noqa: ANN001, ANN202
    return "JSON"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _auth(client, username, password):
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _start_run(client, headers):
    body = {
        "project": "confirm-dialog",
        "name": f"run-{uuid4()}",
        "dataset_content_sha256": sha(f"ds-{uuid4()}"),
        "code_commit_sha": "abc1234def5678",
        "description": "completion summary api test",
        "expected_version": 0,
    }
    res = client.post("/api/runs", json=body, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()


def test_completion_summary_cancel_then_confirm(client):
    researcher = _auth(client, "researcher", "lab123456")
    run = _start_run(client, researcher)
    run_id = run["id"]

    # 记 2 条度量 + 挂 1 个附件（version 推进到 4）
    for step, value in ((1, 0.9), (2, 0.7)):
        res = client.post(
            f"/api/runs/{run_id}/metrics",
            json={"name": "loss", "value": value, "step": step, "expected_version": step},
            headers=researcher,
        )
        assert res.status_code == 200, res.text
    res = client.post(
        f"/api/runs/{run_id}/artifacts",
        json={
            "name": "model.bin",
            "uri": "s3://lab-artifacts/model.bin",
            "content_sha256": sha("model"),
            "media_type": "application/octet-stream",
            "expected_version": 3,
        },
        headers=researcher,
    )
    assert res.status_code == 200, res.text

    # 1) 点完成前先取摘要：两枚指纹 + 条数 + 当前版本全部来自接口
    res = client.get(f"/api/runs/{run_id}/completion-summary", headers=researcher)
    assert res.status_code == 200, res.text
    digest = res.json()
    assert digest["dataset_content_sha256"] == run["dataset_content_sha256"]
    assert digest["code_commit_sha"] == run["code_commit_sha"]
    assert digest["metrics_count"] == 2
    assert digest["artifacts_count"] == 1
    assert digest["version"] == 4
    assert digest["status"] == "running"

    # 2) 取消：只取了摘要、未发 complete —— 仍为进行中，时间线无完成事件
    detail = client.get(f"/api/runs/{run_id}", headers=researcher).json()
    assert detail["status"] == "running"
    events = client.get(f"/api/runs/{run_id}/events", headers=researcher).json()
    assert [e["event_type"] for e in events] == [
        "RunStarted",
        "MetricRecorded",
        "MetricRecorded",
        "ArtifactAttached",
    ]

    # 3) 确认：用摘要里的版本提交 CompleteRun —— 变为已完成
    res = client.post(
        f"/api/runs/{run_id}/complete",
        json={"result_summary": "确认后完成", "expected_version": digest["version"]},
        headers=researcher,
    )
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "completed"

    # 时间线出现完成事件
    events = client.get(f"/api/runs/{run_id}/events", headers=researcher).json()
    assert events[-1]["event_type"] == "RunCompleted"
    assert events[-1]["version"] == 5

    # 条数与详情对得上
    detail = client.get(f"/api/runs/{run_id}", headers=researcher).json()
    digest2 = client.get(f"/api/runs/{run_id}/completion-summary", headers=researcher).json()
    assert digest2["metrics_count"] == len(detail["metrics_json"]) == 2
    assert digest2["artifacts_count"] == len(detail["artifacts_json"]) == 1
    assert digest2["version"] == detail["version"] == 5
    assert digest2["status"] == "completed"


def test_auditor_readonly_cannot_complete(client):
    researcher = _auth(client, "researcher", "lab123456")
    auditor = _auth(client, "auditor", "audit123456")
    run = _start_run(client, researcher)
    run_id = run["id"]

    # 只读账号可读摘要（与其他 GET 一致），但 CompleteRun 被后端拒绝
    res = client.get(f"/api/runs/{run_id}/completion-summary", headers=auditor)
    assert res.status_code == 200, res.text
    res = client.post(
        f"/api/runs/{run_id}/complete",
        json={"result_summary": "x", "expected_version": 1},
        headers=auditor,
    )
    assert res.status_code == 403, res.text

    # 未登录连摘要都取不到
    res = client.get(f"/api/runs/{run_id}/completion-summary")
    assert res.status_code == 401, res.text
