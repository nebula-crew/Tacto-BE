from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_health_shows_deployed_commit(monkeypatch):
    """배포 확인용: Docker 빌드 때 넣은 GIT_SHA를 짧게 보여 주고, 없으면 unknown."""
    with TestClient(app) as c:
        monkeypatch.setenv("GIT_SHA", "47e1076abcdef0123456789")
        assert c.get("/api/health").json()["commit"] == "47e1076"
        monkeypatch.delenv("GIT_SHA")
        assert c.get("/api/health").json()["commit"] == "unknown"


def test_inspection_requires_photo():
    with TestClient(app) as c:
        assert c.post("/api/inspections").status_code == 400
