from fastapi.testclient import TestClient
from api.dashboard import app

def test_version():
    client = TestClient(app)
    r = client.get("/api/version")
    assert r.status_code == 200
    j = r.json()
    assert "name" in j and "version" in j

def test_reset_and_ingest():
    client = TestClient(app)
    assert client.post("/api/test/reset").status_code == 200
    assert client.post("/api/ingest", json={"text": "Hello world"}).status_code == 200
    st = client.get("/api/unit/word/stats").json()
    assert st["bank_total_pairs"] >= 1
