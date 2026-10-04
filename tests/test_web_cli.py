from fastapi.testclient import TestClient

from proofpost.cli import main
from proofpost.web import app

client = TestClient(app)


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_index_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "proofpost" in r.text


def test_demo_endpoint_round_trip():
    example = client.get("/api/example").json()
    r = client.post("/api/demo", json=example)
    assert r.status_code == 200
    body = r.json()
    assert [a["verdict"] for a in body["result"]["attempts"]] == ["blocked", "ok"]
    assert "Evidence trail" in body["markdown"]


def test_bad_input_gives_clean_422():
    example = client.get("/api/example").json()
    example["documents"] = [{"name": "a.md", "text": "   "}]
    r = client.post("/api/demo", json=example)
    assert r.status_code == 422 and "No evidence" in r.json()["detail"]
    assert client.post("/api/demo", json={}).status_code == 422


def test_live_endpoint_without_key_is_503(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PROOFPOST_API_KEY", raising=False)
    example = client.get("/api/example").json()
    assert client.post("/api/run", json=example).status_code == 503


def test_cli_demo_exits_zero(capsys):
    assert main(["run", "--demo"]) == 0
    assert "Evidence trail" in capsys.readouterr().out


def test_cli_missing_args_exits_two():
    assert main(["run"]) == 2
