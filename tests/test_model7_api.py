from fastapi.testclient import TestClient

from model7.api import create_app


class DummyOrchestrator:
    def __init__(self):
        self.calls = []

    def orchestrate(self, input_text):
        self.calls.append(input_text)
        if input_text == "error":
            raise RuntimeError("boom")
        return {
            "input_text": input_text,
            "medicines": [{"name": "Amoxicillin"}],
            "chat_response": "Amoxicillin reviewed",
            "summary": {"medicine_count": 1, "risk_count": 0},
        }


def test_health_endpoint():
    app = create_app(orchestrator=DummyOrchestrator())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_orchestrate_endpoint_returns_payload():
    orchestrator = DummyOrchestrator()
    app = create_app(orchestrator=orchestrator)
    client = TestClient(app)

    response = client.post("/api/v1/orchestrate", json={"input_text": "Take amoxicillin"})

    assert response.status_code == 200
    assert response.json()["input_text"] == "Take amoxicillin"
    assert response.json()["medicines"][0]["name"] == "Amoxicillin"
    assert orchestrator.calls == ["Take amoxicillin"]


def test_validation_error_on_blank_input():
    app = create_app(orchestrator=DummyOrchestrator())
    client = TestClient(app)

    response = client.post("/api/v1/orchestrate", json={"input_text": "   "})

    assert response.status_code == 422
    assert "input_text" in response.text


def test_unexpected_error_returns_consistent_payload():
    app = create_app(orchestrator=DummyOrchestrator())
    client = TestClient(app)

    response = client.post("/api/v1/orchestrate", json={"input_text": "error"})

    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()


def test_openapi_docs_are_available():
    app = create_app(orchestrator=DummyOrchestrator())
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "MediGuard Model 7 API"
