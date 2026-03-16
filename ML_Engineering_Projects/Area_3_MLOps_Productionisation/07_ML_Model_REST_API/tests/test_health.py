def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "labels" in data
    assert isinstance(data["labels"], list)
