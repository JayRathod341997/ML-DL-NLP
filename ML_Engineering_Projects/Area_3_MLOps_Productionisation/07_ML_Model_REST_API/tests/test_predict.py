def test_predict_single(client):
    response = client.post("/predict", json={"text": "This is a great product!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "inference_time_ms" in data
    assert 0 <= data["score"] <= 1


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch(client):
    texts = ["Great!", "Terrible!", "Average."]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(texts)


def test_predict_batch_empty_list(client):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422  # Pydantic validation
