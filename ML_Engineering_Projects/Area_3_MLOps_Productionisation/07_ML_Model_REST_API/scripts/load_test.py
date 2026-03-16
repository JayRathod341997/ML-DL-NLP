"""Locust load test for the ML Model API.

Usage:
    uv run locust -f scripts/load_test.py --headless -u 10 -r 2 --run-time 60s --host http://localhost:8000
    uv run locust -f scripts/load_test.py  # opens browser UI at http://localhost:8089
"""

import random
from locust import HttpUser, task, between

SAMPLE_TEXTS = [
    "This product is absolutely amazing and I love it!",
    "Terrible experience, would not recommend to anyone.",
    "It was okay, nothing special about it.",
    "Great quality for the price, very satisfied.",
    "The worst purchase I have ever made in my life.",
    "Exceeds my expectations in every possible way.",
    "Arrived late and damaged, very disappointing.",
    "Five stars! Will definitely buy again.",
]


class APIUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(4)
    def predict_single(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post("/predict", json={"text": text})

    @task(1)
    def predict_batch(self):
        texts = random.choices(SAMPLE_TEXTS, k=random.randint(2, 8))
        self.client.post("/predict/batch", json={"texts": texts})

    @task(1)
    def health_check(self):
        self.client.get("/health")
