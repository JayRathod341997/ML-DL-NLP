# ML Model REST API

A production-ready FastAPI inference server that serves a HuggingFace text classification (or embedding) model with health checks, Prometheus metrics, request timing, and Docker support.

---

## Architecture

```
HTTP Request
    │
    ▼
┌─────────────────────────────┐
│  FastAPI App                │
│  ├── TimingMiddleware        │  adds X-Inference-Time-Ms header
│  ├── LoggingMiddleware       │  structured request/response logs
│  └── PrometheusInstrument.  │  /metrics endpoint for scraping
└────────────┬────────────────┘
             │
      ┌──────┴──────────────┐
      │                     │
  POST /predict         GET /health
  POST /predict/batch   GET /ready
  GET /model/info       GET /metrics (Prometheus)
      │
      ▼
┌────────────────────┐
│   ModelRegistry    │  singleton, loads model once at startup
│   (lru_cache)      │  CPU or CUDA, configurable via .env
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   inference.py     │  tokenise → model forward → softmax → format
└────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `GET` | `/ready` | Readiness check — verifies model is loaded |
| `GET` | `/model/info` | Model name, type, labels, device |
| `POST` | `/predict` | Single text prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 32 texts) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## Dataset / Models

See [data.txt](data.txt) for model download links.

Default: `distilbert-base-uncased-finetuned-sst-2-english` (sentiment, binary).

---

## Setup

### Local development

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp .env.example .env
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 --env-file .env ml-api
```

---

## Usage

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'

# Response:
# {
#   "label": "POSITIVE",
#   "score": 0.9987,
#   "inference_time_ms": 42.3
# }
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible experience", "It was okay"]}'
```

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

### Run tests

```bash
uv run pytest
```

### Load test

```bash
uv run locust -f scripts/load_test.py --headless -u 10 -r 2 --run-time 30s --host http://localhost:8000
```

---

## Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| Latency p50 | 45ms | 8ms |
| Latency p99 | 120ms | 22ms |
| Throughput | ~22 req/s | ~120 req/s |
| Model size | 268MB | 268MB |

*Tested with DistilBERT-SST2, single worker.*

---

## Project Structure

```
07_ML_Model_REST_API/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── data.txt
├── src/
│   ├── app.py                      # FastAPI app factory + lifespan
│   ├── routers/
│   │   ├── predict.py              # POST /predict, /predict/batch
│   │   ├── health.py               # GET /health, /ready
│   │   └── model_info.py           # GET /model/info
│   ├── models/
│   │   ├── request_schemas.py      # Pydantic input models
│   │   └── response_schemas.py     # Pydantic output models
│   ├── ml/
│   │   ├── model_loader.py         # Singleton model registry
│   │   └── inference.py            # Tokenise + predict + format
│   └── middleware/
│       ├── logging.py              # Structured request logs
│       └── timing.py               # X-Inference-Time-Ms header
├── tests/
│   ├── conftest.py
│   ├── test_predict.py
│   └── test_health.py
├── scripts/
│   └── load_test.py                # Locust load test
└── configs/
    └── model_config.yaml
```

---

## Future Improvements

- Async batching (group concurrent requests into a single forward pass)
- Model hot-swapping without server restart
- gRPC interface for lower latency
- A/B testing with traffic splitting
