# ML Model REST API — Interview Preparation Guide

> Project stack: FastAPI · Uvicorn · Pydantic v2 · prometheus-fastapi-instrumentator · httpx · Locust
> Difficulty labels: ⭐ Mid-level  ⭐⭐ Senior-level

---

## Quick Reference Card

```
KEY FILES
─────────────────────────────────────────────────────────────
src/app.py                    create_app() factory, lifespan, CORS, Prometheus, routers
src/ml/model_loader.py        ModelRegistry singleton + @lru_cache get_model_registry()
src/ml/inference.py           predict(text), batch_predict(texts) → Pydantic response
src/routers/predict.py        POST /predict, POST /batch-predict
src/routers/health.py         GET /health/live, GET /health/ready
src/routers/model_info.py     GET /model-info
src/models/request_schemas.py PredictRequest, BatchPredictRequest (validators)
src/models/response_schemas.py PredictResponse, BatchPredictResponse, HealthResponse
src/middleware/timing.py      TimingMiddleware → X-Request-Time-Ms header
src/middleware/logging.py     RequestLoggingMiddleware
Dockerfile / docker-compose.yml FastAPI + Prometheus services
scripts/load_test.py          Locust HttpUser (predict×4, batch×1, health×1)
tests/conftest.py             TestClient with mocked ModelRegistry

KEY NUMBERS
─────────────────────────────────────────────────────────────
Prometheus histogram buckets  0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10s
p99 latency SLA target        < 500 ms for single prediction
Batch size recommendation     8–128 depending on model size
Uvicorn workers (prod)        (2 × CPU_cores) + 1
CORS preflight timeout        24 hours (max-age)
ModelRegistry lock            asyncio.Lock (single event loop)
LRU cache size                lru_cache(maxsize=None) — keep all loaded models
Health check interval (K8s)   liveness 10s, readiness 5s, startup 30s
```

---

## 1. Core Concepts & Theory

### 1.1 FastAPI Architecture & Async Internals

**Q1. ⭐ What is the difference between FastAPI and Flask for serving ML models? Why did you choose FastAPI?**

FastAPI is built on top of Starlette (an ASGI framework) and Pydantic, while Flask is a WSGI framework. The critical difference for ML serving is that FastAPI natively supports async/await, which means a single worker thread can handle multiple concurrent I/O-bound requests (e.g., waiting for database lookups, downstream HTTP calls) without blocking. Flask's synchronous nature means each request occupies a thread until completion, requiring more threads or worker processes to handle concurrency. FastAPI also provides automatic OpenAPI documentation generation, native request/response validation via Pydantic, and generally 2–3× better throughput on I/O-bound workloads compared to Flask. For CPU-bound ML inference, the async advantage is less pronounced, but it still matters for pre/post-processing steps and metadata lookups.

**Q2. ⭐ Explain the ASGI vs WSGI distinction. Why does it matter for ML APIs?**

WSGI (Web Server Gateway Interface) is a synchronous interface: the server calls the application with a request and blocks until the application returns a response. ASGI (Asynchronous Server Gateway Interface) is the async evolution: the server calls the application with a request and a "send/receive" channel, allowing the application to yield control back to the event loop while waiting for I/O. For ML APIs, ASGI matters because: (1) health checks and metadata endpoints should not block inference endpoints, (2) batch endpoints may need to await multiple async operations, (3) lifespan events (model loading on startup) are natively supported in ASGI. Uvicorn is the ASGI server; Gunicorn with UvicornWorker is common in production for multi-process deployment.

**Q3. ⭐⭐ How does Uvicorn's event loop interact with synchronous CPU-bound inference code? What are the pitfalls?**

Uvicorn runs a single asyncio event loop per worker process. When you call `await` on an async endpoint, control returns to the event loop which can serve other requests. However, if your inference code is synchronous and CPU-bound (e.g., running a PyTorch model forward pass), it blocks the event loop for the entire duration of that inference call, preventing any other request from being handled in that worker — defeating the purpose of async. The solution is to run blocking code in a thread pool using `asyncio.run_in_executor(None, sync_fn, *args)` or FastAPI's `run_in_threadpool`. For pure CPU-bound work, multiple Uvicorn worker processes (Gunicorn with UvicornWorker class) are more effective than a single async worker since Python's GIL prevents true parallelism within one process.

  ↳ Follow-up: "How would you determine whether your inference is I/O-bound or CPU-bound?"

  Profile with `cProfile` or `py-spy`. If wall time ≈ CPU time, it is CPU-bound. If wall time >> CPU time (waiting on network, disk, locks), it is I/O-bound. For transformer inference, it is predominantly CPU/GPU-bound. Run `htop` during load testing — if CPU is saturated at 100% per core, adding async won't help; add worker processes.

  ↳ Follow-up: "What is `run_in_threadpool` in FastAPI and when should you use it?"

  `run_in_threadpool` wraps a synchronous callable and runs it in the default thread pool executor of the event loop, returning an awaitable. Use it when you have a synchronous function you cannot rewrite as async (e.g., a legacy model scoring function). FastAPI automatically calls `run_in_threadpool` for you if you define a route function as a plain `def` rather than `async def` — that is a key FastAPI behavior many developers miss.

  ↳ Follow-up: "If you have 4 CPU cores and pure CPU-bound inference, how many Uvicorn workers do you configure?"

  Formula: `(2 × cores) + 1 = 9`, but for CPU-bound inference, workers should not exceed the number of physical cores to avoid context-switch overhead. Typically `num_cores` to `num_cores + 1` workers. Each worker loads its own model copy in memory, so consider GPU VRAM if using GPU inference — you may need to pin workers to specific GPU devices using `CUDA_VISIBLE_DEVICES`.

**Q4. ⭐ What is the `create_app()` factory pattern and why is it used instead of a module-level `app = FastAPI()`?**

The factory pattern (`create_app()` returning a `FastAPI` instance) separates application construction from module import. Benefits: (1) you can create multiple app instances with different configurations (test app vs production app), (2) it prevents side effects at import time (no model loading, no DB connections when the module is imported by pytest), (3) environment-specific settings (CORS origins, debug mode) can be injected as parameters. In `conftest.py`, `create_app()` is called with test-specific overrides and the `ModelRegistry` is mocked before the app starts, which is impossible with a module-level singleton.

**Q5. ⭐ Explain the `asynccontextmanager` lifespan pattern in FastAPI. What does it replace?**

FastAPI (via Starlette) previously used `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators. These are now deprecated in favor of the `asynccontextmanager` lifespan pattern: a single async generator function decorated with `@asynccontextmanager` that performs startup before the `yield` and shutdown after the `yield`. It is passed to `FastAPI(lifespan=lifespan)`. Advantages: (1) it is a single coherent function rather than split decorators, (2) resources initialized in startup are in scope until shutdown (no need for global variables), (3) it integrates cleanly with dependency injection, (4) it is easier to test in isolation.

**Q6. ⭐⭐ Why should you NOT load the model at module import time? Explain the worker forking problem.**

Loading a model at module import time causes several issues. (1) **Fork safety**: when Gunicorn forks worker processes after the master process has already loaded the model, CUDA contexts and certain file handles do not survive fork safely — this causes silent errors or crashes. The model should be loaded inside each worker's lifespan startup, after forking. (2) **Cold start on import**: any test that imports `app.py` will trigger model loading, making tests slow and requiring ML infrastructure. (3) **Memory inefficiency**: if model loading is at import time and multiple workers are forked, each worker may not get a clean memory state. (4) **Configuration flexibility**: loading in lifespan allows the model path to be read from environment variables which may not be set at import time.

  ↳ Follow-up: "How does Gunicorn's `preload_app` option relate to this?"

  `--preload` loads the application in the master process before forking workers. This saves memory via copy-on-write fork semantics on Linux, but breaks CUDA and multiprocessing. For ML serving, avoid `--preload` unless you are certain your model backend supports post-fork usage. Without `--preload`, each worker starts independently and loads the model in its own lifespan.

  ↳ Follow-up: "What is the `--preload` alternative for memory savings in ML serving?"

  Use model server frameworks designed for this (TorchServe, Triton Inference Server) which handle model lifecycle correctly. Alternatively, load the model from shared memory (`torch.multiprocessing.shared_memory`) or use a model cache server (Redis + serialized model bytes, though this is rarely practical for large models).

  ↳ Follow-up: "How would you handle a model that takes 45 seconds to load at startup?"

  Use Kubernetes startup probes with `failureThreshold=30` and `periodSeconds=10` (300-second total window). Implement a `GET /health/ready` endpoint that returns 503 until the model is loaded, so the pod is not added to the Service before it is ready. Use pre-warmed instances via Azure Container Apps min-replicas=1 or AWS ECS desired_count=1 to avoid cold starts entirely.

### 1.2 Pydantic v2 Validation

**Q7. ⭐ What changed in Pydantic v2 compared to v1? List the most important breaking changes.**

Pydantic v2 was rewritten in Rust (via `pydantic-core`), providing 5–50× faster validation. Breaking changes: (1) `@validator` replaced by `@field_validator` with `@classmethod` and different signature, (2) `__root__` models replaced by `RootModel`, (3) `orm_mode = True` replaced by `model_config = ConfigDict(from_attributes=True)`, (4) `.dict()` replaced by `.model_dump()`, (5) `.json()` replaced by `.model_dump_json()`, (6) `schema()` replaced by `model_json_schema()`, (7) `validator(pre=True)` replaced by `mode='before'` in `@field_validator`. The performance improvement is significant for high-QPS APIs where validation overhead was measurable.

**Q8. ⭐ Explain `@field_validator` in Pydantic v2 with an example relevant to PredictRequest.**

`@field_validator` decorates a classmethod that validates or transforms a specific field. For `PredictRequest.text`:
```python
@field_validator('text')
@classmethod
def validate_text(cls, v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError('text must not be empty or whitespace only')
    if len(v) > 10_000:
        raise ValueError(f'text too long: {len(v)} chars, max 10000')
    return v
```
The `mode='before'` flag runs the validator before type coercion (useful for stripping before checking type). The `mode='after'` (default) runs after type coercion. `@model_validator(mode='after')` runs after all fields are set and receives the model instance, allowing cross-field validation.

**Q9. ⭐⭐ What are discriminated unions in Pydantic v2 and when would you use them in an ML API?**

Discriminated unions allow a field to be one of several model types, selected by a literal "discriminator" field. Example: an ML API accepting different request types (classification vs regression vs embedding) in a single endpoint:
```python
class ClassificationRequest(BaseModel):
    task: Literal["classification"]
    text: str

class EmbeddingRequest(BaseModel):
    task: Literal["embedding"]
    texts: List[str]
    normalize: bool = True

InferenceRequest = Annotated[
    Union[ClassificationRequest, EmbeddingRequest],
    Field(discriminator='task')
]
```
Pydantic uses the `task` field to select the correct model without trying all types, making validation O(1) instead of O(n_types). This is much more efficient than `Union` without discriminator, which tries each type in order and fails through until one succeeds.

  ↳ Follow-up: "What is `@computed_field` in Pydantic v2?"

  `@computed_field` marks a property as part of the serialized output. For ML responses, use it to compute derived fields: e.g., a `confidence_category` property that returns "high"/"medium"/"low" based on the `confidence` float field, without requiring the caller to compute it. It is included in `.model_dump()` and `.model_dump_json()`.

  ↳ Follow-up: "How do you handle Optional fields with defaults in Pydantic v2?"

  Use `Optional[str] = None` (which is `Union[str, None]` with default `None`) or `str | None = None` (Python 3.10+ union syntax). Pydantic v2 distinguishes between a field being absent and a field being explicitly `null` using `model_config = ConfigDict(validate_default=True)` and `Field(default=None)`. Use `model_fields_set` to check which fields were explicitly provided in the request.

  ↳ Follow-up: "How do you document field constraints in the generated OpenAPI schema?"

  Use `Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1", example=0.87)`. FastAPI surfaces these in the Swagger UI automatically. For array length constraints, use `min_length=1, max_length=100` on `List` fields via `Field` or `Annotated[List[str], Field(min_length=1, max_length=100)]`.

**Q10. ⭐ How does Pydantic v2 handle JSON serialization of non-JSON-native types like `numpy.float32`?**

Pydantic v2's default JSON encoder does not know about `numpy.float32`. If your model returns NumPy scalars, you must either: (1) cast to Python native types before creating the response model (`float(score)` instead of `score`), (2) implement a custom `__get_pydantic_core_schema__` for NumPy types, or (3) use `model_config = ConfigDict(arbitrary_types_allowed=True)` with a custom JSON serializer. Option 1 is strongly preferred in practice — always call `.item()` on NumPy scalars or `tolist()` on arrays before passing to Pydantic models to avoid serialization errors at runtime.

### 1.3 Model Loading Patterns (Singleton, LRU Cache, Lifespan)

**Q11. ⭐ Explain the Singleton pattern used in ModelRegistry. Why use `__new__` instead of a class variable?**

The Singleton pattern ensures only one `ModelRegistry` instance exists per process. Using `__new__`:
```python
class ModelRegistry:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```
This intercepts instance creation before `__init__` runs, returning the existing instance if already created. A class variable approach works too, but `__new__` is more idiomatic and prevents double-initialization. In a multi-threaded context, `__new__` alone is not thread-safe; you need a lock around the `if cls._instance is None` check. However, in an async context (single event loop), an `asyncio.Lock` during async initialization prevents concurrent coroutines from initializing twice.

**Q12. ⭐ Why is `@lru_cache` applied to `get_model_registry()`? How does this interact with the Singleton?**

`@lru_cache` on a factory function caches the return value on first call, returning the cached object on subsequent calls with the same arguments. It provides a second layer of caching at the function call level. The combination of Singleton (class-level uniqueness) and `lru_cache` (function-level caching) ensures: (1) no redundant object creation, (2) fast access without constructor overhead, (3) easy testability — in tests, `get_model_registry.cache_clear()` resets the cache so a fresh mock registry is injected. Without `lru_cache`, every dependency injection call to `get_model_registry()` would invoke `__new__` (though it still returns the singleton).

**Q13. ⭐⭐ How do you ensure thread safety when multiple async handlers try to load the model simultaneously at startup?**

Use `asyncio.Lock`:
```python
_lock = asyncio.Lock()

async def load_model_once():
    async with _lock:
        if registry._model is None:
            registry._model = await asyncio.run_in_executor(None, load_heavy_model)
```
The first coroutine acquires the lock and loads the model. Any other coroutine waiting on `async with _lock` will see `registry._model is not None` when the lock is released and skip loading. Note: `asyncio.Lock` is NOT thread-safe across OS threads — if you use `run_in_executor` with a thread pool, use `threading.Lock` instead. For the single event loop case (single Uvicorn worker), `asyncio.Lock` is sufficient.

  ↳ Follow-up: "How would you handle a model that needs periodic refresh (e.g., every 6 hours)?"

  Implement a background task using `asyncio.create_task(refresh_loop())` in the lifespan startup. The refresh loop: `while True: await asyncio.sleep(6 * 3600); new_model = await load_new_model(); async with lock: registry._model = new_model`. The swap must be atomic with the lock held so in-flight requests are not disrupted. Use a double-buffering pattern if you want zero downtime: load new model into a secondary slot, then swap the pointer.

  ↳ Follow-up: "How do you handle model loading failure at startup? Should the app start or crash?"

  It depends on the deployment strategy. For Kubernetes: fail fast (raise an exception in lifespan startup) — this causes the container to exit with non-zero code, K8s restarts it and the startup probe catches the failure. For batch jobs or non-critical paths: log the error, set a `model_available` flag to `False`, have the readiness probe return 503, and keep retrying in a background loop. Never silently swallow model loading errors — it leads to health check false positives where the pod appears ready but inference fails.

  ↳ Follow-up: "Describe how you would support hot-swapping models without restarting the API."

  Implement a `POST /admin/reload-model` endpoint protected by an admin API key. The endpoint: acquires the lock, loads the new model from the configured path, validates it with a test inference call, swaps `registry._model`, releases the lock. Use semantic versioning in model filenames and store the current version in `registry._version`. The `/model-info` endpoint exposes the active version so callers know which model is serving.

### 1.4 Prometheus Metrics for ML APIs

**Q14. ⭐ What are the four Prometheus metric types? Which ones does your API use and for what?**

(1) **Counter**: monotonically increasing value, never decreases. Use for `http_requests_total` (labeled by method, endpoint, status code), `prediction_errors_total`, `model_loads_total`. (2) **Gauge**: value that can go up or down. Use for `active_requests` (in-flight count), `model_load_time_seconds` (current model's load duration). (3) **Histogram**: samples observations into configurable buckets, also tracks count and sum. Use for `http_request_duration_seconds` (request latency) — this enables p50/p95/p99 computation. (4) **Summary**: like histogram but computes quantiles client-side; less useful in distributed systems since summaries cannot be aggregated across replicas. `prometheus-fastapi-instrumentator` creates a Counter and Histogram automatically.

**Q15. ⭐ Explain p50, p95, p99 latency. Why is the median insufficient for SLA monitoring?**

p50 (median) means 50% of requests complete in that time or less. p95 means 95% complete in that time or less — 5% of users experience worse. p99 means 99% complete — 1% of users experience worse. The median is misleading for SLAs because ML APIs often have a bimodal latency distribution: fast cache hits and slow model inferences. The median may show 50 ms (mostly cache hits) while p99 is 2 seconds (uncached heavy inference). SLAs are typically defined on p99 or p99.9 because you are contractually committing to the experience of your slowest users, not your average users. Tail latency (p99) disproportionately affects perceived API reliability.

**Q16. ⭐⭐ How do you configure Prometheus histogram buckets for an ML inference API? What happens with wrong bucket choices?**

Default buckets (`0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10` seconds) work for most HTTP APIs. For an ML API where inference takes 100ms–2s, you want denser buckets in that range: `[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]`. If buckets are too coarse (e.g., only `[0.5, 1.0, 5.0]`), Prometheus can only estimate p99 within bucket boundaries — you might know p99 is between 1.0s and 5.0s, which is useless for SLA monitoring. Too many fine-grained buckets increase memory usage per time series. The rule: place bucket boundaries at your SLA thresholds (200ms, 500ms, 1s, 2s) so you can directly query `histogram_quantile(0.99, ...)` with precision.

  ↳ Follow-up: "What is Prometheus cardinality and why can it kill your monitoring system?"

  Cardinality is the number of unique time series, which equals the product of all unique label value combinations. If you add a `user_id` or `request_id` label to a high-QPS metric, you create millions of time series, consuming gigabytes of RAM in Prometheus and crashing it. Never use high-cardinality fields (user ID, session ID, free-text) as Prometheus labels. Keep labels to status codes, endpoints, methods, and model versions — all low-cardinality (< 1000 unique values).

  ↳ Follow-up: "How do you compute p99 latency from a Prometheus histogram in PromQL?"

  `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`. The `rate()` function computes the per-second rate of increase of the bucket counters over 5 minutes. `sum() by (le)` aggregates across all instances. `histogram_quantile()` interpolates within the bucket containing the 99th percentile. For per-endpoint p99: `histogram_quantile(0.99, sum(rate(...[5m])) by (le, handler))`.

  ↳ Follow-up: "Describe how you would create a custom Prometheus metric in FastAPI for tracking model prediction confidence."

  ```python
  from prometheus_client import Histogram
  confidence_histogram = Histogram(
      'ml_prediction_confidence',
      'Distribution of model prediction confidence scores',
      buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
  )
  # In predict endpoint:
  confidence_histogram.observe(response.confidence)
  ```
  Alert when `histogram_quantile(0.5, ...)` drops below 0.6 — indicates the model is becoming uncertain, a leading indicator of distribution shift.

### 1.5 Middleware Design (Timing, Logging, CORS)

**Q17. ⭐ Explain Starlette middleware execution order. What does "LIFO for response, FIFO for request" mean?**

Middleware is registered in order (first registered = outermost layer). For a request: middleware executes in registration order (FIFO — first in, first out) — outermost middleware processes the request first. For a response: middleware executes in reverse order (LIFO — last in, first out) — innermost middleware processes the response first and passes it outward. Example: if `TimingMiddleware` is registered first and `LoggingMiddleware` second, the request flows `TimingMiddleware → LoggingMiddleware → route handler`, and the response flows `route handler → LoggingMiddleware → TimingMiddleware`. This means `TimingMiddleware` wraps the entire request including logging overhead, giving a more accurate total time measurement.

**Q18. ⭐ How does `TimingMiddleware` add the `X-Request-Time-Ms` header? Show the implementation pattern.**

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers['X-Request-Time-Ms'] = f'{duration_ms:.2f}'
        return response
```
`time.perf_counter()` is used instead of `time.time()` because it has higher resolution and is monotonic (not affected by NTP adjustments). The header is added to the response object after `call_next()` returns. Note: in streaming responses, `call_next()` returns before the body is fully sent, so the timing may not include full response transmission time.

**Q19. ⭐⭐ What is a CORS preflight request and how does Starlette's CORSMiddleware handle it?**

A CORS (Cross-Origin Resource Sharing) preflight is an HTTP `OPTIONS` request sent by the browser before the actual request when: (1) the actual method is not GET/POST/HEAD, or (2) custom headers are used (e.g., `Authorization`, `Content-Type: application/json`). The browser asks the server: "Are you willing to accept this cross-origin request?" Starlette's `CORSMiddleware` intercepts `OPTIONS` requests, checks the `Origin` header against `allow_origins`, and responds with `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, `Access-Control-Allow-Headers`, and `Access-Control-Max-Age` (caches the preflight for up to 24 hours). In production, never use `allow_origins=["*"]` for authenticated APIs — it allows any website to make requests using the user's cookies/credentials.

  ↳ Follow-up: "What is the security risk of `allow_origins=['*']` with `allow_credentials=True`?"

  This combination is actually rejected by browsers (and Starlette raises a `ValueError`). If `allow_credentials=True`, you must specify explicit origins. The risk with wildcard origins (without credentials) is that it allows any website to read your API responses — fine for truly public data but dangerous if the API returns any user-specific information. An attacker can create a malicious website that calls your API and reads the response.

  ↳ Follow-up: "How would you configure CORS for a FastAPI app served in production behind an API gateway?"

  The API gateway (Azure API Management or AWS API Gateway) should handle CORS and strip the `Origin` header before passing to the backend. Configure CORS at the gateway level only. On the FastAPI app, either disable CORSMiddleware or configure it to trust only the gateway's internal hostname. This prevents CORS misconfiguration at the application level from bypassing gateway policies.

**Q20. ⭐ What are the request headers you should always log in `RequestLoggingMiddleware`? What should you NEVER log?**

Always log: `X-Request-ID` (for tracing), `User-Agent`, `Content-Type`, `Content-Length`, HTTP method, URL path, query params (sanitized), response status code, response time, `X-Forwarded-For` (for real client IP). Never log: `Authorization` header (tokens, API keys), request body containing PII (names, emails, health data), `Cookie` header, any field matching patterns for credit card numbers, SSNs. Implement a sanitize function that redacts these fields from logs. In regulated environments (GDPR, HIPAA), even IP addresses may constitute PII requiring consent.

### 1.6 API Design Best Practices for ML

**Q21. ⭐ Compare single prediction vs batch prediction API design. When should you force callers to use batch?**

Single prediction (`POST /predict`) is simpler for callers and has lower latency for a single item. Batch prediction (`POST /batch-predict`) amortizes the overhead of: network round trips, tokenizer calls, GPU kernel launch overhead, and Pydantic validation across N items, giving better throughput. Force batch when: (1) caller processes data offline or near-offline (ETL, nightly scoring), (2) model latency without batching would exceed SLA, (3) per-request overhead is dominant (small models where GPU launch cost > inference cost). Optimal batch size depends on GPU memory (larger = more VRAM) and latency SLA (larger = more latency). Typically 8–64 for transformer models.

**Q22. ⭐ Explain health check design for Kubernetes. What is the difference between liveness, readiness, and startup probes?**

**Liveness probe**: "Is this container still alive? Should Kubernetes restart it?" Fails only for unrecoverable states (deadlock, OOM about to crash). `GET /health/live` returns 200 always unless the process is truly stuck. Killing a live pod prematurely causes more restarts than availability gain. **Readiness probe**: "Is this container ready to receive traffic?" Fails when the model is still loading, a required downstream service is down, or the service is temporarily overloaded. `GET /health/ready` returns 503 if `model_loaded = False`. When readiness fails, the pod is removed from the Service load balancer but NOT restarted. **Startup probe**: "Has the container finished its slow initialization?" Disables liveness and readiness during startup to prevent premature kills. Set `failureThreshold × periodSeconds` to exceed your maximum startup time (e.g., 30s model load: `failureThreshold=6, periodSeconds=10`).

**Q23. ⭐⭐ Describe three API versioning strategies for ML APIs and their trade-offs.**

(1) **URL path versioning** (`/v1/predict`, `/v2/predict`): easiest for clients to understand, breaks bookmarks on version change, easy to proxy different versions to different backends. Most common in ML APIs. (2) **Header versioning** (`Accept: application/vnd.myapi.v2+json`): cleaner URLs, harder for clients to use and test in browsers, requires header inspection in routing middleware. (3) **Query parameter versioning** (`/predict?version=2`): easy to test, but query params should represent state/filters, not API contracts — semantically incorrect. Best practice for ML: URL path versioning with two live versions simultaneously, `v1` with deprecation warnings for 90 days before removal.

  ↳ Follow-up: "How do you handle model versioning separately from API versioning?"

  API version controls the request/response contract (schema). Model version controls which ML model is serving. These are orthogonal: `/v1/predict` may serve model v3. Return `model_version` in the response headers or response body (`X-Model-Version: 3.1.2`). Store model version in `ModelInfoResponse`. This allows model updates without API contract changes, and allows auditing which model version produced a given prediction.

  ↳ Follow-up: "What metadata should `/model-info` return?"

  Model name, version, creation timestamp, training data cutoff date, supported input languages/domains, feature importance top-N, model size (parameters), inference runtime, hardware requirements, and a link to the model card. In regulated industries, also include: approval status, fairness metrics, training dataset description. This endpoint should be unauthenticated and public for auditability.

**Q24. ⭐ Describe the circuit breaker pattern. When would you apply it in an ML API?**

A circuit breaker has three states: **Closed** (normal, all requests pass), **Open** (failures exceeded threshold, all requests fail fast with 503 without trying), **Half-Open** (one test request is allowed; if it succeeds, circuit closes; if it fails, circuit opens again). Apply it in an ML API when: the model depends on an external service for feature retrieval (Redis cache, feature store), calls a downstream ML service (embeddings API), or connects to a database for logging. Example: if the Redis feature cache is down and every request times out after 5 seconds, the circuit breaker opens and returns cached fallback results or degraded predictions within 10ms, preventing cascade failures.

**Q25. ⭐ Explain token bucket vs sliding window rate limiting. Which is better for ML APIs?**

**Token bucket**: a bucket holds up to N tokens, tokens are added at rate R per second, each request consumes 1 token. Allows bursting up to N requests immediately, then sustains R req/s. Good for accommodating legitimate traffic spikes. **Sliding window**: counts requests in a rolling time window (e.g., last 60 seconds). More accurate than fixed windows (no burst at window boundary), less bursty than token bucket. For ML APIs: token bucket is preferred because inference workloads naturally burst (batch upload followed by silence). Rate limiting should be applied per API key, not per IP (proxies share IPs). Use Redis (with INCR + EXPIRE) for distributed rate limiting across multiple API replicas.

### 1.7 Load Testing with Locust

**Q26. ⭐ Explain the Locust task weight system. Why are predict tasks weighted 4 and batch weighted 1?**

Locust's `@task(weight)` annotation determines the relative probability of executing that task. With predict weight=4 and batch weight=1, 80% of simulated requests will be single predictions and 20% will be batch predictions. This reflects realistic production traffic patterns where batch requests are less frequent but heavier. The weight ratio should match your actual production traffic ratio. Additionally, health check at weight 1 reflects monitoring probes. Setting realistic weights is critical for meaningful load test results — a test that's 50% batch requests will show artificially degraded throughput that doesn't reflect real conditions.

**Q27. ⭐⭐ What is the difference between Locust `HttpUser` and `FastHttpUser`? When does it matter?**

`HttpUser` uses the `requests` library (blocking HTTP), each user runs in a gevent greenlet. `FastHttpUser` uses `geventhttpclient`, which is ~3-5× faster with lower overhead per simulated user. The difference matters when: (1) you are load testing a very fast endpoint (<5ms) where client-side overhead becomes the bottleneck, (2) you need to simulate thousands of concurrent users on a single load generator machine, (3) you are doing performance profiling rather than just functional testing. For most ML APIs (>50ms latency), `HttpUser` is sufficient. Use `FastHttpUser` when the load generator itself becomes CPU-bottlenecked before the server.

  ↳ Follow-up: "How do you interpret Locust's RPS, response time percentiles, and failure rate?"

  RPS (requests per second) is your throughput ceiling at the current user count. Response time p50/p95/p99 show latency distribution. Failure rate > 0% indicates the server is dropping requests (502/503/504 status codes, connection timeouts). A healthy load test shows: RPS increasing linearly with users up to a knee point, then flattening (capacity limit). p99 latency should remain stable until the knee point, then spike. Failure rate should be 0% until capacity is exceeded.

  ↳ Follow-up: "What is a soak test and what does it detect that a spike test does not?"

  A soak test runs at moderate load (e.g., 60% capacity) for an extended period (hours to days). It detects: memory leaks (RSS grows over time), connection pool exhaustion, log file disk saturation, certificate expiry during test, database connection leak, gradual GC pressure. A spike test (sudden high load) detects: autoscaling responsiveness, circuit breaker behavior, connection pool limits, cold-start latency under pressure. You need both.

  ↳ Follow-up: "How would you simulate realistic user think time in Locust?"

  Use `self.wait_time = between(1, 5)` on the `HttpUser` class to add 1–5 seconds random wait between tasks. This prevents Locust from sending requests as fast as possible, which would test the server's burst handling rather than sustained concurrency. Think time should match actual user behavior from production logs.

---

## 2. System Design Discussions

**Q28. ⭐⭐ Design a high-availability ML inference API that handles 10,000 requests per second with p99 < 100ms.**

```
                    ┌─────────────────┐
                    │  API Gateway    │
                    │ (Rate Limiting, │
                    │  Auth, TLS)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Load Balancer  │
                    │  (L7, sticky    │
                    │   sessions off) │
                    └──┬──────────┬───┘
               ┌───────▼──┐  ┌───▼──────┐
               │ FastAPI  │  │ FastAPI  │
               │ Pod 1    │  │ Pod 2    │  ... N pods
               │ (4 CPU)  │  │ (4 CPU)  │
               └────┬─────┘  └────┬─────┘
                    │              │
          ┌─────────▼──────────────▼──────────┐
          │         Redis Cluster              │
          │  (prediction cache, TTL=300s)      │
          └───────────────────────────────────┘
```

Key design decisions: (1) Redis cache for repeated predictions — 60–80% cache hit rates on typical NLP workloads reduce p99 dramatically. (2) Horizontal scaling with stateless pods (model loaded in each pod). (3) Connection pooling (uvloop + httpx AsyncClient pools). (4) Response pre-computation for known inputs. At 10k RPS with 4 pods, each pod handles 2,500 RPS — with 4 Uvicorn workers each, that's 625 RPS per worker, or ~1.6ms budget per request for CPU-bound work. This likely requires GPU inference or a very small/quantized model.

**Q29. ⭐⭐ How would you design an A/B testing infrastructure for two versions of an ML model behind a single API?**

```
Client → API Gateway
       → Experiment Router (reads user_id → experiment assignment)
         ├── 90% traffic → Model A pod group (control)
         └── 10% traffic → Model B pod group (treatment)
       → Both groups log: user_id, model_version, prediction,
         confidence, response_time to experiment event stream
       → Experiment analysis service
         ├── Computes: conversion rate, engagement, error rate per group
         └── Statistical significance via Mann-Whitney U or bootstrap CI
```

Route by user ID (consistent hashing) so the same user always gets the same model. Log the `model_version` and `experiment_id` in every response header. Minimum experiment duration: determine sample size via power analysis (e.g., detect 5% improvement with 80% power → ~1,600 samples per group). Track both business metrics (conversion) and technical metrics (latency, error rate). Have automatic kill switches if treatment error rate exceeds control by >2%.

**Q30. ⭐⭐ Design a shadow deployment system to test a new model without serving its predictions.**

```
Request → Primary Router
        ├── → Production Model (serves response to user) [sync]
        └── → Shadow Model (fire-and-forget async call)    [async, no SLA]
              → Shadow Response Logger
                → Compare: production vs shadow predictions
                → Alert if divergence > threshold
                → Store for offline analysis
```

Implementation: the primary router copies the request and sends it to a shadow service asynchronously using `asyncio.create_task()` — the user gets the production response immediately. The shadow service processes with the new model and logs the comparison. Key metrics: prediction agreement rate, confidence distribution difference, latency profile of new model. Promote shadow to production when: agreement rate > 95%, new model has better confidence distribution, latency SLA is met.

**Q31. ⭐ How would you add result caching to the predict endpoint? What are the cache key design considerations?**

Cache key should include: (1) the exact input text (or its hash for memory efficiency), (2) the model version, (3) any configurable inference parameters (temperature, threshold). Use SHA-256 hash of `f"{model_version}:{input_text}"` as the Redis key. TTL should reflect data freshness requirements: 5 minutes for frequently changing scores, 1 hour for stable classifications. Do NOT cache: predictions with uncertainty above a threshold (they may be wrong), requests with user-specific context. Cache invalidation on model update: prefix all cache keys with model version so deploying a new model auto-invalidates old cache entries. Expected hit rate for NLP APIs: 40–70% (many duplicate or near-duplicate inputs in practice).

---

## 3. Coding & Implementation Questions

**Q32. ⭐ Write the FastAPI lifespan context manager that loads the ModelRegistry on startup and cleans up on shutdown.**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.ml.model_loader import get_model_registry
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model registry...")
    registry = get_model_registry()
    await registry.load()
    logger.info(f"Model loaded: {registry.model_version}")
    app.state.registry = registry
    yield
    # Shutdown
    logger.info("Shutting down model registry...")
    await registry.cleanup()
    get_model_registry.cache_clear()

def create_app() -> FastAPI:
    app = FastAPI(title="ML Inference API", lifespan=lifespan)
    # Add middleware, routers...
    return app
```

**Q33. ⭐ Write a `PredictRequest` Pydantic v2 model with validators for an NLP API.**

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      description="Input text for classification")
    language: Optional[str] = Field(None, pattern=r'^[a-z]{2}$',
                                     description="ISO 639-1 language code")
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                              description="Confidence threshold")

    @field_validator('text', mode='before')
    @classmethod
    def clean_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('text must not be empty or whitespace')
        # Reject null bytes (injection attempt)
        if '\x00' in v:
            raise ValueError('text contains null bytes')
        return v

    @model_validator(mode='after')
    def validate_language_text_consistency(self) -> 'PredictRequest':
        if self.language == 'zh' and len(self.text) < 2:
            raise ValueError('Chinese text must be at least 2 characters')
        return self
```

**Q34. ⭐⭐ Implement a batch predict endpoint with timeout handling and partial failure support.**

```python
import asyncio
from fastapi import APIRouter, HTTPException
from src.models.request_schemas import BatchPredictRequest
from src.models.response_schemas import BatchPredictResponse

router = APIRouter()
BATCH_TIMEOUT_SECONDS = 30.0

@router.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    if len(request.texts) > 100:
        raise HTTPException(status_code=422,
                            detail="Batch size exceeds maximum of 100")
    try:
        results = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, inference.batch_predict, request.texts
            ),
            timeout=BATCH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Batch inference timed out after {BATCH_TIMEOUT_SECONDS}s"
        )
    return BatchPredictResponse(
        predictions=results.predictions,
        errors=results.errors,   # per-item errors
        processing_time_ms=results.duration_ms
    )
```

**Q35. ⭐ Write a pytest fixture that fully mocks the ModelRegistry for unit testing endpoints.**

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from src.app import create_app
from src.ml.model_loader import get_model_registry

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.model_version = "test-v1.0"
    registry.is_loaded = True
    registry.predict = MagicMock(return_value={"label": "positive", "score": 0.95})
    return registry

@pytest.fixture
def client(mock_registry):
    with patch('src.ml.model_loader.get_model_registry',
               return_value=mock_registry):
        get_model_registry.cache_clear()
        app = create_app()
        with TestClient(app) as c:
            yield c
```

---

## 4. Common Bugs & Issues

| # | Bug | Root Cause | Symptom | Fix |
|---|-----|------------|---------|-----|
| 1 | Model loads on every request | Singleton not used; object created in route handler | High latency, high memory | Move to lifespan, use `get_model_registry()` in dependency |
| 2 | CORS error on frontend | `allow_origins=["*"]` with credentials, or origin mismatch | Browser console CORS error | Set explicit origins, check for trailing slash mismatch |
| 3 | Pydantic `ValidationError` on `numpy.float32` | NumPy scalars not JSON-serializable by default | 500 error on serialization | Cast to `float()` before creating response model |
| 4 | `asynccontextmanager` not cleaning up | Exception in startup, `yield` never reached | Resource leak on startup failure | Wrap startup in try/except, ensure yield always reached |
| 5 | Health check always returns 200 | Not checking model load status in readiness | K8s sends traffic to unready pods | Return 503 from `/health/ready` until `registry.is_loaded` |
| 6 | Prometheus cardinality explosion | URL path with ID as label (e.g., `/predict/user-123`) | Prometheus OOM | Strip dynamic path segments, use only route template |
| 7 | TimingMiddleware shows 0ms | `time.time()` instead of `time.perf_counter()`, precision loss | Misleading metrics | Use `time.perf_counter()` |
| 8 | Middleware blocks event loop | `time.sleep()` in middleware | All requests blocked | Use `asyncio.sleep()` or non-blocking IO |
| 9 | Request body logged | Logging middleware logs raw body | PII in logs | Strip/redact request body in logging middleware |
| 10 | LRU cache across tests | `lru_cache` persists between test cases | Test isolation failure | Call `get_model_registry.cache_clear()` in pytest teardown |
| 11 | Worker OOM on fork with `--preload` | CUDA context duplicated in forked workers | Worker crash on first GPU call | Remove `--preload`, load model in lifespan |
| 12 | 422 on valid batch request | Pydantic `min_length` on List field not configured | Client gets confusing error | Set `Field(min_length=1, max_length=100)` on texts list |
| 13 | Uvicorn ignores `--workers` flag | Running `uvicorn` directly instead of `gunicorn -k UvicornWorker` | Single worker in production | Use Gunicorn for multi-process, Uvicorn for dev |
| 14 | Batch endpoint 502 on large inputs | Nginx/proxy `client_max_body_size` too small | 502 before hitting FastAPI | Increase proxy body size limit or add request size validation |
| 15 | X-Request-Time-Ms on streamed responses | `call_next()` returns before stream complete | Inaccurate timing | Document limitation; use middleware only for non-streaming |
| 16 | 500 on Unicode in text field | Encoding not normalized | Edge case with emoji or RTL text | Add `unicodedata.normalize('NFC', v)` in validator |
| 17 | Locust shows 0 failures but p99 is 60s | Locust default timeout is 60s | Missed timeouts masking real SLA violations | Set `self.client.timeout = 5.0` in Locust user |
| 18 | Model version not in response | `ModelInfoResponse` not included in predict response | Cannot audit which model produced prediction | Add `model_version` field to `PredictResponse` |

---

## 5. Deployment — Azure

**Q36. ⭐ How would you deploy this FastAPI ML API on Azure Container Apps?**

```
Build & Push Image:
──────────────────
Local Code
    → docker build -t myapi:v1.2.3 .
    → docker tag myapi:v1.2.3 myregistry.azurecr.io/mlapi:v1.2.3
    → docker push myregistry.azurecr.io/mlapi:v1.2.3

Azure Resource Topology:
────────────────────────
Azure Container Registry (myregistry.azurecr.io)
    │
    └─→ Azure Container Apps Environment
            │
            ├─→ Container App: ml-api
            │     Image: myregistry.azurecr.io/mlapi:v1.2.3
            │     CPU: 2.0 cores, Memory: 4Gi
            │     Min replicas: 1 (avoid cold start)
            │     Max replicas: 10
            │     Scale rule: HTTP requests > 20 per replica → scale out
            │     Liveness:  GET /health/live  (every 10s)
            │     Readiness: GET /health/ready (every 5s, initial 30s)
            │     Env vars: MODEL_PATH=/mnt/models/model.pkl
            │               LOG_LEVEL=INFO
            │
            └─→ Azure Blob Storage (model artifacts, mounted via FUSE)

Azure API Management (gateway layer)
    │   Rate limit: 100 req/min per API key
    │   Auth: API key in header X-Api-Key
    │   Policies: IP filtering, request size limit 1MB
    └─→ Container App ingress URL
```

**Q37. ⭐⭐ Design the full Azure observability stack for this API.**

```
FastAPI Application
    │
    ├── Prometheus /metrics endpoint
    │       └─→ Azure Monitor managed Prometheus (scrape every 30s)
    │               └─→ Azure Managed Grafana (dashboards)
    │                     ├── p99 latency dashboard
    │                     ├── RPS by endpoint
    │                     └── Error rate alerts
    │
    ├── Structured JSON logs (stdout)
    │       └─→ Azure Log Analytics Workspace
    │               ├── KQL queries for error analysis
    │               └── Azure Monitor Alerts (error rate > 1%)
    │
    ├── Application Insights SDK (optional, for distributed tracing)
    │       └─→ End-to-end request trace: client → APIM → Container App → model
    │
    └── Custom metrics via Azure Monitor REST API
            └─→ model_load_time, prediction_confidence_p50
```

Key alert rules:
- p99 latency > 500ms for 5 minutes → PagerDuty
- Error rate > 2% for 2 minutes → Slack + PagerDuty
- Pod count at max replicas for 10 minutes → capacity review notification
- Memory > 80% of limit for 5 minutes → potential OOM alert

**Q38. ⭐⭐ How do you implement zero-downtime blue-green deployments on Azure Container Apps?**

```
Current State:                    Deployment State:
──────────────                    ─────────────────
APIM → Container App             APIM → Container App
       (blue: v1, 100%)                  ├── blue:  v1, 50%
                                         └── green: v2, 50%

Final State (after validation):
────────────────────────────────
APIM → Container App
       ├── blue:  v1,   0% (kept for rollback)
       └── green: v2, 100%
```

Azure Container Apps supports traffic splitting natively. Steps: (1) deploy new revision (green) with `az containerapp revision copy`, (2) set traffic split 90/10 (blue/green) via `az containerapp ingress traffic set`, (3) monitor green revision metrics for 15 minutes, (4) shift to 100% green if p99 < SLA and error rate < 1%, (5) deactivate blue revision (keep for 24h for emergency rollback). Rollback: `az containerapp ingress traffic set --revision blue=100 --revision green=0`.

  ↳ Follow-up: "How does Azure API Management help with ML API versioning?"

  APIM supports API version sets, allowing you to publish `/v1/predict` and `/v2/predict` as distinct APIs with different policies. Each version can route to different Container App revisions. APIM also supports: subscription keys (API keys), JWT validation, rate limiting per subscription, request/response transformation (e.g., adding model version to response headers), and developer portal for API documentation.

  ↳ Follow-up: "How would you configure Azure Cache for Redis for prediction caching?"

  Deploy Azure Cache for Redis (C1 Standard tier for dev, P1 Premium for prod with geo-replication). In FastAPI: use `aioredis` for async Redis access. Cache key: `f"predict:v{model_version}:{sha256(text).hexdigest()}"`. TTL: 300 seconds. Set `maxmemory-policy allkeys-lru` so Redis evicts least-recently-used keys when full. Cache hit rate metric: expose as Prometheus Gauge and alert if it drops below 30% (indicates cache too small or TTL too short).

**Q39. ⭐ What Azure Container Registry features are important for production ML model API deployments?**

(1) **Geo-replication**: replicate images to the region where Container Apps are deployed for fast pull times. (2) **Content trust**: sign images with Notary to prevent running tampered images. (3) **Vulnerability scanning**: integrate with Microsoft Defender for Containers to scan for CVEs before deployment. (4) **Retention policies**: auto-delete untagged manifests older than 30 days to control storage costs. (5) **Webhooks**: trigger Azure DevOps pipeline on new image push for automated deployment. (6) **Managed identity**: Container Apps pull images using assigned managed identity — no registry credentials stored in config.

---

## 6. Deployment — AWS

**Q40. ⭐⭐ Design the AWS deployment architecture for this FastAPI ML API at scale.**

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  AWS API Gateway (REST API)                             │
│  ├── Usage plans (100 req/min per API key)              │
│  ├── API keys in X-Api-Key header                       │
│  ├── Request validation (body schema)                   │
│  └── Custom domain: api.company.com                     │
└────────────────────┬────────────────────────────────────┘
                     │  (HTTP integration to ALB)
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Application Load Balancer                              │
│  ├── Target group: ECS Fargate tasks                    │
│  ├── Health check: GET /health/ready, 200 only          │
│  └── Sticky sessions: DISABLED (stateless)              │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│  ECS Fargate     │  │  ECS Fargate     │  ... N tasks
│  Task (FastAPI)  │  │  Task (FastAPI)  │
│  2 vCPU, 4GB RAM │  │  2 vCPU, 4GB RAM │
│  ENV: MODEL_PATH │  │  MODEL pulled    │
│  from S3 on start│  │  from S3 on start│
└────────┬─────────┘  └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  ElastiCache Redis   │
         │  (prediction cache)  │
         │  r6g.large, cluster  │
         └──────────────────────┘

Observability:
──────────────
FastAPI → CloudWatch Container Insights (CPU, memory, RPS)
FastAPI → X-Ray (distributed tracing, per-segment latency)
FastAPI → CloudWatch Custom Metrics via EMF (p99 latency)
CloudWatch Alarms → SNS → PagerDuty + Slack
```

**Q41. ⭐ How do you configure ECS Fargate autoscaling for this API?**

Use ECS Service Auto Scaling with Application Auto Scaling. Scaling policy: **Target Tracking** on `RequestCountPerTarget` = 50 (scale out when each task handles > 50 concurrent requests). Also add CPU-based tracking at 70% CPU utilization. Scale-out cooldown: 60 seconds (fast response to traffic spikes). Scale-in cooldown: 300 seconds (avoid flapping — model reload takes time). Min tasks: 2 (high availability across 2 AZs). Max tasks: 20. Pre-scale for known traffic spikes (batch jobs, marketing campaigns) using scheduled scaling.

**Q42. ⭐⭐ Explain how AWS X-Ray traces a request through API Gateway → ECS FastAPI → ElastiCache.**

```
X-Ray Trace:
────────────
[API Gateway Segment: 45ms total]
  ├── [Lambda Authorizer: 12ms]       (if using Lambda auth)
  └── [HTTP Integration: 33ms]
        └── [FastAPI Segment: 28ms]
              ├── [Middleware: 1ms]   (timing, logging)
              ├── [Redis GET: 2ms]    (cache lookup)  ← CACHE HIT
              │     └── [ElastiCache subsegment]
              └── [Serialization: 0.5ms]

(Cache miss path)
[FastAPI Segment: 180ms total]
  ├── [Middleware: 1ms]
  ├── [Redis GET: 2ms]               ← CACHE MISS
  ├── [Model Inference: 165ms]       ← bottleneck
  │     └── [CPU subsegment annotated with model_version]
  ├── [Redis SET: 1ms]               (write cache)
  └── [Serialization: 0.5ms]
```

Instrument with `aws_xray_sdk` and the `XRayMiddleware` for FastAPI. Annotate inference with `xray_recorder.begin_subsegment('inference')`. Add metadata: `xray_recorder.current_subsegment().put_metadata('model_version', registry.version)`. X-Ray Service Map shows latency breakdown per component and identifies the model inference as the dominant cost.

  ↳ Follow-up: "How do CloudWatch EMF logs differ from standard CloudWatch metrics?"

  EMF (Embedded Metric Format) embeds structured metric data in log lines. The CloudWatch Agent or Lambda runtime parses these logs and converts them to CloudWatch Custom Metrics without requiring explicit `put_metric_data` API calls. For ML APIs: log `{"_aws": {"Metrics": [{"Name": "PredictionLatencyMs", "Unit": "Milliseconds"}]}, "PredictionLatencyMs": 145.3, "ModelVersion": "v2.1"}`. This creates a CloudWatch metric with a `ModelVersion` dimension, allowing you to compare latency across model versions in CloudWatch Metrics.

  ↳ Follow-up: "Compare AWS API Gateway REST API vs HTTP API for ML serving."

  HTTP API is 70% cheaper and has lower latency (~1ms vs ~5ms overhead). REST API offers more features: usage plans, API keys, request validation against JSON schema, detailed CloudWatch logging. For ML APIs: use REST API if you need usage plans per customer, request validation, or AWS WAF integration. Use HTTP API if you only need routing, JWT auth, and minimal overhead. Both integrate with ECS via HTTP integrations to ALB.

---

## 7. Post-Production Issues

| # | Issue | Cause | How to Detect | Solution | Prevention |
|---|-------|-------|---------------|----------|------------|
| 1 | Memory leak in model singleton | Model object accumulates references over time (intermediate tensors, callbacks not cleaned) | RSS memory growth in Container Insights over 24h; OOMKilled events | Add `gc.collect()` after inference; profile with `tracemalloc`; restart pods on schedule | Use weak references for caches; profile memory before release |
| 2 | Cold start latency spike | Model loads on first request after scale-out | p99 spikes to 30–60s periodically; correlates with new pod starts | Implement readiness probe — pod only receives traffic after model is loaded | Set min replicas ≥ 1; use startup probe with generous timeout |
| 3 | OOM kill under sustained load | Each request retains input tensor in memory; batch size too large | Container exit code 137 (SIGKILL from OOM); memory gauge spikes before kill | Reduce batch size max; add `del` + `gc.collect()` after batch inference; tune memory limit | Load test with sustained load before prod; set memory limit 2× expected peak |
| 4 | Prometheus cardinality explosion | URL template not extracted; unique IDs in metric labels | Prometheus `go_memstats_heap_inuse_bytes` grows continuously; Prometheus query latency increases | Identify and drop high-cardinality labels; use Prometheus relabeling rules | Review all metric label sets in code review; set cardinality budget |
| 5 | CORS errors after new deployment | New subdomain added to frontend but not to CORS allow_origins list | Browser console: "No 'Access-Control-Allow-Origin' header"; only affects browser clients | Add new origin to environment variable / config; redeploy | CORS origins should be in centralized config, not hardcoded; test CORS in staging |
| 6 | Health check false positive | `/health/ready` always returns 200 but model fails silently | Real errors in logs but pod never restarted or removed from LB | Fix readiness probe to call `registry.is_loaded` and run test inference | Implement integration-level health check that does a real model call |
| 7 | Batch endpoint 502 errors at peak | Nginx/ALB timeout shorter than batch processing time | 502s only for `/batch-predict`, not `/predict`; correlates with large batches | Increase ALB idle timeout to 60s for batch endpoint; add async processing queue | Set endpoint-specific timeouts; document batch size limits in API contract |
| 8 | API key leaked in logs | `Authorization` or `X-Api-Key` header logged by RequestLoggingMiddleware | API keys in CloudWatch logs accessible to all developers | Redact sensitive headers in logging middleware; rotate leaked keys | Add header redaction to code review checklist; use log scanning tools |
| 9 | Model version mismatch after rollback | LRU cache not cleared after rollback; old model version in cache | Predictions use old model version despite new version deployed | Clear `get_model_registry.cache_clear()` on startup; include model version in health check | Tie cache lifetime to pod lifetime; log model version on startup |
| 10 | Locust test passes but prod fails | Locust ran against dev with mocked model; prod has real model | p99 latency 10× higher than load test showed | Run load tests against prod-equivalent environment with real model | Load test environment must match prod including model size and hardware |
| 11 | Pydantic v2 validation regression | Upgrade from Pydantic v1 to v2 broke custom validators | 422 errors on previously valid requests | Audit all `@validator` → `@field_validator` migrations; test edge cases | Maintain request validation integration tests for edge cases |
| 12 | X-Request-Time-Ms negative value | Server clock adjustment (NTP sync) mid-request; `time.time()` used | Negative values in timing header; affects SLA dashboards | Replace `time.time()` with `time.perf_counter()` (monotonic) | Use monotonic timers everywhere; add negative-value monitoring alert |
| 13 | Rate limiter not shared across pods | In-memory rate limiting (not Redis-backed) | Rate limit only effective per pod; global rate exceeded | Move to Redis-backed rate limiter (e.g., slowapi + Redis) | Architecture review: any per-request shared state must use external store |
| 14 | Silent model loading failure | Exception caught and swallowed in lifespan startup | Pods start, health checks pass, all predictions return 500 | Reraise exceptions in lifespan startup; they cause container exit (non-zero exit code) | Never swallow exceptions in startup; add startup smoke test |
| 15 | Request queue buildup on model reload | During hot-reload, new requests queue up waiting for lock | p99 latency spike for 30–60s during reload | Implement timeout on lock acquisition; return 503 if lock wait > 5s | Use graceful degradation during reload; cache last successful response |
| 16 | Stale Prometheus metrics after scale-in | Prometheus scrapes dead pod endpoints | Metrics show phantom traffic from removed pods | Configure Prometheus `--storage.tsdb.retention.time=15d`; use pod labels for routing | Use Kubernetes service discovery for Prometheus targets; pods removed from scrape on deletion |

---

## 8. General ML Interview Topics (MLOps)

**Q43. ⭐ Define SLA, SLO, and SLI. How do they apply to this ML API?**

**SLI (Service Level Indicator)**: a specific metric that measures service behavior. Examples: `http_request_duration_seconds` p99, error rate (5xx / total requests). **SLO (Service Level Objective)**: an internal target for an SLI. Examples: p99 latency < 500ms for 99.9% of time, error rate < 0.1% over 30 days. **SLA (Service Level Agreement)**: a contractual commitment to customers, with penalties for breach. SLA is typically less aggressive than SLO (e.g., SLA: p99 < 1s, SLO: p99 < 500ms). For the ML API: SLI = Prometheus `http_request_duration_seconds[0.99]`, SLO = < 500ms, SLA = < 1s. Error budget = 1 - SLO = 0.1% over 30 days = ~43 minutes of acceptable downtime.

**Q44. ⭐ What is an error budget and how does it drive ML operations decisions?**

Error budget = the amount of unreliability you are allowed under your SLO. If SLO is 99.9% availability, error budget is 0.1% = 43 minutes/month. Error budgets drive decisions: when budget is plentiful, prioritize new features and model deployments. When budget is low (> 50% consumed), freeze risky deployments, focus on reliability. When budget is exhausted, stop new deployments until the next measurement period. For ML APIs, model version upgrades are the most common risk — new models may have slightly higher latency or higher error rates. Automated rollback triggers when error budget burn rate exceeds 5× nominal.

**Q45. ⭐ What is "toil" in SRE and how does it apply to ML model deployment?**

Toil is manual, repetitive, automatable operational work that scales with service growth and provides no enduring value. ML-specific toil: manually updating model paths in deployment configs, manually running load tests after each release, manually checking Prometheus dashboards for drift, manually updating API documentation when model outputs change. The SRE goal is to keep toil < 50% of engineering time. Automate: model deployment pipelines (CI/CD), automated smoke tests after deployment, automated load tests as CI/CD gates, auto-generated API docs from Pydantic models (FastAPI does this).

**Q46. ⭐⭐ Describe a complete CI/CD pipeline for an ML model REST API.**

```
Stage 1: Code Quality (every PR)
  ├── pytest (unit + integration, mocked model)
  ├── mypy type checking
  ├── ruff linting
  └── bandit security scan

Stage 2: Build (on merge to main)
  ├── docker build
  ├── Container vulnerability scan (Trivy)
  └── docker push to registry

Stage 3: Deploy to Staging
  ├── Update Container App revision
  ├── Integration tests (real model, staging data)
  └── Performance tests (Locust, 5-minute run)

Stage 4: Deploy to Production
  ├── Blue-green deployment (10% traffic to new revision)
  ├── Monitor for 15 minutes (p99, error rate)
  ├── Automated rollback if p99 > 500ms or error rate > 0.5%
  └── Shift 100% traffic on success

Stage 5: Post-Deploy
  ├── Smoke tests (10 prediction requests)
  ├── Update model registry metadata
  └── Notify team via Slack
```

**Q47. ⭐ What is model governance and why does it matter for production ML APIs?**

Model governance is the set of processes ensuring ML models are developed, deployed, and operated responsibly, with appropriate oversight. Components: (1) **Model registry**: tracked versioning of all models with metadata (training data, performance metrics, approvals). (2) **Approval workflow**: models must be reviewed and approved before production deployment (technical + business + risk review). (3) **Audit trail**: every prediction logged with model version for retrospective analysis. (4) **Fairness monitoring**: track model performance across demographic groups. (5) **Rollback policy**: clear criteria and process for reverting to previous model version. Without governance, you risk deploying biased models, losing track of which model made a decision, and being unable to comply with regulatory audits.

---

## 9. Behavioral / Scenario Questions

**Q48. ⭐ Tell me about a time you had to debug a production API performance issue under pressure.**

Structure your answer with the STAR method (Situation, Task, Action, Result). Key points to include: (1) describe the detection (alert, customer report, dashboard spike), (2) describe the isolation process (which endpoint, which environment, which time window), (3) show systematic debugging (logs → metrics → traces → code), (4) describe the root cause (something specific: memory leak, N+1 query, wrong model loaded), (5) describe the fix and prevention (code change, monitoring improvement, runbook). Avoid vague answers like "I fixed the performance issue" — be specific about tools used (Prometheus, X-Ray, py-spy) and numbers (latency improved from 5s to 150ms).

**Q49. ⭐⭐ A model update caused a 300% latency increase in production 10 minutes after deployment. Walk me through your incident response.**

Step 1: Immediately open the incident. Step 2: Check if the latency SLO is breached — if yes, initiate rollback before debugging (MTTD < 5 min). Step 3: Roll back: shift 100% traffic to previous revision. Confirm latency returns to baseline. Step 4: Analyze the new model revision in staging: profile inference time, check model size, check batch processing changes. Step 5: Root cause analysis — common causes: new model is 3× larger (more layers), input preprocessing added synchronous I/O, new validation logic. Step 6: Fix the issue (optimize model, add async processing, reduce model size). Step 7: Write incident postmortem: timeline, root cause, action items (add latency gate to deployment pipeline). The key insight: rollback first, debug after. Never debug a live incident at the cost of extended customer impact.

**Q50. ⭐ How would you explain Pydantic v2 validation errors to a non-technical product stakeholder?**

"Our API has strict input rules to ensure the model receives data it was trained on. When a user sends a request that breaks these rules — for example, an empty text field, or text that's 50,000 characters long when we only support 10,000 — the API immediately returns a clear error message explaining what's wrong, instead of silently producing a bad prediction. These rules are defined as code, automatically documented in our API reference, and checked in under 1 millisecond. This is a feature, not a limitation: it prevents bad data from reaching the model and producing confusing results."

**Q51. ⭐⭐ You are designing the monitoring strategy for a new ML API. What are the first five metrics you instrument?**

(1) **p99 request latency** (`http_request_duration_seconds` histogram) — the single most important API health indicator, SLA-linked. (2) **Error rate** (5xx / total requests, as a rate) — detects model failures, validation regressions, infrastructure issues. (3) **Request rate** (RPS) — baseline for capacity planning, anomaly detection (sudden drop = possible upstream failure). (4) **Model prediction confidence distribution** (histogram of confidence scores) — leading indicator of model degradation before error rate spikes. (5) **Memory usage** (RSS per pod) — detects memory leaks, predicts OOM before it occurs. Add these five first; you can always add more. An alert on all five covers the primary failure modes.

---

## 10. Quick-Fire Questions

*Answer each in 1-3 sentences.*

**Q52. ⭐** What HTTP status code does FastAPI return for a Pydantic validation error?
**A:** 422 Unprocessable Entity, with a JSON body containing `detail` array describing each validation failure, field location, and error message.

**Q53. ⭐** What is the difference between `@app.get()` and `@router.get()`?
**A:** `@app.get()` registers a route directly on the main application. `@router.get()` registers on an `APIRouter` which is included in the app with `app.include_router(router, prefix="/v1")`, enabling modular route organization.

**Q54. ⭐** Why use `httpx` instead of `requests` in a FastAPI application?
**A:** `httpx` supports both sync and async HTTP clients. Using `httpx.AsyncClient` allows non-blocking outbound HTTP calls within async route handlers, preventing event loop blocking that `requests` would cause.

**Q55. ⭐** What is Uvicorn and how does it differ from Gunicorn?
**A:** Uvicorn is a single-process ASGI server (event loop). Gunicorn is a multi-process WSGI/ASGI server manager. In production, run Gunicorn with `UvicornWorker` class: Gunicorn manages multiple Uvicorn worker processes for CPU parallelism.

**Q56. ⭐** What does `response_model` do in a FastAPI route decorator?
**A:** It tells FastAPI to validate the return value against the specified Pydantic model, serialize it to JSON, exclude any extra fields not in the model, and document the response schema in OpenAPI.

**Q57. ⭐** How do you return a custom HTTP status code from a FastAPI route?
**A:** Use `response.status_code = 201` with a `Response` parameter, or use `JSONResponse(content=data, status_code=201)`, or decorate the route with `@router.post("/...", status_code=201)` for the default success code.

**Q58. ⭐** What is `Depends()` in FastAPI?
**A:** A dependency injection mechanism. `Depends(get_model_registry)` in a function parameter tells FastAPI to call `get_model_registry()` and inject the result. Dependencies can be async, can have sub-dependencies, and are called once per request (or cached within a request).

**Q59. ⭐** What is the purpose of `BaseHTTPMiddleware` vs pure ASGI middleware?
**A:** `BaseHTTPMiddleware` provides a simple `dispatch(request, call_next)` interface for writing middleware in a Django-like style. Pure ASGI middleware (implementing `__call__(scope, receive, send)`) is lower-level, more flexible, and has lower overhead — important for latency-sensitive middleware.

**Q60. ⭐** What does `pin_memory=True` do in PyTorch DataLoader?
**A:** Allocates tensors in pinned (page-locked) host memory, enabling faster asynchronous DMA transfers from CPU to GPU. It is effective only when using CUDA and increases host memory usage.

**Q61. ⭐** How do you add a global exception handler in FastAPI?
**A:** `@app.exception_handler(Exception)` decorator with `async def global_exception_handler(request, exc)` returning a `JSONResponse` with appropriate status code and error details.

**Q62. ⭐** What is the `X-Request-ID` header used for?
**A:** A unique identifier per request, used to correlate logs, traces, and metrics across distributed systems. If the client sends one, echo it back; if not, generate a UUID in middleware and attach it to all log entries for that request.

**Q63. ⭐** What is `model_dump()` in Pydantic v2?
**A:** The replacement for `.dict()` in Pydantic v1. Returns the model as a Python dictionary. Use `model_dump(exclude_none=True)` to omit null fields, `model_dump(include={'field1', 'field2'})` for partial serialization.

**Q64. ⭐** When does Locust report a "failure"?
**A:** When the response status code is 4xx or 5xx, when the connection times out, or when the test code explicitly calls `self.environment.events.request.fire(exception=...)`. Locust does not automatically fail on incorrect response bodies — you must add assertions in task code.

**Q65. ⭐⭐** What is the "thundering herd" problem in ML API startup?
**A:** When a new pod starts after scale-out, all N pending requests (that were queued) hit the pod simultaneously after model loading completes, causing a spike that may exceed the pod's capacity and crash it. Solution: gradual traffic ramping, readiness probe with a warm-up period, or an initial capacity limit.

**Q66. ⭐** What is `asyncio.get_event_loop()` vs `asyncio.get_running_loop()`?
**A:** `get_running_loop()` is preferred in async contexts — it raises `RuntimeError` if no loop is running, catching bugs early. `get_event_loop()` creates a new loop if none exists, which can create unexpected loops in test environments.

**Q67. ⭐** How do you configure FastAPI to return camelCase JSON instead of snake_case?
**A:** Use `model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)` with Pydantic's `to_camel` function, and set `response_model_by_alias=True` in route decorators.

**Q68. ⭐** What is content negotiation in HTTP APIs?
**A:** The client specifies preferred response format via the `Accept` header (e.g., `Accept: application/json` vs `application/msgpack`). The server selects the best matching format. FastAPI returns JSON by default; add `msgpack` support via custom response classes for binary-efficient payloads.

**Q69. ⭐⭐** What is request coalescing and when is it useful for ML APIs?
**A:** Multiple incoming requests for the same input are combined into one model call, with the result returned to all waiting callers. Useful when many users simultaneously request predictions for the same trending content (viral content scenario). Reduces model calls from N to 1 for N identical concurrent requests. Implement with a pending-requests dict keyed by input hash.

**Q70. ⭐** What does `--reload` do in Uvicorn development mode and why should it never be used in production?
**A:** `--reload` watches files for changes and restarts the server automatically. In production, it adds file-watching overhead, may restart on unintended file changes (logs, temp files), and does not support the multi-worker setup required for production capacity.

**Q71. ⭐** What is HTTP/2 and does FastAPI support it?
**A:** HTTP/2 supports multiplexing multiple requests over a single TCP connection, header compression, and server push. FastAPI/Uvicorn supports HTTP/2 via the `h2` library and `--http h2` flag, or via a proxy like nginx or Caddy terminating HTTP/2 and passing HTTP/1.1 to Uvicorn.

**Q72. ⭐** What is the purpose of `--limit-concurrency` in Uvicorn?
**A:** Sets the maximum number of concurrent connections. When exceeded, Uvicorn returns 503. Used to prevent OOM under DDoS or traffic spikes by rejecting excess requests early rather than queueing them and causing cascading timeouts.

**Q73. ⭐⭐** How does FastAPI handle background tasks, and what are their limitations?
**A:** `BackgroundTasks` runs callables after the response is sent: `background_tasks.add_task(log_prediction, text, result)`. Limitations: (1) no retry on failure, (2) no persistence (lost if pod restarts), (3) shares the event loop with request handling. For production: use a proper task queue (Celery, RQ, AWS SQS) for any background work that must not be lost.

**Q74. ⭐** What HTTP methods should the predict endpoint support? Should GET be used?
**A:** POST only. GET should not be used for ML predictions because: GET requests are cached by browsers and proxies (you don't want predictions cached at network layer), GET parameters are in the URL (logged in server access logs — PII risk), and the request body convention for ML inputs (JSON) is not standard in GET.

**Q75. ⭐** What is the difference between 502 Bad Gateway and 503 Service Unavailable?
**A:** 502 means the gateway/proxy received an invalid response from the upstream server (often a crash or protocol error). 503 means the server is temporarily unavailable (overloaded, starting up, or in maintenance). For ML APIs: return 503 when model is loading or rate limited; 502 typically comes from the load balancer when the FastAPI process is dead.

**Q76. ⭐** What is `Field(exclude=True)` in Pydantic v2?
**A:** Marks a field to be excluded from serialization (`.model_dump()`, `.model_dump_json()`). Useful for internal fields like `_raw_scores` that are used in computed fields but should not be exposed in the API response.

**Q77. ⭐⭐** Explain the retry storm anti-pattern and how to prevent it in ML API clients.
**A:** When an ML API returns 503 (overloaded), all clients retry simultaneously, generating 10× the original traffic and ensuring the server never recovers. Prevention: exponential backoff with jitter (`min(cap, base * 2^attempt) + random(0, jitter)`) and circuit breakers on the client side that stop retrying after N failures.

**Q78. ⭐** What does `app.state` provide in FastAPI?
**A:** A namespace for storing application-wide state, accessible from route handlers via `request.app.state.registry`. Set it during lifespan startup: `app.state.registry = registry`. It is the recommended way to share application-level resources without module-level globals.

**Q79. ⭐** How many request handlers can a single Uvicorn worker handle concurrently?
**A:** Theoretically unlimited for async handlers (all non-blocking I/O), limited by: event loop CPU overhead, OS file descriptor limits (default 1024, increase to 65536), and memory. Practically, benchmark shows ~1000 concurrent long-polling connections per worker; for CPU-bound inference, concurrency = 1 (event loop blocked during inference).

**Q80. ⭐⭐** What is the `anyio` library and how does it relate to FastAPI?
**A:** `anyio` is an asynchronous compatibility layer supporting both asyncio and Trio backends. FastAPI/Starlette uses `anyio` internally for async primitives, meaning FastAPI can theoretically run on Trio (better structured concurrency) as well as asyncio. As a developer, you can use `anyio.to_thread.run_sync()` as an alternative to `asyncio.run_in_executor()` for running sync code in a thread pool.

**Q81. ⭐** What is OpenAPI and how does FastAPI generate it?
**A:** OpenAPI (formerly Swagger) is a specification for describing HTTP APIs in JSON/YAML. FastAPI generates it automatically from route decorators (method, path, status codes), function type hints (request/response types), Pydantic model schemas, and docstrings. Access at `/docs` (Swagger UI) and `/openapi.json` (raw spec). The spec is generated at startup and cached.

**Q82. ⭐** How do you disable the OpenAPI docs in production?
**A:** `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`. Disabling in production prevents reconnaissance attacks (attackers can enumerate all endpoints from the spec) and hides implementation details. Keep docs enabled in internal/staging environments.

**Q83. ⭐⭐** What is a "dark launch" for an ML model feature?
**A:** Releasing a new ML model feature to production infrastructure but not showing its output to users — instead, logging the output for analysis. Allows real-traffic validation of model performance, latency, and error behavior without user-facing risk. Useful for validating new model versions before A/B testing. Implemented as a background task in the predict endpoint that calls the new model and logs results without including them in the response.

**Q84. ⭐** What is `uvloop` and how does it improve FastAPI performance?
**A:** `uvloop` is a high-performance event loop implementation for asyncio, written in Cython/C, based on `libuv` (the same C library used by Node.js). It replaces Python's default asyncio event loop. Performance improvement: 2–4× faster async I/O operations. Enable via `uvicorn --loop uvloop` or `import uvloop; uvloop.install()`.

**Q85. ⭐** How do you implement request ID propagation across microservices?
**A:** Generate a UUID in the outermost gateway or first service if `X-Request-ID` is absent. Pass it downstream in all outbound HTTP headers. Log it in every service. In FastAPI: add middleware that reads `X-Request-ID` from incoming headers (or generates one), stores it in `request.state.request_id`, and attaches it to the Python logging context so all log entries within that request carry the ID.

**Q86. ⭐⭐** What is structured logging and why is it essential for ML APIs?
**A:** Structured logging emits log records as JSON objects rather than free-form text strings. Each record has consistent keys: `timestamp`, `level`, `request_id`, `endpoint`, `model_version`, `latency_ms`, `status_code`. This enables: (1) querying logs with SQL-like syntax in CloudWatch Insights or Azure Log Analytics, (2) automatic metric extraction from logs, (3) correlation across services by `request_id`. Plain text logs cannot be reliably parsed at scale. Use `python-json-logger` or `structlog` in FastAPI.

**Q87. ⭐** What does the `Annotated` type hint do in Python and how does FastAPI use it?
**A:** `Annotated[T, metadata]` attaches metadata to a type hint without changing the type itself. FastAPI reads the metadata to extract `Field()` definitions, `Depends()` injections, and path/query/header parameter specifications. Example: `Annotated[str, Query(min_length=1, max_length=100)]` as a query parameter type.

**Q88. ⭐** What is middleware vs a dependency in FastAPI? When should you use each?
**A:** Middleware intercepts every request/response before routing — use for cross-cutting concerns that apply to all routes (timing, logging, CORS, rate limiting). Dependencies are per-route — use for route-specific logic (authentication for specific endpoints, injecting specific services, request validation beyond Pydantic). Dependencies are testable in isolation and appear in OpenAPI docs; middleware does not.

**Q89. ⭐⭐** How would you implement an admin endpoint to dump current model metrics?
**A:** Create a `GET /admin/metrics/model` endpoint with admin auth (API key or OAuth scope). Return: `model_version`, `requests_total`, `errors_total`, `p99_latency_ms` (read from Prometheus client registry), `model_load_time_s`, `uptime_seconds`, `active_requests`. Protect with `Depends(require_admin_token)`. This endpoint supplements Prometheus metrics with model-specific context that is difficult to express as pure metrics.

**Q90. ⭐** What is `orjson` and why might you use it with FastAPI?
**A:** `orjson` is a fast JSON library written in Rust, 2–10× faster than Python's standard `json` module. Use `ORJSONResponse` from FastAPI (or `fastapi.responses.ORJSONResponse`) as the default response class for high-QPS APIs where JSON serialization is measurable overhead. Particularly impactful for large batch responses or high-throughput (>5000 RPS) APIs.

**Q91. ⭐** What is `model_config` in Pydantic v2?
**A:** Replaces the inner `class Config` from Pydantic v1. It is a class-level attribute: `model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, frozen=True)`. Key options: `str_strip_whitespace` (auto-strip strings), `frozen` (immutable model, hashable), `extra='forbid'` (reject unexpected fields), `from_attributes=True` (ORM mode).

**Q92. ⭐** What is the default behavior when FastAPI receives a request with extra JSON fields not in the Pydantic model?
**A:** By default, Pydantic v2 ignores extra fields (they are silently dropped). Change to `model_config = ConfigDict(extra='forbid')` to return a 422 error if unexpected fields are present (useful for strict APIs to catch client-side bugs early) or `extra='allow'` to retain them in `model_extra`.

**Q93. ⭐⭐** How do you implement graceful shutdown in FastAPI to drain in-flight requests?
**A:** Uvicorn handles `SIGTERM` by: (1) stopping accepting new connections, (2) waiting for `--graceful-timeout` seconds (default 30s) for in-flight requests to complete, (3) then forcefully terminating. In the lifespan's shutdown section (after `yield`), release resources: close database connections, flush log buffers, write a "shutting down" metric. Kubernetes sends SIGTERM and waits `terminationGracePeriodSeconds` (default 30s) before SIGKILL.

---

*End of ML Model REST API Interview Preparation Guide — 93 numbered questions + follow-up chains*
*Total coverage: ~200+ distinct questions including follow-ups*
