# BBAP-Sec — Real-Time Vision ML Serving Platform

**Production-oriented computer vision inference platform for real-time object detection.**

BBAP-Sec is an extensible ML serving platform built with **FastAPI and Ultralytics YOLO11**. It focuses on the serving side of machine learning: model registry, dynamic batching, structured errors, request tracing, and measured performance.

<p align="center">
  <img width="600" alt="Object detection example" src="tests/fixtures/streetAndpeople.jpg">
</p>

---

## Key Features

**Implemented**

* REST API for object detection (`POST /api/v1/inference`)
* YOLO11 via an Ultralytics model adapter
* Model registry with YAML configuration and model metadata
* **Dynamic, model-aware batching** — concurrent requests are grouped into a single forward pass
* Non-blocking inference (model runs in a worker thread, the event loop stays responsive)
* Structured error responses with request IDs (`X-Request-ID`)
* Request validation with Pydantic
* Health and readiness endpoints
* Environment-based configuration (batching can be switched on/off per deployment)
* Concurrent load benchmark (p50 / p95 / throughput / average batch size)
* Unit and integration tests, Ruff linting and formatting

**Planned**

* Docker / Docker Compose deployment
* Prometheus metrics and Grafana dashboards
* RTSP / camera streaming pipeline with WebSocket results
* Object tracking
* MLflow experiment tracking and DVC dataset versioning

---

## Architecture

```text
                 ┌─────────────────┐
                 │     Client      │
                 └────────┬────────┘
                          │  HTTP (image)
                 ┌────────▼────────┐
                 │     FastAPI     │  validation · request ID · errors
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Inference Engine│
                 └────────┬────────┘
                          │  submit() → Future
                 ┌────────▼────────┐
                 │ Dynamic Batcher │  max_batch_size · max_wait_ms
                 └────────┬────────┘  one queue, batches split per model
                          │
                 ┌────────▼────────┐
                 │ Model Registry  │
                 └────────┬────────┘
                          │  predict_batch() in worker thread
                 ┌────────▼────────┐
                 │  YOLO11 (Ultra- │
                 │   lytics)       │
                 └─────────────────┘
```

### Request lifecycle

1. The client uploads an image to `/api/v1/inference`.
2. Middleware assigns a request ID; the image is validated and decoded.
3. The engine submits the image to the dynamic batcher and awaits a `Future`.
4. The batcher collects requests for the same model until the batch is full or `max_wait_ms` expires.
5. The whole batch runs through YOLO11 in **one forward pass**, off the event loop.
6. Each request receives only its own detections, plus `inference_time_ms` and `batch_size`.

### Planned real-time pipeline

```text
Camera / RTSP → Frame pipeline → Inference → Tracking → WebSocket → Client
```

---

## Tech Stack

| Layer               | Technology                     | Status  |
| ------------------- | ------------------------------ | ------- |
| Language            | Python 3.11+                   | ✅      |
| API                 | FastAPI                        | ✅      |
| ML                  | Ultralytics YOLO11             | ✅      |
| Validation          | Pydantic / pydantic-settings   | ✅      |
| Computer Vision     | OpenCV                         | ✅      |
| Testing             | Pytest / pytest-asyncio        | ✅      |
| Code Quality        | Ruff / MyPy                    | ✅      |
| Containerization    | Docker                         | Planned |
| Monitoring          | Prometheus / Grafana           | Planned |
| Experiment Tracking | MLflow                         | Planned |
| Data Versioning     | DVC                            | Planned |

---

## Project Structure

```text
app/
├── api/          # Routes, schemas, dependencies
├── core/         # Configuration, logging, exceptions, request IDs
├── inference/    # Engine, dynamic batcher, pre/post-processing
├── models/       # Model interface, registry, Ultralytics adapter
└── streaming/    # Camera / RTSP / WebSocket pipeline (planned)

configs/          # Model and environment configuration
tests/            # Unit and integration tests
scripts/          # Benchmark scripts
models/           # Local model artifacts (e.g. yolo11n.pt)
doc/              # Architecture and API notes
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ties2/Real-Time-Vision-Platform
cd Real-Time-Vision-Platform
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
make install
```

### 4. Configure

```bash
cp .env.example .env
```

Place the model weights in `models/` (default: `models/yolo11n.pt`). The device is set in `configs/models.yaml` (`auto`, `cpu`, `cuda`, or `mps` on Apple Silicon).

### 5. Run

```bash
make run
```

* API: `http://localhost:8000`
* Interactive docs: `http://localhost:8000/docs`
* Health check: `curl http://localhost:8000/health`

---

## API Usage

```bash
curl -X POST "http://localhost:8000/api/v1/inference?model=yolo11" \
  -F "file=@tests/fixtures/streetAndpeople.jpg"
```

Example response (truncated to 3 of 10 detections):

```json
{
  "model": "yolo11",
  "model_version": "1.0.0",
  "inference_time_ms": 438.23,
  "batch_size": 1,
  "image_width": 1024,
  "image_height": 677,
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.8954,
      "bbox": { "x1": 239.05, "y1": 296.70, "x2": 420.01, "y2": 384.32 }
    },
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8894,
      "bbox": { "x1": 483.91, "y1": 238.74, "x2": 576.55, "y2": 499.45 }
    },
    {
      "class_id": 9,
      "class_name": "traffic light",
      "confidence": 0.8108,
      "bbox": { "x1": 169.67, "y1": 123.49, "x2": 200.42, "y2": 202.48 }
    }
  ]
}
```

`inference_time_ms` is the model time for the batch this request was part of; `batch_size` shows how many requests shared that forward pass.

Errors are returned in a consistent format:

```json
{
  "code": "INVALID_INPUT",
  "message": "Unable to decode image.",
  "request_id": "3f1c2a..."
}
```

---

## Configuration

Dynamic batching is configured through environment variables (or `.env`):

| Variable                   | Default | Description                                  |
| -------------------------- | ------- | -------------------------------------------- |
| `BATCHING__ENABLED`        | `true`  | Disable to process every request on its own  |
| `BATCHING__MAX_BATCH_SIZE` | `4`     | Maximum requests per forward pass            |
| `BATCHING__MAX_WAIT_MS`    | `10`    | Maximum time to wait for a batch to fill     |

---

## Performance

Benchmarks use `scripts/benchmark_concurrent.py`: concurrent clients send the same image, warm-up requests are sent at the same concurrency and discarded (so every batch shape is compiled before measuring), then p50 / p95 latency, throughput, and average batch size are reported.

```bash
# Terminal 1 — pick one
make serve              # batching enabled
make serve-no-batch     # batching disabled

# Terminal 2
make benchmark-concurrent                  # 8 clients, 200 requests
make benchmark-concurrent CONCURRENCY=16   # custom load
make benchmark-sequential                  # single client baseline
```

**Environment:** Apple Silicon (MPS), YOLO11n, 200 requests, 40 warm-up requests.

| Setup                          | Concurrency | p50 (ms) | p95 (ms) | Throughput (req/s) | Avg batch |
| ------------------------------ | ----------- | -------- | -------- | ------------------ | --------- |
| Single client                  | 1           | 43.92    | 45.39    | 22.80              | 1.00      |
| Batching disabled              | 8           | 248.41   | 437.45   | 29.40              | 1.00      |
| Batching enabled (max 4, 10ms) | 8           | 203.01   | 296.15   | **36.66**          | 3.91      |

**Result (8 concurrent clients, batching enabled vs. disabled):**

* **+24.7% throughput** (29.40 → 36.66 req/s)
* **−18.3% p50 latency** (248.41 → 203.01 ms)
* **−32.3% p95 latency** (437.45 → 296.15 ms)

Latency under load includes queueing time: with 8 clients in flight, mean latency ≈ concurrency / throughput (Little's law), so it is not comparable to the single-client row. Batching improved throughput and tail latency at the same time because the queue drains faster.

> Batching gains depend on model size and hardware. Small models such as YOLO11n may not saturate the accelerator, which is why batching is configurable per deployment rather than always on.

---

## Development

```bash
make test        # run tests
make lint        # Ruff lint
make format      # Ruff format
make typecheck   # MyPy
make check       # lint + format check + tests
```

---

## Roadmap

### Phase 1 — Foundation

* [x] Project architecture
* [x] FastAPI application
* [x] Configuration management
* [x] Logging
* [x] Health/readiness endpoints
* [x] Model abstraction
* [x] Model registry foundation
* [x] Automated tests

### Phase 2 — Model Serving

* [x] Ultralytics model adapter
* [x] YOLO11 inference
* [x] Image inference API
* [x] Model metadata
* [x] Structured errors and request IDs
* [x] Dynamic, model-aware batching
* [x] Concurrent benchmark
* [x] Published benchmark results (batching on vs. off)

### Phase 3 — Production

* [ ] Docker / Docker Compose
* [ ] Prometheus metrics
* [ ] Grafana dashboards
* [ ] CI/CD
* [ ] Security hardening
* [ ] GPU optimization

### Phase 4 — Real-Time Vision

* [ ] Webcam source
* [ ] RTSP source
* [ ] Frame pipeline
* [ ] WebSocket streaming
* [ ] Object tracking
* [ ] FPS / latency optimization

### Phase 5 — MLOps

* [ ] MLflow experiments
* [ ] Model version management
* [ ] DVC datasets