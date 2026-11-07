# LoRA Fine-Tune & Serve Platform

A small backend system that trains LoRA adapters asynchronously and serves them from a shared base model. Training runs as a background job, serving is a synchronous API, and the two are connected by a Redis queue. The whole system runs under Docker Compose or on Kubernetes.

The ML part is deliberately small (one model, one adapter format). The focus is the backend around it: async job processing, shared state, containerisation, a bounded concurrency-safe cache and Kubernetes deployment.

## What it does

1. **Upload** a dataset (JSONL, one `{"text": "..."}` per line). The server validates it and returns a `dataset_id`.
2. **Fine-tune**: submit a job for a base model and a `dataset_id`. A worker trains a LoRA adapter in the background while you poll job status.
3. **Generate**: send a prompt and a `job_id`. The API attaches that adapter to the base model and returns generated text.

## Architecture

```
                 +----------------------------------------------+
 client -------> |  API (FastAPI)                               |
                 |  /datasets/upload  /finetune  /generate      |
                 +-------+--------------------------+-----------+
                         | enqueue job_id           | load adapter
                         v                          v
                    +---------+            +------------------------+
                    |  Redis  |            | Adapter cache (LRU,    |
                    | (queue) |            | max 3, thread-safe)    |
                    +----+----+            | over one shared base   |
                         | consume         | model in memory        |
                         v                 +------------------------+
                 +----------------+
                 | Worker (Celery)|  trains LoRA adapter (PEFT + Trainer)
                 +-------+--------+
                         | writes adapter files, updates job status
                         v
          +--------------------------------------+
          | Shared volume: registry.db (SQLite), |
          | datasets/, adapters/                 |
          +--------------------------------------+
```

**Training path:** `POST /finetune` writes a job row (`queued`), sends the `job_id` to Redis and returns immediately. The worker marks the job `training`, trains, saves the adapter, then marks it `ready` (or `failed`).

**Serving path:** `POST /generate` looks up the job, checks it is `ready`, gets the adapter from the cache (loading it from disk on a miss) and runs inference.

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Pydantic, Uvicorn |
| Async jobs | Celery, Redis |
| State | SQLite via SQLAlchemy |
| ML | PyTorch, Hugging Face `transformers`, `peft`, `datasets` |
| Packaging | Docker, Docker Compose |
| Orchestration | Kubernetes (tested on `kind`) |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/datasets/upload` | Upload a JSONL file (multipart). Returns `dataset_id` and `num_examples` |
| `POST` | `/finetune` | Body: `{"base_model": "distilgpt2", "dataset_id": "..."}`. Returns `job_id` |
| `GET` | `/finetune/{job_id}` | Job status: `queued`, `training`, `ready` or `failed` |
| `POST` | `/generate` | Body: `{"job_id": "...", "prompt": "...", "max_new_tokens": 50}` |

Dataset format, one JSON object per line:
```json
{"text": "The weather today is sunny and warm."}
{"text": "Kubernetes helps manage containerized applications at scale."}
```
The upload endpoint rejects the file if any line is invalid JSON or is missing the `text` field. Clients never supply file paths. The server generates the IDs and decides where files are saved.

## Run it

### Option 1: Docker Compose

```bash
docker compose up --build
```
The first build takes several minutes because of PyTorch. This starts three services (`api`, `worker`, `redis`). The API listens on `http://127.0.0.1:8000`, and interactive docs are at `/docs`.

```bash
docker compose down        # stop, keep data
docker compose down -v     # stop and delete the volumes (database, datasets, adapters)
```

### Option 2: Kubernetes (kind)

Requires Docker, `kind` and `kubectl`.

```bash
docker build -t lora-platform:latest .
kind create cluster --name lora
kind load docker-image lora-platform:latest --name lora   # the cluster can't see local images otherwise
kubectl apply -f k8s/
kubectl get pods                                          # wait for redis, api and worker to be Running
kubectl port-forward service/api 8000:8000                # keep this terminal open
```
The image is several GB because of PyTorch, so the `kind load` step takes a few minutes.

### Try the full flow

```bash
curl -F "file=@sample.jsonl" http://127.0.0.1:8000/datasets/upload
# -> {"dataset_id": "...", "num_examples": 5}

curl -X POST http://127.0.0.1:8000/finetune \
  -H "Content-Type: application/json" \
  -d '{"base_model": "distilgpt2", "dataset_id": "<dataset_id>"}'
# -> {"job_id": "...", "status": "queued"}

curl http://127.0.0.1:8000/finetune/<job_id>              # repeat until "status": "ready"

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job_id>", "prompt": "Hello", "max_new_tokens": 20}'
```
The first job is slower because the worker downloads the base model into a cached volume.

## Design decisions

- **Queue and database do different jobs.** Redis (via Celery) only carries the message "run job X". It is fast and disposable. SQLite holds the durable record of what each job is and what state it is in. The API and worker never call each other. They coordinate through the queue and the shared database.
- **Training is decoupled from serving.** Training takes seconds to hours, so the API returns immediately and the client polls. The API and worker are separate processes, so they can be scaled and restarted independently.
- **The cache holds loaded models in process memory, not in Redis.** A loaded model is a large in-memory object. Serialising it through Redis would cost more than reloading it from disk, which defeats the purpose.
- **The adapter cache is a bounded LRU with a lock.** It holds at most 3 adapters. When a 4th is requested, the least recently used one is evicted. FastAPI runs the sync route on a thread pool, so a `threading.Lock` makes check, evict and load a single critical section.
- **Adapters are tiny.** LoRA freezes the base model and trains small low-rank matrices (rank 8, attention layers only), so each adapter is a small file rather than a full model copy.
- **One image, two commands.** The API and worker share one Docker image and differ only in their start command. PyTorch is installed once, and the same image runs in Compose and Kubernetes.

## Measured results

- **Adapter size:** 594 KB, against roughly 330 MB for the `distilgpt2` base model (about 0.18%).
- **Training time:** about 9 seconds of training for a 5-example, 3-epoch run on CPU in the worker container. The first job also spends about 20 seconds downloading the base model.
- **LRU eviction:** with the cache size set to 3, requesting a 4th distinct adapter logs `Evicting adapter <id> to make room for <id>`. This was observed with real trained adapters.
- **Kubernetes:** the full upload, fine-tune, generate flow works across separate API, worker and Redis pods sharing one persistent volume. Deleting the Redis pod causes the Deployment to recreate it.

No load-test numbers yet.

## Known limitations and next steps

- **Single API replica.** SQLite on a `ReadWriteOnce` volume allows only one writer. Running several API replicas or autoscaling needs Postgres for job state and S3 or similar object storage for datasets and adapters.
- **The cache is per process.** Each API replica would have its own cache, and a loaded model can't be shared between processes.
- **Adapter isolation is not yet verified.** Adapters are attached with `PeftModel.from_pretrained` onto a shared in-memory base model object. Before relying on concurrent requests for different adapters, this needs a test with adapters trained on clearly different data, and probably a move to PEFT's `load_adapter` / `set_adapter` API.
- **No auth or rate limiting.** There is no per-user isolation, and any client can use any `job_id`. Uploads have no size limit.
- **No Kubernetes probes or resource limits yet.**
- **CPU-only, tiny models.** Training is forced to CPU and tested only with `distilgpt2` on a 5-example dataset, so outputs barely differ from the base model. This validates the pipeline, not model quality.
- **`AdapterJob` is defined in both `api/main.py` and `worker/tasks.py`.** It should live in one shared module.
- **No CI or automated tests.**

## Project layout

```
lora-platform/
├── Dockerfile              # one image for api and worker
├── docker-compose.yml      # redis + api + worker, shared volumes
├── requirements.txt
├── sample.jsonl            # tiny example dataset
├── api/
│   └── main.py             # FastAPI app, job registry, adapter cache, /generate
├── worker/
│   └── tasks.py            # Celery app and the LoRA training task
└── k8s/
    ├── storage.yaml        # PersistentVolumeClaims
    ├── redis.yaml          # Deployment + Service
    ├── api.yaml            # Deployment + Service
    └── worker.yaml         # Deployment
```