# FastAPI app: /finetune, /finetune/{job_id}, /generate, /health, /ready endpoints + job registry DB model. To be implemented.
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi import FastAPI, HTTPException
from worker.tasks import train_lora_adapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from api.adapters import AdapterManager
import os
import threading


app = FastAPI()

DB_URL = "sqlite:///./registry.db"

Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class AdapterJob(Base):
    __tablename__ = "adapter_jobs"
    id = Column(String, primary_key=True)
    status = Column(String, default="queued")
    base_model = Column(String)
    dataset_path = Column(String)
    adapter_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(engine)
@app.get("/health")
def health():
    return {"status": "ok"}

class FinetuneRequest(BaseModel):
    base_model: str = "distilgpt2"
    dataset_id: str

@app.post("/finetune")
def finetune(req: FinetuneRequest):
    dataset_path = f"datasets/{req.dataset_id}.jsonl"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="dataset not found")

    job_id = str(uuid.uuid4())
    db = SessionLocal()
    job = AdapterJob(
        id=job_id,
        status="queued",
        base_model=req.base_model,
        dataset_path=dataset_path,
    )
    db.add(job)
    db.commit()
    db.close()

    train_lora_adapter.delay(job_id, req.base_model, dataset_path)
    return {"job_id": job_id, "status": "queued"}

@app.get("/finetune/{job_id}")
def job_status(job_id: str):
    db = SessionLocal()
    job = db.get(AdapterJob, job_id)
    db.close()

    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job.id,
        "status": job.status,
        "base_model": job.base_model,
        "dataset_path": job.dataset_path,
    }

_managers = {}              # base_model name -> (AdapterManager, tokenizer)
_managers_lock = threading.Lock()


def _get_manager(base_model: str):
    if base_model not in _managers:
        with _managers_lock:
            if base_model not in _managers:
                tok = AutoTokenizer.from_pretrained(base_model)
                base = AutoModelForCausalLM.from_pretrained(base_model)
                _managers[base_model] = (AdapterManager(base), tok)
    return _managers[base_model]


class GenerateRequest(BaseModel):
    job_id: str
    prompt: str
    max_new_tokens: int = 50


@app.post("/generate")
def generate(req: GenerateRequest):
    db = SessionLocal()
    job = db.get(AdapterJob, req.job_id)
    db.close()

    if not job or job.status != "ready":
        raise HTTPException(status_code=404, detail="adapter not found or not ready")

    manager, tokenizer = _get_manager(job.base_model)
    text = manager.generate(req.job_id, job.adapter_path, tokenizer, req.prompt, req.max_new_tokens)

    return {"text": text, "job_id": req.job_id}

import uuid
import json
from fastapi import UploadFile, File

@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()

    lines = contents.decode("utf-8").strip().split("\n")
    if len(lines) < 1:
        raise HTTPException(status_code=400, detail="dataset is empty")

    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"line {i+1} is not valid JSON")
        if "text" not in row:
            raise HTTPException(status_code=400, detail=f"line {i+1} is missing a 'text' field")

    dataset_id = str(uuid.uuid4())
    os.makedirs("datasets", exist_ok=True)
    save_path = f"datasets/{dataset_id}.jsonl"
    with open(save_path, "wb") as f:
        f.write(contents)

    return {"dataset_id": dataset_id, "num_examples": len(lines)}