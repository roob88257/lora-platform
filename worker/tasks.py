import time
from datetime import datetime, timezone
import os
from celery import Celery
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "lora_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

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


def train_lora_adapter_task(job_id: str, base_model: str, dataset_path: str):
    db = SessionLocal()
    job = db.get(AdapterJob, job_id)
    if not job:
        db.close()
        return

    job.status = "training"
    db.commit()

    try:
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"] if "gpt2" in base_model else ["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

        output_dir = f"./adapters/{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        job.adapter_path = output_dir
        job.status = "ready"
        db.commit()

    except Exception as e:
        job.status = "failed"
        db.commit()
        raise
    finally:
        db.close()


@celery_app.task
def train_lora_adapter(job_id: str, base_model: str, dataset_path: str):
    train_lora_adapter_task(job_id, base_model, dataset_path)