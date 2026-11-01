from celery import Celery

celery_app = Celery(
    "lora_platform",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)