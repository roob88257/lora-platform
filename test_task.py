from worker.tasks import add

result = add.delay(3, 4)
print("Task sent, waiting for result...")
print(result.get(timeout=10))