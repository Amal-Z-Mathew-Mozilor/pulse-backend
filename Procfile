web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
worker: PYTHONPATH=. procrastinate --app=app.worker.app worker
