"""FastAPI entrypoint. Run: uvicorn app.api.main:app --reload"""

from fastapi import FastAPI

app = FastAPI(title="AutoML Playground")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
