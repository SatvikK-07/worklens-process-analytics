from fastapi import FastAPI

from backend.routes import analytics, cases, health, models, predictions

app = FastAPI(
    title="WorkLens AI API",
    description="Guarded prediction and process-analytics endpoints.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(cases.router)
app.include_router(analytics.router)
app.include_router(models.router)
