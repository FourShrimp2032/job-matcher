from fastapi import FastAPI

from app.database import Base, engine
from app.routers import candidates, jobs, matches

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="JobMatch AI",
    description="AI-powered CV and job vacancy matching API.",
    version="0.1.0",
)

app.include_router(candidates.router)
app.include_router(jobs.router)
app.include_router(matches.router)


@app.get("/")
def healthcheck():
    return {
        "name": "JobMatch AI",
        "status": "ok",
        "docs": "/docs",
    }
