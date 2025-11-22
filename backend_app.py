# backend/app.py

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend_scoring import score_transcript


app = FastAPI(
    title="Nirmaan AI – Self Introduction Scoring API",
    description="Scores student self-introduction transcripts using rule-based + NLP + rubric weighting.",
    version="1.0.0",
)

# For simplicity, allow all origins (fine for prototype)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    transcript: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/score")
def score_endpoint(payload: ScoreRequest):
    result = score_transcript(payload.transcript)
    return result
