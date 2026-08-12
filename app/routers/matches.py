from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Candidate, Job, Match
from app.schemas import MatchCreate, MatchRead
from app.services.ai_service import explain_match
from app.services.matching_service import calculate_match

router = APIRouter(prefix="/matches", tags=["Matches"])


@router.post("", response_model=MatchRead, status_code=status.HTTP_201_CREATED)
def create_or_update_match(payload: MatchCreate, db: Session = Depends(get_db)):
    candidate = db.get(Candidate, payload.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    job = db.get(Job, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = calculate_match(candidate.profile, job.profile)

    # AI only explains the deterministic result; it is not allowed to choose the score.
    try:
        explanation = explain_match(candidate.profile, job.profile, result)
        result["ai_explanation"] = explanation.model_dump()
    except Exception as exc:
        # Matching is still useful even if the explanation API call fails.
        result["ai_explanation"] = {
            "summary": "Match calculated successfully, but AI explanation was unavailable.",
            "strengths": [],
            "gaps": result.get("required_missing", []),
            "interview_focus": [],
            "error": str(exc),
        }

    match = db.scalar(
        select(Match).where(
            Match.candidate_id == payload.candidate_id,
            Match.job_id == payload.job_id,
        )
    )

    if match:
        match.score = result["score"]
        match.recommendation = result["recommendation"]
        match.details = result
    else:
        match = Match(
            candidate_id=payload.candidate_id,
            job_id=payload.job_id,
            score=result["score"],
            recommendation=result["recommendation"],
            details=result,
        )
        db.add(match)

    db.commit()
    db.refresh(match)
    return match


@router.get("", response_model=list[MatchRead])
def list_matches(db: Session = Depends(get_db)):
    return db.scalars(select(Match).order_by(Match.score.desc())).all()


@router.get("/{match_id}", response_model=MatchRead)
def get_match(match_id: int, db: Session = Depends(get_db)):
    match = db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match
