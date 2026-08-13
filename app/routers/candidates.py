from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Candidate
from app.schemas import CandidateCreate, CandidateRead
from app.services.ai_service import AIServiceError, parse_candidate_cv
from app.services.pdf_service import (
    PDFExtractionError,
    extract_text_from_pdf,
)

router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.post("", response_model=CandidateRead, status_code=status.HTTP_201_CREATED)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    try:
        profile = parse_candidate_cv(payload.cv_text)
    except AIServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc

    candidate = Candidate(
        name=payload.name,
        email=payload.email,
        cv_text=payload.cv_text,
        profile=profile.model_dump(),
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return candidate

@router.post(
    "/upload",
    response_model=CandidateRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_candidate_cv(
    name: str = Form(...),
    email: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_bytes = await file.read()

    max_file_size = 5 * 1024 * 1024

    if len(file_bytes) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail="PDF file is too large. Maximum size is 5 MB.",
        )

    try:
        cv_text = extract_text_from_pdf(file_bytes)

    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if len(cv_text) < 30:
        raise HTTPException(
            status_code=400,
            detail="The PDF does not contain enough readable CV text.",
        )

    try:
        profile = parse_candidate_cv(cv_text)

    except AIServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI provider error: {exc}",
        ) from exc

    candidate = Candidate(
        name=name,
        email=email,
        cv_text=cv_text,
        profile=profile.model_dump(),
    )

    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return candidate

@router.get("", response_model=list[CandidateRead])
def list_candidates(db: Session = Depends(get_db)):
    return db.scalars(select(Candidate).order_by(Candidate.created_at.desc())).all()


@router.get("/{candidate_id}", response_model=CandidateRead)
def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate
