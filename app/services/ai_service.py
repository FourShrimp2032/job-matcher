from openai import OpenAI

from app.schemas import CandidateAIProfile, JobAIProfile
from app.settings import settings


class AIServiceError(RuntimeError):
    pass


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise AIServiceError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your API key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def parse_candidate_cv(cv_text: str) -> CandidateAIProfile:
    client = _client()

    response = client.responses.parse(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You extract a candidate profile from a CV. "
                    "Only use evidence present in the CV. Do not invent skills or experience. "
                    "Normalize obvious aliases when useful, e.g. bs4 -> BeautifulSoup. "
                    "Estimate experience_years conservatively from explicit work/project dates or descriptions. "
                    "If English level is not stated, return null."
                ),
            },
            {"role": "user", "content": cv_text},
        ],
        text_format=CandidateAIProfile,
    )

    if response.output_parsed is None:
        raise AIServiceError("The model did not return a structured candidate profile.")

    return response.output_parsed


def parse_job_description(description: str) -> JobAIProfile:
    client = _client()

    response = client.responses.parse(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You extract structured hiring requirements from a job description. "
                    "Classify requirements as required, preferred, or optional based only on wording in the vacancy. "
                    "Do not promote nice-to-have skills to required. "
                    "Normalize obvious technology aliases. "
                    "If years of experience or English level are not stated, return null."
                ),
            },
            {"role": "user", "content": description},
        ],
        text_format=JobAIProfile,
    )

    if response.output_parsed is None:
        raise AIServiceError("The model did not return a structured job profile.")

    return response.output_parsed


def explain_match(candidate_profile: dict, job_profile: dict, match_result: dict):
    """Explain an already-calculated match without changing its score or recommendation."""
    import json

    from app.schemas import MatchAIExplanation

    client = _client()
    payload = {
        "candidate_profile": candidate_profile,
        "job_profile": job_profile,
        "calculated_match": match_result,
    }

    response = client.responses.parse(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You explain a CV-to-job match that has already been calculated by backend code. "
                    "Do not alter, recalculate, or contradict the score or APPLY/MAYBE/SKIP recommendation. "
                    "Use only the supplied candidate and vacancy data. Be concise and practical. "
                    "Strengths should explain relevant evidence. Gaps should prioritize required missing skills. "
                    "Interview focus should contain likely preparation topics based on the vacancy and gaps."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text_format=MatchAIExplanation,
    )

    if response.output_parsed is None:
        raise AIServiceError("The model did not return a structured match explanation.")

    return response.output_parsed
