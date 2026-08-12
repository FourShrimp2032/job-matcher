import re
from difflib import SequenceMatcher


IMPORTANCE_WEIGHT = {
    "required": 1.0,
    "preferred": 0.5,
    "optional": 0.2,
}

SKILLS_WEIGHT = 0.75
EXPERIENCE_WEIGHT = 0.15
ENGLISH_WEIGHT = 0.10

CEFR_LEVELS = {
    "a1": 1,
    "a2": 2,
    "b1": 3,
    "b2": 4,
    "c1": 5,
    "c2": 6,
}
UNKNOWN_EVIDENCE_SCORE = 50.0

MATCH_VALUE = {
    "match": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}

ALIASES = {
    "bs4": "beautifulsoup",
    "beautiful soup": "beautifulsoup",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "git hub": "github",
    "rest api development": "rest api",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
}

RELATED_SKILL_GROUPS = [
    {
        "rest api",
        "backend api",
        "backend api development",
        "api development",
        "web api",
        "http api",
        "restful services",
    },
]

def normalize_skill(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[._/+\-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return ALIASES.get(value, value)

def _same_related_group(left: str, right: str) -> bool:
    for group in RELATED_SKILL_GROUPS:
        if left in group and right in group:
            return True

    return False


def compare_skill(requirement: str, candidate_skills: list[str]) -> tuple[str, str | None, float]:
    required = normalize_skill(requirement)
    normalized_candidates = [(skill, normalize_skill(skill)) for skill in candidate_skills]

    for original, normalized in normalized_candidates:
        if normalized == required:
            return "match", original, 1.0


    for original, normalized in normalized_candidates:
        if _same_related_group(required, normalized):
            return "partial", original, 0.78

    best_skill = None
    best_ratio = 0.0
    for original, normalized in normalized_candidates:
        ratio = SequenceMatcher(None, required, normalized).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_skill = original

    if best_ratio >= 0.88:
        return "match", best_skill, best_ratio
    if best_ratio >= 0.68:
        return "partial", best_skill, best_ratio

    return "missing", None, best_ratio

def _calculate_experience(
    candidate_years: float | None,
    required_years: float | None,
) -> tuple[float, str]:

    if required_years is None or required_years <= 0:
        return 100.0, "not_required"

    if candidate_years is None:
        return UNKNOWN_EVIDENCE_SCORE, "unknown"

    if candidate_years >= required_years:
        return 100.0, "meets"

    if candidate_years > 0:
        score = (candidate_years / required_years) * 100
        return score, "partial"

    return 0.0, "below"

def _extract_cefr(value: str | None) -> int | None:
    if not value:
        return None

    match = re.search(r"\b([abc][12])\b", value.lower())

    if not match:
        return None

    return CEFR_LEVELS[match.group(1)]

def _calculate_english(
    candidate_level: str | None,
    required_level: str | None,
) -> tuple[float, str]:

    required = _extract_cefr(required_level)

    if required is None:
        return 100.0, "not_required"

    candidate = _extract_cefr(candidate_level)

    if candidate is None:
        return UNKNOWN_EVIDENCE_SCORE, "unknown"

    if candidate >= required:
        return 100.0, "meets"

    if candidate == required - 1:
        return 60.0, "partial"

    return 0.0, "below"

def calculate_match(candidate_profile: dict, job_profile: dict) -> dict:
    candidate_skill_objects = candidate_profile.get("skills", [])
    candidate_skills = [item.get("name", "") for item in candidate_skill_objects if item.get("name")]
    requirements = job_profile.get("requirements", [])

    weighted_total = 0.0
    weighted_earned = 0.0
    skill_results = []

    for requirement in requirements:
        skill = requirement.get("skill", "").strip()
        if not skill:
            continue

        importance = requirement.get("importance", "required")
        weight = IMPORTANCE_WEIGHT.get(importance, 1.0)
        status, candidate_skill, confidence = compare_skill(skill, candidate_skills)


        strength = MATCH_VALUE[status]

        weighted_total += weight
        weighted_earned += weight * strength

        skill_results.append(
            {
                "requirement": skill,
                "importance": importance,
                "candidate_skill": candidate_skill,
                "status": status,
                "similarity": round(confidence, 2),
                "earned": round(weight * strength, 3),
                "possible": weight,
            }
        )

    skills_score = (weighted_earned / weighted_total * 100) if weighted_total else 0.0

    candidate_years = candidate_profile.get("experience_years")
    required_years = job_profile.get("experience_years_required")

    experience_score, experience_status = _calculate_experience(
        candidate_years,
        required_years,
    )
    candidate_english = candidate_profile.get("english_level")
    required_english = job_profile.get("english_level")

    english_score, english_status = _calculate_english(
        candidate_english,
        required_english,
    )

    final_score = (
        skills_score * SKILLS_WEIGHT
        + experience_score * EXPERIENCE_WEIGHT
        + english_score * ENGLISH_WEIGHT
    )
    final_score = round(final_score, 1)

    required_missing = [
        item["requirement"]
        for item in skill_results
        if item["importance"] == "required" and item["status"] == "missing"
    ]

    if final_score >= 80 and len(required_missing) <= 1:
        recommendation = "APPLY"
    elif final_score >= 60:
        recommendation = "MAYBE"
    else:
        recommendation = "SKIP"

    return {
        "score": final_score,
        "recommendation": recommendation,
        "skills_score": round(skills_score, 1),
        "experience_score": round(experience_score, 1),
        "experience_status": experience_status,
        "english_score": round(english_score, 1),
        "english_status": english_status,
        "candidate_english_level": candidate_english,
        "required_english_level": required_english,
        "candidate_experience_years": candidate_years,
        "required_experience_years": required_years,
        "score_weights": {
                    "skills": SKILLS_WEIGHT,
                    "experience": EXPERIENCE_WEIGHT,
                    "english": ENGLISH_WEIGHT,
                },
        "required_missing": required_missing,
        "skills": skill_results,
    }
