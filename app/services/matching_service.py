import re
from difflib import SequenceMatcher

from app.services.local_embedding_service import (
    EmbeddingServiceError,
    build_candidate_skill_text,
    build_requirement_text,
    find_best_semantic_match,
    generate_passage_embeddings,
    generate_query_embeddings,
)

SEMANTIC_MIN_SCORE = 0.85
SEMANTIC_MIN_MARGIN = 0.02

PREFERRED_BONUS_POINTS = 2.0
OPTIONAL_BONUS_POINTS = 1.0
MAX_SKILL_BONUS = 10.0

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
    {
        "sql",
        "sqlite",
        "postgresql",
        "mysql",
        "mariadb",
        "relational database",
        "relational databases",
    },
]

GENERIC_SKILL_WORDS = {
    "integration",
    "development",
    "programming",
    "framework",
    "frameworks",
    "service",
    "services",
    "system",
    "systems",
    "application",
    "applications",
}


def _has_meaningful_token_overlap(
    left: str,
    right: str,
) -> bool:
    left_tokens = set(left.split()) - GENERIC_SKILL_WORDS
    right_tokens = set(right.split()) - GENERIC_SKILL_WORDS

    return bool(left_tokens & right_tokens)

def normalize_skill(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[._/+\-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return ALIASES.get(value, value)

def _capability_overlap(
    required_capabilities: list[str],
    candidate_capabilities: list[str],
) -> tuple[int, float]:
    if not required_capabilities or not candidate_capabilities:
        return 0, 0.0

    required = {
        normalize_skill(capability)
        for capability in required_capabilities
    }

    candidate = {
        normalize_skill(capability)
        for capability in candidate_capabilities
    }

    common = required & candidate

    overlap_count = len(common)
    coverage = overlap_count / len(required)

    return overlap_count, coverage

def _compare_capabilities(
    required_capabilities: list[str],
    candidate_capabilities: list[str],
) -> tuple[list[str], list[str]]:
    if not required_capabilities:
        return [], []

    normalized_candidate = {
        normalize_skill(capability)
        for capability in candidate_capabilities
    }

    matched = []
    missing = []

    for capability in required_capabilities:
        normalized = normalize_skill(capability)

        if normalized in normalized_candidate:
            matched.append(capability)
        else:
            missing.append(capability)

    return matched, missing

def _same_related_group(left: str, right: str) -> bool:
    for group in RELATED_SKILL_GROUPS:
        if left in group and right in group:
            return True

    return False


def compare_skill(
    requirement: str,
    candidate_skills: list[str],
    requirement_alternatives: list[str] | None = None,
    requirement_capabilities: list[str] | None = None,
    candidate_skill_capabilities: dict[str, list[str]] | None = None,
    requirement_text: str | None = None,
    candidate_skill_texts: dict[str, str] | None = None,
    query_embeddings: dict[str, object] | None = None,
    passage_embeddings: dict[str, object] | None = None,
):
    required = normalize_skill(requirement)

    normalized_candidates = [
        (skill, normalize_skill(skill))
        for skill in candidate_skills
    ]

    # 1. Exact / alias match
    for original, normalized in normalized_candidates:
        if normalized == required:
            return "match", original, 1.0, "exact"

    if requirement_alternatives:
        normalized_alternatives = {
            normalize_skill(alternative)
            for alternative in requirement_alternatives
        }

        for original, normalized in normalized_candidates:
            if normalized in normalized_alternatives:
                return (
                    "match",
                    original,
                    1.0,
                    "alternative",
                )
    # 2. Known deterministic relationships
    for original, normalized in normalized_candidates:
        if _same_related_group(required, normalized):
            return "partial", original, 0.78, "related"

    # 3. Capability matching
    if (
        requirement_capabilities
        and candidate_skill_capabilities
    ):
        best_capability_skill = None
        best_overlap_count = 0
        best_coverage = 0.0

        for skill in candidate_skills:
            capabilities = candidate_skill_capabilities.get(
                skill,
                [],
            )

            overlap_count, coverage = _capability_overlap(
                requirement_capabilities,
                capabilities,
            )

            if (
                overlap_count > best_overlap_count
                or (
                    overlap_count == best_overlap_count
                    and coverage > best_coverage
                )
            ):
                best_capability_skill = skill
                best_overlap_count = overlap_count
                best_coverage = coverage

        is_capability_match = (
            best_overlap_count >= 2
            or (
                len(requirement_capabilities) == 1
                and best_overlap_count == 1
            )
        )

        if (
            best_capability_skill is not None
            and is_capability_match
        ):
            return (
                "partial",
                best_capability_skill,
                best_coverage,
                "capability",
            )

    # 4. Fuzzy matching
    best_skill = None
    best_ratio = 0.0

    for original, normalized in normalized_candidates:
        ratio = SequenceMatcher(
            None,
            required,
            normalized,
        ).ratio()

        if not _has_meaningful_token_overlap(
            required,
            normalized,
        ):
            continue

        if ratio > best_ratio:
            best_ratio = ratio
            best_skill = original

    if best_ratio >= 0.88:
        return "match", best_skill, best_ratio, "fuzzy"

    if best_ratio >= 0.68:
        return "partial", best_skill, best_ratio, "fuzzy"

    # 5. E5 semantic fallback
    if (
        requirement_text
        and candidate_skill_texts
        and query_embeddings
        and passage_embeddings
    ):
        (
            semantic_skill,
            semantic_score,
            semantic_margin,
        ) = find_best_semantic_match(
            requirement_text,
            candidate_skill_texts,
            query_embeddings,
            passage_embeddings,
        )

        if (
            semantic_skill is not None
            and semantic_score >= SEMANTIC_MIN_SCORE
            and semantic_margin >= SEMANTIC_MIN_MARGIN
        ):
            return (
                "partial",
                semantic_skill,
                semantic_score,
                "embedding",
            )

    return "missing", None, best_ratio, "none"

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
    requirements = job_profile.get("requirements", [])

    candidate_skill_entries = candidate_profile.get(
        "skills",
        []
    )

    candidate_skills = [
        item["name"]
        for item in candidate_skill_entries
        if item.get("name")
    ]


    candidate_skill_texts = {
        item["name"]: build_candidate_skill_text(
            item["name"],
            item.get("evidence"),
            item.get("capabilities", []),
        )
        for item in candidate_skill_entries
    }

    candidate_skill_capabilities = {
        item["name"]: item.get("capabilities", [])
        for item in candidate_skill_entries
    }


    requirement_texts = {
        item["skill"]: build_requirement_text(
            item["skill"],
            item.get("description"),
            item.get("capabilities", []),
        )
        for item in requirements
    }
    requirement_capabilities = {
        item["skill"]: item.get("capabilities", [])
        for item in requirements
    }
    requirement_alternatives = {
        item["skill"]: item.get("alternatives", [])
        for item in requirements
    }

    try:
        query_embeddings = generate_query_embeddings(
            list(requirement_texts.values())
        )

        passage_embeddings = generate_passage_embeddings(
            list(candidate_skill_texts.values())
        )

    except EmbeddingServiceError:
        query_embeddings = {}
        passage_embeddings = {}

    required_total = 0.0
    required_earned = 0.0

    non_required_total = 0.0
    non_required_earned = 0.0

    bonus_points = 0.0

    skill_results = []

    for requirement in requirements:
        skill = requirement.get("skill", "").strip()
        if not skill:
            continue

        importance = requirement.get("importance", "required")
        weight = IMPORTANCE_WEIGHT.get(importance, 1.0)
        status, candidate_skill, confidence, match_method = compare_skill(
            skill,
            candidate_skills,
            requirement_alternatives=requirement_alternatives.get(
                skill,
                [],
            ),
            requirement_capabilities=requirement_capabilities.get(
                skill,
                [],
            ),
            candidate_skill_capabilities=candidate_skill_capabilities,
            requirement_text=requirement_texts.get(skill),
            candidate_skill_texts=candidate_skill_texts,
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
        )

        required_caps = requirement_capabilities.get(
            skill,
            [],
        )

        candidate_caps = (
            candidate_skill_capabilities.get(
                candidate_skill,
                [],
            )
            if candidate_skill
            else []
        )

        if match_method == "exact":
            matched_capabilities = required_caps
            missing_capabilities = []

        elif match_method == "capability":
            matched_capabilities, missing_capabilities = (
                _compare_capabilities(
                    required_caps,
                    candidate_caps,
                )
            )

        else:
            matched_capabilities = []
            missing_capabilities = []

        if status == "match":
            strength = 1.0

        elif status == "partial":
            strength = confidence

        else:
            strength = 0.0

        if importance == "required":
            possible = 1.0
            earned = strength

            required_total += possible
            required_earned += earned

            score_role = "base"

        elif importance == "preferred":
            possible = PREFERRED_BONUS_POINTS
            earned = possible * strength

            bonus_points += earned

            non_required_total += IMPORTANCE_WEIGHT.get(
                importance,
                0.5,
            )
            non_required_earned += (
                IMPORTANCE_WEIGHT.get(
                    importance,
                    0.5,
                )
                * strength
            )

            score_role = "bonus"

        else:
            possible = OPTIONAL_BONUS_POINTS
            earned = possible * strength

            bonus_points += earned

            non_required_total += IMPORTANCE_WEIGHT.get(
                importance,
                0.2,
            )
            non_required_earned += (
                IMPORTANCE_WEIGHT.get(
                    importance,
                    0.2,
                )
                * strength
            )

            score_role = "bonus"

        skill_results.append(
            {
                "requirement": skill,
                "importance": importance,
                "candidate_skill": candidate_skill,
                "status": status,
                "similarity": round(confidence, 2),
                "match_method": match_method,
                "matched_capabilities": matched_capabilities,
                "missing_capabilities": missing_capabilities,
                "score_role": score_role,
                "earned": round(earned, 3),
                "possible": possible,
            }
        )

    if required_total > 0:
        required_skills_score = (
            required_earned
            / required_total
            * 100
        )

        skill_bonus = min(
            bonus_points,
            MAX_SKILL_BONUS,
        )

        skills_score = min(
            required_skills_score + skill_bonus,
            100.0,
        )

    else:
        # Fallback for unusual vacancies that contain
        # only preferred/optional technical requirements.
        required_skills_score = 0.0
        skill_bonus = 0.0

        skills_score = (
            non_required_earned
            / non_required_total
            * 100
            if non_required_total
            else 0.0
        )

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

    active_scores = [
        (skills_score, SKILLS_WEIGHT),
    ]

    if experience_status != "not_required":
        active_scores.append(
            (experience_score, EXPERIENCE_WEIGHT)
        )

    if english_status != "not_required":
        active_scores.append(
            (english_score, ENGLISH_WEIGHT)
        )

    total_weight = sum(
        weight
        for _, weight in active_scores
    )

    effective_weights = {
        "skills": SKILLS_WEIGHT / total_weight,
        "experience": (
            EXPERIENCE_WEIGHT / total_weight
            if experience_status != "not_required"
            else 0.0
        ),
        "english": (
            ENGLISH_WEIGHT / total_weight
            if english_status != "not_required"
            else 0.0
        ),
    }

    final_score = sum(
        score * weight
        for score, weight in active_scores
    ) / total_weight

    final_score = round(final_score, 1)

    required_missing = [
        item["requirement"]
        for item in skill_results
        if item["importance"] == "required" and item["status"] == "missing"
    ]

    required_partial = [
        item["requirement"]
        for item in skill_results
        if item["importance"] == "required"
        and item["status"] == "partial"
    ]

    unknown_fields = []

    if experience_status == "unknown":
        unknown_fields.append("experience_years")

    if english_status == "unknown":
        unknown_fields.append("english_level")

    if (
        final_score >= 80
        and len(required_missing) <= 1
        and experience_status != "below"
        and english_status != "below"
    ):
        recommendation = "APPLY"

    elif (
        final_score >= 60
        and len(required_missing) <= 2
    ):
        recommendation = "MAYBE"

    else:
        recommendation = "SKIP"

    recommendation_reasons = []

    if required_missing:
        recommendation_reasons.append(
            f"{len(required_missing)} required skill(s) missing"
        )

    if required_partial:
        recommendation_reasons.append(
            f"{len(required_partial)} required skill(s) partially matched"
        )

    if experience_status == "unknown":
        recommendation_reasons.append(
            "experience duration is unknown"
        )

    elif experience_status == "below":
        recommendation_reasons.append(
            "experience is below the job requirement"
        )

    if english_status == "unknown":
        recommendation_reasons.append(
            "English level is unknown"
        )

    elif english_status == "below":
        recommendation_reasons.append(
            "English level is below the job requirement"
        )

    return {
        "score": final_score,
        "recommendation": recommendation,
        "recommendation_reasons": recommendation_reasons,
        "skills_score": round(skills_score, 1),
        "required_skills_score": round(
            required_skills_score,
            1,
        ),
        "skill_bonus": round(
            skill_bonus,
            1,
        ),
        "experience_score": round(experience_score, 1),
        "experience_status": experience_status,
        "english_score": round(english_score, 1),
        "english_status": english_status,
        "candidate_english_level": candidate_english,
        "required_english_level": required_english,
        "candidate_experience_years": candidate_years,
        "required_experience_years": required_years,
        "score_weights": {
            "base": {
                "skills": SKILLS_WEIGHT,
                "experience": EXPERIENCE_WEIGHT,
                "english": ENGLISH_WEIGHT,
            },
            "effective": {
                "skills": round(effective_weights["skills"], 3),
                "experience": round(effective_weights["experience"], 3),
                "english": round(effective_weights["english"], 3),
            },
        },
        "required_missing": required_missing,
        "required_partial": required_partial,
        "unknown_fields": unknown_fields,
        "skills": skill_results,
    }
