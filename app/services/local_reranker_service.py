from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RerankerServiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    try:
        return CrossEncoder(MODEL_NAME)

    except Exception as exc:
        raise RerankerServiceError(
            f"Could not load reranker model: {exc}"
        ) from exc


def rerank_candidates(
    requirement: str,
    candidates: list[tuple[str, str]],
) -> list[tuple[str, float]]:
    if not candidates:
        return []

    model = get_reranker_model()

    pairs = [
        (requirement, candidate_text)
        for _, candidate_text in candidates
    ]

    try:
        scores = model.predict(pairs)

    except Exception as exc:
        raise RerankerServiceError(
            f"Could not rerank candidates: {exc}"
        ) from exc

    results = [
        (
            candidate_name,
            float(score),
        )
        for (
            candidate_name,
            _
        ), score in zip(candidates, scores)
    ]

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results