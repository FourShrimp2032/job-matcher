from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/e5-small-v2"


class EmbeddingServiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME)

    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not load embedding model: {exc}"
        ) from exc


def build_requirement_text(
    skill: str,
    description: str | None = None,
    capabilities: list[str] | None = None,
) -> str:
    text = f"Technical job requirement: {skill}."

    if capabilities:
        text += (
            " Required capabilities: "
            + ", ".join(capabilities)
            + "."
        )

    if description:
        text += f" Description: {description}"

    return text


def build_candidate_skill_text(
    name: str,
    evidence: str | None = None,
    capabilities: list[str] | None = None,
) -> str:
    text = f"Candidate technical skill: {name}."

    if capabilities:
        text += (
            " Technical capabilities: "
            + ", ".join(capabilities)
            + "."
        )

    if evidence:
        text += f" Evidence: {evidence}"

    return text


def generate_query_embeddings(
    texts: list[str],
) -> dict[str, object]:
    unique_texts = list(dict.fromkeys(texts))

    if not unique_texts:
        return {}

    model = get_embedding_model()

    inputs = [
        f"query: {text}"
        for text in unique_texts
    ]

    try:
        embeddings = model.encode(
            inputs,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not generate query embeddings: {exc}"
        ) from exc

    return {
        text: embedding
        for text, embedding in zip(
            unique_texts,
            embeddings,
        )
    }


def generate_query_embedding(
    text: str,
):
    embeddings = generate_query_embeddings([text])

    return embeddings[text]


def generate_passage_embeddings(
    texts: list[str],
) -> dict[str, object]:
    unique_texts = list(dict.fromkeys(texts))

    if not unique_texts:
        return {}

    model = get_embedding_model()

    inputs = [
        f"passage: {text}"
        for text in unique_texts
    ]

    try:
        embeddings = model.encode(
            inputs,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not generate passage embeddings: {exc}"
        ) from exc

    return {
        text: embedding
        for text, embedding in zip(
            unique_texts,
            embeddings,
        )
    }


def cosine_similarity(
    left_embedding,
    right_embedding,
) -> float:
    similarity = left_embedding @ right_embedding

    return float(similarity.item())


def find_best_semantic_match(
    requirement_text: str,
    candidate_skill_texts: dict[str, str],
    query_embeddings: dict[str, object],
    passage_embeddings: dict[str, object],
) -> tuple[str | None, float, float]:

    requirement_embedding = query_embeddings.get(
        requirement_text
    )

    if requirement_embedding is None:
        return None, 0.0, 0.0

    results = []

    for skill_name, skill_text in candidate_skill_texts.items():

        skill_embedding = passage_embeddings.get(
            skill_text
        )

        if skill_embedding is None:
            continue

        similarity = cosine_similarity(
            requirement_embedding,
            skill_embedding,
        )

        results.append(
            (skill_name, similarity)
        )

    if not results:
        return None, 0.0, 0.0

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    best_skill, best_score = results[0]

    second_score = (
        results[1][1]
        if len(results) > 1
        else 0.0
    )

    margin = best_score - second_score

    return best_skill, best_score, margin

def retrieve_semantic_candidates(
    requirement_text: str,
    candidate_skill_texts: dict[str, str],
    query_embeddings: dict[str, object],
    passage_embeddings: dict[str, object],
    top_k: int = 8,
) -> list[tuple[str, float]]:

    requirement_embedding = query_embeddings.get(
        requirement_text
    )

    if requirement_embedding is None:
        return []

    results = []

    for skill_name, skill_text in candidate_skill_texts.items():
        skill_embedding = passage_embeddings.get(
            skill_text
        )

        if skill_embedding is None:
            continue

        similarity = cosine_similarity(
            requirement_embedding,
            skill_embedding,
        )

        results.append(
            (skill_name, similarity)
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results[:top_k]