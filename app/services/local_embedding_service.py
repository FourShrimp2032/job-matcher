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


def generate_query_embedding(
    text: str,
):
    model = get_embedding_model()

    try:
        embedding = model.encode(
            f"query: {text}",
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not generate query embedding: {exc}"
        ) from exc

    return embedding

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

def build_requirement_text(
    skill: str,
    description: str | None = None,
) -> str:
    text = f"Technical job requirement: {skill}."

    if description:
        text += f" Description: {description}"

    return text

def build_candidate_skill_text(
    name: str,
    evidence: str | None = None,
) -> str:
    text = f"Candidate technical skill: {name}."

    if evidence:
        text += f" Evidence: {evidence}"

    return text

def find_best_semantic_match(
    requirement: str,
    candidate_skills: list[str],
) -> tuple[str | None, float]:

    if not candidate_skills:
        return None, 0.0

    requirement_embedding = generate_query_embedding(
        requirement
    )

    candidate_embeddings = (
        generate_passage_embeddings(
            candidate_skills
        )
    )

    best_skill = None
    best_similarity = -1.0

    for skill in candidate_skills:
        skill_embedding = candidate_embeddings[skill]

        similarity = cosine_similarity(
            requirement_embedding,
            skill_embedding,
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_skill = skill

    if best_skill is None:
        return None, 0.0

    return best_skill, best_similarity