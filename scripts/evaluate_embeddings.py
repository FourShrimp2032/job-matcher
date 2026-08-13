from app.services.local_embedding_service import (
    generate_query_embedding,
    generate_passage_embeddings,
    cosine_similarity,
)


TEST_CASES = [
    {
        "requirement": (
            "Asynchronous Python programming. "
            "Experience with async await and non-blocking Python code."
        ),
        "skills": [
            "Python. Used for backend development and automation.",
            "asyncio. Implemented asynchronous data collection.",
            "Docker. Used for containerization.",
            "Git. Used for version control.",
            "SQLite. Used as a relational database.",
        ],
        "expected": "asyncio",
    },
    {
        "requirement": (
            "Browser automation and automated interaction "
            "with dynamic websites."
        ),
        "skills": [
            "Python. Backend programming language.",
            "Playwright. Automated dynamically loaded web pages.",
            "PostgreSQL. Relational database.",
            "Docker. Containerization.",
        ],
        "expected": "Playwright",
    },
    {
        "requirement": (
            "Experience working with relational SQL databases."
        ),
        "skills": [
            "FastAPI. Python web framework.",
            "SQLite. Used as a relational SQL database.",
            "Playwright. Browser automation.",
            "Git. Version control.",
        ],
        "expected": "SQLite",
    },
]


def evaluate_case(case):
    query = case["requirement"]
    skills = case["skills"]

    query_embedding = generate_query_embedding(query)
    passage_embeddings = generate_passage_embeddings(skills)

    results = []

    for skill in skills:
        score = cosine_similarity(
            query_embedding,
            passage_embeddings[skill],
        )

        results.append((skill, score))

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

    expected = case["expected"]

    success = best_skill.lower().startswith(
        expected.lower()
    )

    print("=" * 70)
    print(f"Requirement: {query}")
    print(f"Expected:    {expected}")
    print()

    for skill, score in results:
        print(f"{score:.4f}  {skill}")

    print()
    print(f"Best score: {best_score:.4f}")
    print(f"Margin:     {margin:.4f}")
    print(f"Result:     {'PASS' if success else 'FAIL'}")


def main():
    for case in TEST_CASES:
        evaluate_case(case)


if __name__ == "__main__":
    main()