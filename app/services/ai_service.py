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
                    "If English level is not stated, return null. "

                    "CVs may be written in any language. "
                    "Always normalize skill names and technical capabilities to English. "

                    "Extract both concrete technologies and broader technical competencies "
                    "when those competencies are explicitly supported by the CV. "
                    "Do not limit extracted skills only to framework, library, or tool names. "

                    "For example, if the CV explicitly describes backend development, "
                    "include Backend development as a skill. "
                    "If the CV explicitly describes integrating external APIs or services, "
                    "include API integration as a skill. "
                    "If the CV explicitly describes browser automation, "
                    "include Browser automation as a skill. "
                    "If the CV explicitly describes asynchronous programming, "
                    "include Asynchronous programming as a skill. "

                    "Do not infer unrelated competencies only from a technology name. "
                    "For example, using an API does not imply LLM integration, "
                    "and using SQLite does not imply advanced database engineering. "

                    "For each technical skill, also extract up to 3 technical capabilities "
                    "that describe what the skill enables the candidate to do. "
                    "Capabilities should describe functionality, not simply repeat the skill name. "

                    "Examples: "
                    "Playwright -> browser automation, dynamic website interaction, automated browser control. "
                    "Puppeteer -> browser automation, automated navigation, form automation. "
                    "Selenium -> browser automation, automated browser testing. "
                    "asyncio -> asynchronous programming, non-blocking I/O, concurrency. "
                    "SQLite -> relational databases, SQL data storage. "
                    "Docker -> containerization, application packaging. "
                    "FastAPI -> backend API development, REST APIs, web services. "
                    "HTML -> web markup, page structure. "

                    "Only include capabilities that are reasonably implied by the technology "
                    "or supported by evidence in the CV."
                ),
            },
            {
                "role": "user",
                "content": cv_text,
            },
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
                    "If years of experience or English level are not stated, return null. "

                    "Job descriptions may be written in any language. "
                    "Always normalize skill names, technical requirements, capabilities, "
                    "and alternatives to English. "
                    "Keep the extracted meaning unchanged. "

                    "Only extract requirements that are concrete and materially useful "
                    "for candidate-job matching. "

                    "Do not create scored requirements for generic expectations that are "
                    "normally expected from software engineers or cannot be reliably verified "
                    "from a CV. "

                    "Examples that should NOT become requirements: "
                    "clean and maintainable code, clean code, code quality, good communication, "
                    "problem solving, analytical thinking, attention to detail, teamwork, "
                    "responsibility, proactivity, willingness to learn, motivation, adaptability, "
                    "time management, ownership, and similar generic expectations. "

                    "Focus requirements on concrete and verifiable candidate attributes such as "
                    "programming languages, frameworks, libraries, databases, infrastructure, "
                    "cloud platforms, APIs, integrations, protocols, AI or ML technologies, "
                    "security competencies, domain-specific technical knowledge, deployment skills, "
                    "technical methodologies, explicitly required experience, and language levels. "

                    "For each technical requirement, also extract up to 3 technical capabilities "
                    "that describe the actual functionality or technical ability the employer is asking for. "
                    "Capabilities should describe functionality, not simply repeat the requirement name. "

                    "Examples of capabilities: "
                    "Browser automation -> browser automation, dynamic website interaction, automated browser control. "
                    "Asynchronous Python programming -> asynchronous programming, non-blocking I/O, concurrency. "
                    "Relational databases -> relational databases, SQL, structured data storage. "
                    "REST APIs -> backend API development, HTTP APIs, web services. "
                    "Docker -> containerization, application packaging, deployment consistency. "
                    "RAG -> retrieval-augmented generation, context retrieval, grounded generation. "
                    "Embeddings -> vector representation, semantic similarity, embedding-based retrieval. "

                    "Only include capabilities that are reasonably implied by the job description. "

                    "For each requirement, extract alternatives when the vacancy explicitly lists "
                    "interchangeable technologies, tools, frameworks, providers, or platforms where "
                    "any one of them can satisfy the requirement. "

                    "Examples of alternatives: "
                    "FastAPI / Flask / Django or similar -> FastAPI, Flask, Django. "
                    "LangChain / LlamaIndex or similar -> LangChain, LlamaIndex. "
                    "AWS / GCP / Azure -> AWS, GCP, Azure. "
                    "pgvector, Qdrant, Weaviate or similar -> pgvector, Qdrant, Weaviate. "
                    "OpenAI, Anthropic, Gemini or open-source LLMs -> OpenAI, Anthropic, Gemini, open-source LLMs. "

                    "Do not use alternatives when multiple abilities are jointly required. "
                    "For example, 'API and webhook integrations' means both API integration "
                    "and webhook integration are relevant requirements and should not be represented as alternatives. "

                    "Split compound requirements into separate atomic requirements "
                    "when multiple independently testable technical abilities are jointly required. "

                    "When splitting a compound requirement, preserve the original importance "
                    "(required, preferred, or optional) for each resulting requirement unless "
                    "the job wording clearly assigns different importance levels. "

                    "Examples of compound requirements: "
                    "'API and webhook integrations' -> create separate requirements "
                    "'API integration' and 'Webhook integration'. "

                    "'RAG, embeddings and vector search' -> create separate requirements "
                    "'Retrieval-augmented generation', 'Embeddings', and 'Vector search'. "

                    "'Python and SQL' -> create separate requirements 'Python' and 'SQL' "
                    "if both are independently required. "

                    "Do not split interchangeable alternatives. "
                    "For example, 'FastAPI / Flask / Django' should remain one requirement "
                    "with FastAPI, Flask, and Django in alternatives. "

                    "Do not create duplicate requirements that represent the same technical competency. "
                    "Prefer one normalized requirement instead of several near-duplicate requirements. "

                    "Do not invent a requirement merely because a technology would commonly be used "
                    "for the described work. Only extract requirements that are stated or clearly implied "
                    "by the vacancy itself."
                ),
            },
            {
                "role": "user",
                "content": description,
            },
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
