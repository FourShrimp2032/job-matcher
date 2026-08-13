# JobMatch AI

JobMatch AI is a backend service that compares a candidate's CV with a job description and estimates how well they match.

I built this project because simple keyword matching is not enough for real job descriptions. The same technical skill can be described in many different ways, some requirements are mandatory while others are only nice-to-have, and a candidate may have relevant experience without using exactly the same wording as the vacancy.

The goal of JobMatch AI is to make this comparison more structured and explainable.

## What it does

A candidate can upload a PDF CV or provide CV text.

A job description can be provided in English, Ukrainian, or another language.

The application then:

1. Extracts structured information from the CV.
2. Extracts technical requirements from the job description.
3. Normalizes technologies and competencies.
4. Matches candidate skills against job requirements.
5. Calculates a match score.
6. Returns an APPLY / MAYBE / SKIP recommendation.
7. Generates an explanation of the candidate's strengths, gaps, and possible interview focus areas.

## Example

A vacancy may contain:

```text
Experience with FastAPI / Flask / Django or similar frameworks
```

Instead of treating all three frameworks as separate mandatory requirements, JobMatch AI understands them as alternatives:

```text
Python web frameworks
Alternatives: FastAPI, Flask, Django
```

So a candidate with FastAPI experience satisfies the requirement.

The same applies to requirements such as:

```text
AWS / GCP / Azure
LangChain / LlamaIndex
pgvector / Qdrant / Weaviate
```

On the other hand, requirements such as:

```text
API and webhook integrations
```

are split into separate requirements because both abilities are expected:

```text
API integration
Webhook integration
```

## Matching approach

The matcher does not rely on a single similarity score.

It uses several matching strategies:

- exact skill matching
- normalized aliases
- explicit technology alternatives
- known related technologies
- capability-based matching
- fuzzy matching with safeguards against false positives
- semantic similarity using local embeddings

For semantic matching, the project uses the `intfloat/e5-small-v2` model locally through Sentence Transformers.

This helps identify related technical concepts while keeping deterministic rules for cases where semantic similarity alone would be unreliable.

## Scoring

Required and preferred requirements are treated differently.

**Required skills** form the base skill score.

Missing a required technology lowers the match score.

**Preferred skills** do not lower the base score when they are missing. Instead, matching preferred technologies adds bonus points.

For example:

```text
Required:
Python                ✅
FastAPI               ✅
SQL                   🟡
RAG                   ❌

Preferred:
Docker                ✅ + bonus
AWS                   ❌ no penalty
```

Experience and English requirements are evaluated separately and included in the final score only when they are actually specified in the vacancy.

The final result includes:

```json
{
  "score": 82.4,
  "recommendation": "APPLY",
  "skills_score": 86.0,
  "experience_score": 75.0,
  "required_missing": [],
  "required_partial": ["SQL"]
}
```

## AI parsing

OpenAI structured outputs are used to convert unstructured CVs and job descriptions into predictable Pydantic models.

For candidates, the parser extracts information such as:

- technical skills
- skill evidence
- technical capabilities
- experience
- English level
- education

For jobs, it extracts:

- required skills
- preferred skills
- alternatives
- technical capabilities
- experience requirements
- English requirements

Generic phrases such as:

```text
good communication
problem solving
willingness to learn
clean and maintainable code
```

are intentionally excluded from technical scoring because they cannot be reliably evaluated from a CV.

## Explainability

I wanted the score to be understandable instead of returning only a percentage.

Each requirement includes information such as:

```json
{
  "requirement": "Python web frameworks",
  "candidate_skill": "FastAPI",
  "status": "match",
  "similarity": 1.0,
  "match_method": "alternative"
}
```

The API also generates a higher-level explanation containing:

- strengths
- missing requirements
- experience gaps
- interview focus areas

This makes it possible to understand why a candidate received a particular score.

## Tech stack

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite
- OpenAI API
- Sentence Transformers
- E5 embeddings
- pypdf
- Uvicorn

## Project structure

```text
app/
├── main.py
├── database.py
├── models.py
├── schemas.py
├── settings.py
│
├── routers/
│   ├── candidates.py
│   ├── jobs.py
│   └── matches.py
│
└── services/
    ├── ai_service.py
    ├── matching_service.py
    ├── pdf_service.py
    └── local_embedding_service.py

scripts/
└── evaluate_embeddings.py
```

## Running locally

Clone the repository:

```bash
git clone https://github.com/FourShrimp2032/job-matcher.git
cd job-matcher
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

FastAPI documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

## Why I built it

I built JobMatch AI as a practical project for experimenting with backend development, LLM structured outputs, semantic search, and hybrid matching systems.

One of the most interesting parts of the project was combining deterministic rules with AI and embeddings.

LLMs are useful for understanding unstructured CVs and job descriptions, while deterministic scoring makes the final result more predictable and explainable.

The result is a system where AI helps understand the data, but does not completely control the final score.

## Possible next steps

Some ideas I may explore later:

- ranking multiple vacancies for one candidate
- PostgreSQL support
- storing and comparing multiple CV versions
- improved semantic matching
- evaluation datasets for measuring matcher accuracy
- frontend interface
