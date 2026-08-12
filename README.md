# JobMatch AI

An AI-powered backend that extracts structured information from a candidate CV and a job vacancy, then calculates a deterministic match score.

## MVP features

- Parse CV text into a structured candidate profile with an LLM
- Parse job descriptions into structured requirements
- Store candidates, jobs and match results in SQLite
- Deterministic Python scoring engine
- `MATCH / PARTIAL / MISSING` skill comparison
- Requirement importance: `required / preferred / optional`
- `APPLY / MAYBE / SKIP` recommendation
- AI explanation of strengths, gaps and interview focus without letting the LLM choose the score
- Interactive Swagger docs through FastAPI

## Architecture

```text
CV text -> AI parser -> Candidate profile --\
                                        Matching engine -> Score + recommendation
Job text -> AI parser -> Job profile -------/
```

The LLM extracts structured data. It does **not** choose the final percentage. The score is calculated by application code so it is inspectable and reproducible.

## Run locally

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Using `python -m pip` is recommended, especially on Windows systems where calling `pip.exe` directly may be blocked.

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and add your OpenAI API key.

Windows:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 4. Start FastAPI

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API flow

### 1. Create candidate

`POST /candidates`

```json
{
  "name": "Test Candidate",
  "email": "candidate@example.com",
  "cv_text": "Python backend developer with experience using FastAPI, SQL, PostgreSQL and web scraping..."
}
```

### 2. Create job

`POST /jobs`

```json
{
  "company": "Example Company",
  "title": "Junior Python Developer",
  "description": "We are looking for a Python developer. Required: Python, FastAPI, PostgreSQL, Git and Docker. AWS is nice to have. 1 year of experience is preferred."
}
```

### 3. Match them

`POST /matches`

```json
{
  "candidate_id": 1,
  "job_id": 1
}
```

Example result shape:

```json
{
  "score": 78.4,
  "recommendation": "MAYBE",
  "details": {
    "skills_score": 75.0,
    "experience_score": 100.0,
    "required_missing": ["Docker"],
    "skills": []
  }
}
```

## Next versions

- PDF/DOCX CV upload
- PostgreSQL
- semantic matching with embeddings
- interview question generator
- learning plan from missing skills
- vacancy URL scraping
- application statuses and dashboard
- frontend
