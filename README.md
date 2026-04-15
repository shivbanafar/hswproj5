# Financial Sentiment Analyzer

A REST API that analyzes financial news headlines and predicts whether they are **positive**, **negative**, or **neutral** for stocks/markets. Built with VADER (NLP), FastAPI (REST API), Docker (containerization), and GitHub Actions (CI/CD).

---

## Table of Contents

1. [Where DevOps, CI/CD, REST, and AI are Used](#where-devops-cicd-rest-and-ai-are-used)
2. [Project Architecture](#project-architecture)
3. [Tech Stack Explained](#tech-stack-explained)
4. [How the Sentiment Model Works (VADER)](#how-the-sentiment-model-works-vader)
5. [The Dataset — Financial PhraseBank](#the-dataset--financial-phrasebank)
6. [REST API — Endpoints and Design](#rest-api--endpoints-and-design)
7. [How FastAPI Works in This Project](#how-fastapi-works-in-this-project)
8. [CI/CD Pipeline — GitHub Actions](#cicd-pipeline--github-actions)
9. [Docker and Containerization](#docker-and-containerization)
10. [Testing with Pytest](#testing-with-pytest)
11. [Streamlit Dashboard](#streamlit-dashboard)
12. [How to Run the Project](#how-to-run-the-project)
13. [Viva Q&A — Expected Questions and Answers](#viva-qa--expected-questions-and-answers)

---

## Where DevOps, CI/CD, REST, and AI are Used

This project is specifically designed to cover the intersection of **AI/ML** and **DevOps**. Here's exactly where each concept shows up:

### AI / Machine Learning / NLP

| Concept | Where it's used | File |
|---------|----------------|------|
| Sentiment Analysis (NLP) | Core feature — classifies text as positive/negative/neutral | `app/model.py` |
| VADER Model | Rule-based NLP model with a sentiment lexicon | `app/model.py` → `SentimentIntensityAnalyzer` |
| Labeled Dataset | Financial PhraseBank — 2264 sentences with ground-truth labels | `FinancialPhraseBank/Sentences_AllAgree.txt` |
| Model Benchmarking | Measure accuracy of VADER against the labeled dataset | `scripts/benchmark.py` |
| Confidence Scores | Per-class probabilities (positive/negative/neutral) returned with every prediction | API response JSON |

### REST API

| Concept | Where it's used | File |
|---------|----------------|------|
| REST Architecture | The entire API follows REST conventions (resources, HTTP methods, status codes) | `app/routers/sentiment.py` |
| POST `/api/v1/analyze` | Accepts a JSON body, returns sentiment — standard REST resource creation pattern | `app/routers/sentiment.py` |
| GET `/api/v1/analyze?headline=...` | Query-parameter based retrieval — idempotent GET request | `app/routers/sentiment.py` |
| POST `/api/v1/analyze/batch` | Batch processing endpoint — multiple resources in one request | `app/routers/sentiment.py` |
| GET `/api/v1/samples` | Read-only data retrieval with query filters (`count`, `sentiment`) | `app/routers/sentiment.py` |
| GET `/health` | Health check endpoint — standard pattern for monitoring | `app/main.py` |
| JSON Request/Response | All endpoints accept and return JSON | Pydantic schemas in `app/schemas.py` |
| HTTP Status Codes | 200 (success), 422 (validation error) — proper REST status code usage | All endpoints |
| API Versioning | `/api/v1/` prefix — allows future breaking changes under `/api/v2/` | `app/main.py` → `prefix="/api/v1"` |
| Swagger/OpenAPI Docs | Auto-generated interactive API documentation at `/docs` | FastAPI auto-generates this |

### DevOps

| Concept | Where it's used | File |
|---------|----------------|------|
| Containerization (Docker) | App packaged as a Docker image — runs identically anywhere | `Dockerfile` |
| Multi-container orchestration | API + Dashboard run together via docker-compose | `docker-compose.yml` |
| Health checks | Docker monitors if the API container is healthy before starting the dashboard | `docker-compose.yml` → `healthcheck` |
| Service discovery | Dashboard finds the API by service name (`http://api:8000`) not hardcoded IP | `docker-compose.yml` → `API_BASE` env var |
| Environment variables | `API_BASE` configurable per environment (local vs Docker vs production) | `dashboard/streamlit_app.py` |
| Dependency isolation | Virtual environment (`.venv`) isolates Python packages | `requirements.txt` |

### CI/CD (Continuous Integration / Continuous Deployment)

| Concept | Where it's used | File |
|---------|----------------|------|
| Automated pipeline | Triggered on every `git push` or pull request — no manual steps | `.github/workflows/ci.yml` |
| Linting (code quality gate) | `ruff check .` runs automatically — blocks merge if code has errors | CI Job 1, Step "Lint with ruff" |
| Automated testing | `pytest tests/ -v` runs all 12 tests automatically | CI Job 1, Step "Run tests" |
| Docker build in CI | Builds the Docker image in the pipeline to catch build failures early | CI Job 2, Step "Build Docker image" |
| Container health check in CI | Starts the container and verifies `/health` responds before proceeding | CI Job 2, Step "Start container and health check" |
| Integration test in CI | Sends a real HTTP request to the running container and validates the JSON response | CI Job 2, Step "Test API endpoint" |
| Job dependency | Docker build job only runs if lint+test job passes first (`needs: lint-and-test`) | `.github/workflows/ci.yml` → `needs` |
| Dependency caching | `cache: "pip"` in CI caches pip downloads so subsequent runs are faster | `.github/workflows/ci.yml` |

---

## Project Architecture

```
HSWProj5/
├── .github/
│   └── workflows/
│       └── ci.yml              ← CI/CD pipeline (GitHub Actions)
├── app/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI app entry point + lifespan
│   ├── model.py                ← VADER analyzer + dataset loader
│   ├── schemas.py              ← Pydantic request/response models
│   └── routers/
│       ├── __init__.py
│       └── sentiment.py        ← All API endpoint handlers
├── dashboard/
│   └── streamlit_app.py        ← Web UI (Streamlit)
├── scripts/
│   └── benchmark.py            ← Model accuracy evaluation
├── tests/
│   ├── __init__.py
│   └── test_sentiment.py       ← 12 automated tests (Pytest)
├── FinancialPhraseBank/
│   ├── Sentences_AllAgree.txt   ← Dataset (2264 labeled sentences)
│   ├── Sentences_75Agree.txt
│   ├── Sentences_66Agree.txt
│   └── Sentences_50Agree.txt
├── Dockerfile                  ← Container definition
├── docker-compose.yml          ← Multi-container orchestration
├── pyproject.toml              ← Ruff linter configuration
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

### How data flows through the system

```
User types headline
        ↓
   [Streamlit Dashboard]  ──HTTP POST──→  [FastAPI Server]
        or                                      ↓
   [curl / Swagger UI]                    [VADER Model]
                                               ↓
                                    { sentiment, confidence,
                                      scores }
                                               ↓
                              ←──JSON Response──┘
```

---

## Tech Stack Explained

| Technology | What it is | Why we used it |
|-----------|-----------|---------------|
| **VADER** | Rule-based NLP sentiment model | Lightweight, no GPU needed, works offline, established baseline |
| **FastAPI** | Python web framework for building APIs | Fastest Python API framework, auto-generates Swagger docs, type-safe with Pydantic |
| **Uvicorn** | ASGI server that runs FastAPI | Production-grade async server |
| **Pydantic** | Data validation library | Validates all API inputs/outputs, prevents bad data from reaching the model |
| **Pytest** | Python testing framework | Industry standard, easy to write, integrates with CI |
| **Streamlit** | Python library that turns scripts into web apps | Zero frontend code needed, great for ML dashboards |
| **Docker** | Containerization platform | Package the app so it runs identically on any machine |
| **Docker Compose** | Multi-container orchestration | Run API + Dashboard together with one command |
| **GitHub Actions** | CI/CD platform | Automates testing, linting, and Docker builds on every push |
| **Ruff** | Python linter | Catches code errors and enforces import ordering |

---

## How the Sentiment Model Works (VADER)

### What is VADER?

**VADER** = **V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner

It is a **rule-based** sentiment analysis model. Unlike neural networks (BERT, GPT), VADER does not learn from data. Instead, it uses:

1. **A lexicon** — a dictionary of ~7,500 words, each with a human-assigned sentiment score
   - "profit" → positive (+2.1)
   - "bankruptcy" → negative (-3.4)
   - "announced" → neutral (0.0)

2. **Rules** that handle:
   - **Capitalization**: "GREAT" is more intense than "great"
   - **Punctuation**: "Great!!!" is more intense than "Great"
   - **Negation**: "not good" flips the sentiment
   - **Degree modifiers**: "very good" > "good", "slightly bad" < "bad"

### How VADER scores text

For any input text, VADER returns 4 scores:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores("Company reports record profits")
# {'neg': 0.0, 'neu': 0.556, 'pos': 0.444, 'compound': 0.6249}
```

| Score | Meaning |
|-------|---------|
| `pos` | Proportion of text that is positive (0.0 to 1.0) |
| `neg` | Proportion of text that is negative (0.0 to 1.0) |
| `neu` | Proportion of text that is neutral (0.0 to 1.0) |
| `compound` | Normalized overall score (-1.0 to +1.0) |

`pos + neg + neu = 1.0` (they sum to 1)

### Decision rule (how we classify)

```
compound >=  0.05  →  POSITIVE
compound <= -0.05  →  NEGATIVE
otherwise          →  NEUTRAL
```

These thresholds are from the VADER paper (Hutto & Gilbert, 2014).

### Where this lives in the code

```python
# app/model.py — the core logic

class VADERAnalyzer:
    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()  # loads the lexicon

    def analyze(self, headline: str) -> SentimentResult:
        scores = self._analyzer.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return SentimentResult(
            headline=headline,
            sentiment=sentiment,
            confidence=round(class_scores[sentiment], 4),
            scores={...},  # pos, neg, neu
        )
```

### Limitation of VADER (important for viva)

VADER is **general-purpose**, not financial-specific. It doesn't understand financial jargon:

| Headline | VADER says | Correct answer |
|----------|-----------|---------------|
| "Apple beats Q3 earnings expectations" | neutral | positive |
| "Revenue surges 30% year-over-year" | neutral | positive |
| "Company downgrades guidance" | neutral | negative |

A financial-specific model like **FinBERT** (which we originally planned but couldn't use due to disk constraints) handles these correctly because it was trained on financial text.

---

## The Dataset — Financial PhraseBank

### What is it?

The **Financial PhraseBank** is a labeled dataset created by researchers (Malo et al., 2014). It contains sentences from financial news articles, each labeled by 16 human annotators as positive, negative, or neutral.

### The files

| File | Annotator Agreement | Sentences |
|------|-------------------|-----------|
| `Sentences_AllAgree.txt` | 100% (all 16 agreed) | 2,264 |
| `Sentences_75Agree.txt` | 75%+ agreed | 3,453 |
| `Sentences_66Agree.txt` | 66%+ agreed | 4,217 |
| `Sentences_50Agree.txt` | 50%+ agreed | 4,846 |

**We use `Sentences_AllAgree.txt`** — highest quality labels since every annotator agreed on the sentiment.

### File format

```
According to Gran , the company has no plans to move...@neutral
For the last quarter of 2010 , net sales doubled...@positive
The company reported a net loss of EUR 5.7 mn...@negative
```

Each line: `sentence@label` — separated by `@`.

### How it's used in this project

1. **Samples endpoint** (`GET /api/v1/samples`) — returns labeled examples from the dataset for the dashboard
2. **Benchmark script** (`scripts/benchmark.py`) — measures VADER's accuracy against the ground-truth labels
3. **Dashboard** — pre-loaded examples users can analyze

### Where the loading happens

```python
# app/model.py — loaded once at startup

def _load_phrasebank(path: Path) -> list[SampleHeadline]:
    samples = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            sentence, label = line.rsplit("@", 1)  # split on last @
            samples.append(SampleHeadline(sentence=sentence, label=label))
    return samples
```

---

## REST API — Endpoints and Design

### What is a REST API?

**REST** = **RE**presentational **S**tate **T**ransfer

A set of rules for building web APIs:
- Resources are identified by URLs (`/api/v1/analyze`)
- Actions are defined by HTTP methods (`GET`, `POST`, `PUT`, `DELETE`)
- Data is exchanged as JSON
- Each request is stateless (server doesn't remember previous requests)

### Our endpoints

#### 1. Health Check

```
GET /health

Response: {"status": "ok", "model": "VADER (vaderSentiment)"}
```

**Why**: Standard DevOps pattern. Docker, Kubernetes, and monitoring tools (like Prometheus) call this to check if the service is alive. Our `docker-compose.yml` uses it to wait for the API before starting the dashboard.

#### 2. Analyze Single Headline (POST)

```
POST /api/v1/analyze
Content-Type: application/json

{"headline": "Company reports record quarterly profits"}

Response:
{
    "headline": "Company reports record quarterly profits",
    "sentiment": "positive",
    "confidence": 0.4404,
    "scores": {
        "positive": 0.4404,
        "negative": 0.0,
        "neutral": 0.5596
    }
}
```

**Why POST**: We're sending data (the headline) in the request body. REST convention is POST for this.

#### 3. Analyze Single Headline (GET)

```
GET /api/v1/analyze?headline=Company+reports+record+profits

Response: (same as above)
```

**Why GET too**: Convenience. You can test this directly in a browser URL bar or with a simple `curl`. GET requests are also cacheable.

#### 4. Batch Analysis

```
POST /api/v1/analyze/batch

{"headlines": ["Revenue up 30%", "CEO arrested", "Board meets"]}

Response:
{
    "results": [
        {"headline": "Revenue up 30%", "sentiment": "positive", ...},
        {"headline": "CEO arrested", "sentiment": "negative", ...},
        {"headline": "Board meets", "sentiment": "neutral", ...}
    ],
    "model_used": "VADER (vaderSentiment)"
}
```

**Why**: Real-world use case — if you have 50 headlines from a news feed, sending 50 individual requests is slow. Batch endpoints are a standard API optimization.

**Validation**: Maximum 50 headlines per request. Returns HTTP 422 if you exceed this.

#### 5. Sample Headlines

```
GET /api/v1/samples?count=5&sentiment=negative

Response:
{
    "samples": [
        {"sentence": "The company reported a net loss of...", "label": "negative"},
        ...
    ],
    "dataset": "takala/financial_phrasebank (sentences_allagree)",
    "total_available": 2264
}
```

**Why**: Serves labeled examples from the dataset. Powers the dashboard so users have real headlines to test with.

### API Versioning

All endpoints are under `/api/v1/`. If we ever make breaking changes, we'd create `/api/v2/` while keeping `/api/v1/` working. This is a standard REST best practice.

---

## How FastAPI Works in This Project

### Request lifecycle

```
HTTP request arrives
       ↓
  [Uvicorn server] receives it
       ↓
  [FastAPI router] matches URL to the correct function
       ↓
  [Pydantic] validates the request body against the schema
       ↓  (returns 422 if invalid)
  [Endpoint function] runs → calls model.analyzer.analyze()
       ↓
  [Pydantic] serializes the response to JSON
       ↓
  HTTP response sent back
```

### Pydantic schemas — input validation

```python
# app/schemas.py

class HeadlineRequest(BaseModel):
    headline: str   # Must be a string, must be present

class SentimentResult(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]  # Only these 3 values allowed
    confidence: float
    scores: dict[str, float]
```

If someone sends `{"headline": 123}` or `{}`, Pydantic automatically returns a 422 error with a description of what's wrong. We don't write any manual validation for this.

### Lifespan — model loading at startup

```python
# app/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server starts
    model_module.analyzer = model_module.VADERAnalyzer()  # loads model + dataset
    yield
    # Runs ONCE when the server stops
    model_module.analyzer = None  # cleanup
```

**Why not load per-request?** Loading the dataset (2264 sentences) on every request would be wasteful. The lifespan pattern loads it once and keeps it in memory.

### Swagger UI — auto-generated documentation

FastAPI automatically generates an interactive API documentation page at `http://localhost:8000/docs`. You can:
- See all endpoints and their expected inputs/outputs
- Send test requests directly from the browser
- View the OpenAPI schema

No code is written for this — FastAPI generates it from the Pydantic schemas and function signatures.

---

## CI/CD Pipeline — GitHub Actions

### What is CI/CD?

- **CI (Continuous Integration)**: Automatically test and lint code every time it's pushed
- **CD (Continuous Deployment)**: Automatically deploy code after CI passes

Our pipeline handles CI (test + lint + Docker build). CD would be deploying to AWS/GCP/Azure — not implemented here but the Docker image is ready for it.

### The pipeline (`.github/workflows/ci.yml`)

```
git push to main/master or open a PR
              ↓
    ┌─────────────────────┐
    │  Job 1: Lint & Test │
    │  1. Checkout code   │
    │  2. Setup Python    │
    │  3. Install deps    │
    │  4. ruff check .    │  ← Code quality gate
    │  5. pytest tests/   │  ← All 12 tests must pass
    └─────────┬───────────┘
              ↓ (only if Job 1 passes)
    ┌─────────────────────────┐
    │  Job 2: Docker Build    │
    │  1. Checkout code       │
    │  2. docker build        │  ← Build the container image
    │  3. docker run + health │  ← Start container, wait for /health
    │  4. curl POST /analyze  │  ← Send a real request to the container
    │  5. docker stop         │  ← Cleanup
    └─────────────────────────┘
```

### Key CI/CD concepts demonstrated

| Concept | How we use it |
|---------|--------------|
| **Triggered on push** | `on: push: branches: [main, master]` — runs automatically |
| **Quality gate** | Lint must pass before tests run |
| **Job dependency** | `needs: lint-and-test` — Docker build only runs if tests pass |
| **Dependency caching** | `cache: "pip"` — pip packages are cached between runs for speed |
| **Integration testing** | The Docker job doesn't just build — it starts the container and sends a real HTTP request |
| **Fail-fast** | If lint fails, tests don't run. If tests fail, Docker doesn't build. Saves time and compute. |

### What `ruff check .` catches

Ruff is a fast Python linter. Our config (`pyproject.toml`):

```toml
[tool.ruff.lint]
select = ["E", "F", "I"]  # E=errors, F=pyflakes, I=import ordering
```

- **E**: Syntax errors, invalid Python
- **F**: Unused imports, undefined variables, unreachable code
- **I**: Unsorted imports (enforces consistent ordering)

---

## Docker and Containerization

### What is Docker?

Docker packages an application and all its dependencies into a **container** — a lightweight, isolated environment that runs identically on any machine. "Works on my machine" is solved.

### Our Dockerfile

```dockerfile
FROM python:3.13-slim          # Start with a minimal Python image
WORKDIR /app                   # Set working directory inside container
COPY requirements.txt .        # Copy deps first (cached layer)
RUN pip install --no-cache-dir -r requirements.txt  # Install deps
COPY app/ ./app/               # Copy source code
COPY dashboard/ ./dashboard/
COPY FinancialPhraseBank/ ./FinancialPhraseBank/    # Copy dataset
EXPOSE 8000                    # Document that the API runs on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker layer caching

The Dockerfile is ordered so that `requirements.txt` is copied and installed **before** the source code. Why? Docker caches each layer. If you only change source code (not dependencies), Docker skips the `pip install` step on rebuild — making rebuilds much faster.

### Docker Compose — multi-container orchestration

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 3

  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_BASE=http://api:8000/api/v1    # Service discovery by name
    depends_on:
      api:
        condition: service_healthy          # Wait for API health check
```

Key concepts:
- **Two services**: `api` and `dashboard` are separate containers
- **Health check**: Docker pings `/health` every 10 seconds to verify the API is alive
- **`depends_on` with `service_healthy`**: Dashboard doesn't start until the API is confirmed healthy
- **Service discovery**: Dashboard calls `http://api:8000` — Docker resolves `api` to the correct container IP automatically
- **Environment variables**: `API_BASE` is set per-environment (local = `localhost`, Docker = `api`)

### .dockerignore

```
.venv/
__pycache__/
*.pyc
.git/
tests/
```

Like `.gitignore` but for Docker. Prevents unnecessary files from being copied into the image (smaller image, faster builds).

---

## Testing with Pytest

### What is Pytest?

Pytest is Python's standard testing framework. It:
- Discovers test files automatically (`test_*.py`)
- Discovers test functions automatically (`test_*()`)
- Provides clear output on failures
- Integrates with CI/CD pipelines

### Our test suite — 12 tests

```
tests/test_sentiment.py

test_health                          ← API health check returns "ok"
test_positive_headline               ← Positive news → "positive"
test_negative_headline               ← Bankruptcy news → "negative"
test_neutral_headline                ← Routine news → "neutral"
test_get_endpoint                    ← GET method works
test_empty_headline_rejected         ← Empty input → 422 error
test_batch_analysis                  ← Batch of 3 headlines works
test_batch_limit_enforced            ← 51 headlines → 422 error
test_batch_empty_rejected            ← Empty list → 422 error
test_samples_endpoint                ← Samples returns labeled data
test_samples_filter_by_sentiment     ← Filter by positive/negative/neutral
test_samples_invalid_sentiment_rejected ← Invalid filter → 422 error
```

### How TestClient works

```python
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # Starts the app (triggers lifespan → loads model)
        yield c                  # All tests in this module share one client

def test_positive_headline(client):
    r = client.post("/api/v1/analyze",
                    json={"headline": "Company reports record quarterly profits"})
    assert r.status_code == 200
    assert r.json()["sentiment"] == "positive"
```

`TestClient` simulates HTTP requests **without starting a real server**. This is why tests run in 0.3 seconds — no network, no port binding.

`scope="module"` means the model is loaded once for all 12 tests, not once per test.

---

## Streamlit Dashboard

### What is Streamlit?

A Python library that converts a script into a web application. No HTML, CSS, or JavaScript needed.

### Our dashboard

- **Text input** — type any financial headline
- **Analyze button** — sends it to the FastAPI server via HTTP POST
- **Big sentiment display** — shows result with color coding (green/red/gray) and emoji
- **"Get More Info" expander** — click to see per-class scores, confidence, and raw JSON

### How it connects to the API

```python
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

def analyze(headline: str) -> dict:
    r = httpx.post(f"{API_BASE}/analyze", json={"headline": headline}, timeout=30)
    return r.json()
```

- Locally: calls `http://localhost:8000`
- In Docker: calls `http://api:8000` (set via `API_BASE` env var in docker-compose)

---

## How to Run the Project

### Option 1: Local (without Docker)

```bash
# Create virtual environment
python3 -m venv --system-site-packages .venv

# Install dependencies
.venv/bin/pip install -r requirements.txt

# Start the API
.venv/bin/uvicorn app.main:app --port 8000

# Open Swagger UI
# → http://localhost:8000/docs

# (In a second terminal) Start the dashboard
.venv/bin/streamlit run dashboard/streamlit_app.py

# Run tests
.venv/bin/python -m pytest tests/ -v

# Run linter
.venv/bin/python -m ruff check .

# Run benchmark
.venv/bin/python -m scripts.benchmark
```

### Option 2: Docker Compose

```bash
# Start both API and Dashboard
docker compose up --build

# API  → http://localhost:8000/docs
# Dashboard → http://localhost:8501

# Stop everything
docker compose down
```

---

## Viva Q&A — Expected Questions and Answers

### About the AI/ML component

**Q: What is sentiment analysis?**
> Sentiment analysis is a Natural Language Processing (NLP) technique that classifies text into emotional categories — positive, negative, or neutral. We use it to determine whether a financial news headline is good or bad for a stock.

**Q: What is VADER and how does it work?**
> VADER stands for Valence Aware Dictionary and sEntiment Reasoner. It's a rule-based model that uses a dictionary of ~7,500 words, each with a pre-assigned sentiment score. It also applies rules for capitalization, negation, and punctuation. It outputs a compound score from -1 to +1, which we threshold to classify as positive (>=0.05), negative (<=-0.05), or neutral.

**Q: Why did you use VADER instead of a deep learning model?**
> We originally planned to use FinBERT (a BERT model fine-tuned on financial text), but it requires PyTorch (~2GB) and the model weights (~450MB), which exceeded our disk space. VADER is lightweight (~125KB), requires no GPU, and is a valid and widely-used baseline for sentiment analysis. The trade-off is that VADER doesn't understand financial jargon as well as FinBERT.

**Q: What is the Financial PhraseBank dataset?**
> It's a dataset of 4,846 financial news sentences labeled by 16 human annotators as positive, negative, or neutral. We use the subset where all 16 annotators agreed (2,264 sentences) — the highest quality labels. It was created by Malo et al. in 2014 and is a standard benchmark for financial NLP.

**Q: What is the accuracy of your model?**
> You can run `python -m scripts.benchmark` to see exact numbers. VADER achieves approximately 55-65% accuracy on financial text (compared to ~85% for FinBERT). The gap is because VADER doesn't understand domain-specific phrases like "earnings miss" or "revenue headwinds."

### About the REST API

**Q: What is a REST API?**
> REST (Representational State Transfer) is an architectural style for web APIs. It uses standard HTTP methods (GET, POST, PUT, DELETE) to operate on resources identified by URLs. Data is exchanged as JSON. Each request is stateless — the server doesn't remember previous requests.

**Q: What is FastAPI?**
> FastAPI is a modern Python web framework for building REST APIs. It's the fastest Python framework (comparable to Node.js and Go), auto-generates Swagger documentation, and uses Pydantic for input/output validation. It's based on ASGI (Asynchronous Server Gateway Interface).

**Q: What is Pydantic and why is it important?**
> Pydantic is a data validation library. We define schemas (like `HeadlineRequest` with a `headline: str` field), and Pydantic automatically validates every incoming request. If someone sends invalid data (wrong type, missing field), it returns a 422 error with a clear message. This prevents bad data from reaching our model.

**Q: Why do you have both GET and POST endpoints for `/analyze`?**
> POST is the standard way — you send the headline in the JSON body. GET is a convenience endpoint — you can test it directly in a browser URL bar (`/api/v1/analyze?headline=...`). GET requests are also cacheable by browsers and CDNs.

**Q: What is the `/health` endpoint for?**
> It's a standard DevOps pattern. Monitoring tools, load balancers, Docker health checks, and Kubernetes readiness probes all call this endpoint to verify the service is alive. Our `docker-compose.yml` uses it to wait for the API before starting the dashboard.

### About DevOps / CI/CD

**Q: What is Docker and why did you use it?**
> Docker packages an application and all its dependencies into a container — a lightweight, isolated environment that runs identically on any machine. We used it so our project can be deployed anywhere without "works on my machine" issues. Our Docker image includes Python, all pip packages, the source code, and the dataset.

**Q: What is Docker Compose?**
> Docker Compose is a tool for running multiple Docker containers together. We have two services: `api` (FastAPI on port 8000) and `dashboard` (Streamlit on port 8501). Docker Compose handles networking between them — the dashboard finds the API by service name (`http://api:8000`), not by IP address.

**Q: Explain your CI/CD pipeline.**
> We use GitHub Actions. On every `git push` or pull request, it automatically:
> 1. Installs dependencies
> 2. Runs `ruff` to lint the code (catches errors and import issues)
> 3. Runs all 12 pytest tests
> 4. If those pass, builds a Docker image
> 5. Starts the Docker container and verifies the `/health` endpoint responds
> 6. Sends a real POST request to `/api/v1/analyze` and checks the response
>
> If any step fails, the pipeline stops and the team is notified. This prevents broken code from being merged.

**Q: What is the difference between CI and CD?**
> CI (Continuous Integration) automatically tests and validates code on every push. CD (Continuous Deployment) automatically deploys passing code to production. Our pipeline handles CI. For CD, we'd add a step to push the Docker image to a registry (like Docker Hub or AWS ECR) and deploy it to a cloud service.

**Q: What is a health check in Docker?**
> A health check is a command Docker runs periodically to verify a container is functioning correctly. Ours calls the `/health` endpoint every 10 seconds. If it fails 3 times, Docker marks the container as "unhealthy." Our dashboard uses `depends_on: condition: service_healthy` so it only starts after the API is confirmed healthy.

### About testing

**Q: What is Pytest?**
> Pytest is Python's standard testing framework. It automatically discovers test files (`test_*.py`) and test functions (`test_*()`). It provides detailed error messages on failure and integrates with CI/CD pipelines.

**Q: How does TestClient work?**
> FastAPI's TestClient simulates HTTP requests without starting a real server — no network, no port binding. It triggers the app's lifespan (loading the model and dataset) and lets you test endpoints by calling `client.get()` and `client.post()` as if they were real HTTP requests.

**Q: What do your tests cover?**
> We test 4 categories:
> 1. **Happy path**: positive, negative, and neutral headlines return correct sentiment
> 2. **Input validation**: empty headlines and oversized batches return 422 errors
> 3. **Batch processing**: multiple headlines are processed and returned correctly
> 4. **Data endpoints**: samples endpoint returns labeled data and supports filtering

### General / Architecture

**Q: Why is the model loaded at startup and not per-request?**
> Loading the dataset (2,264 sentences) on every request would be wasteful — it would add unnecessary latency. The lifespan pattern loads it once when the server starts, keeps it in memory, and all requests share the same loaded instance. This is a standard pattern for serving ML models.

**Q: How does the dashboard communicate with the API?**
> The dashboard (Streamlit) sends HTTP requests to the FastAPI server using the `httpx` library. The API URL is configurable via the `API_BASE` environment variable — locally it's `http://localhost:8000/api/v1`, in Docker it's `http://api:8000/api/v1`. This is an example of service discovery and environment-based configuration.

**Q: What would you add if you had more time?**
> 1. **FinBERT** for better financial-domain accuracy
> 2. **CD step** — push Docker image to a registry and deploy to AWS/GCP
> 3. **Rate limiting** — prevent abuse of the API
> 4. **Database** — store prediction history for analytics
> 5. **Monitoring** — Prometheus metrics + Grafana dashboard for API latency and error rates
