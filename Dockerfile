FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and local dataset
COPY app/ ./app/
COPY dashboard/ ./dashboard/
COPY FinancialPhraseBank/ ./FinancialPhraseBank/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
