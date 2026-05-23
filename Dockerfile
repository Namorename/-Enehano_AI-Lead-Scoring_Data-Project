# Enehano Lead Intelligence — single image, two services (Streamlit + FastAPI)
# Build:    docker build -t enehano-leads .
# Run app:  docker run --rm -p 8501:8501 enehano-leads
# Run API:  docker run --rm -p 8000:8000 enehano-leads \
#               uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
# In production, docker-compose.yml runs both services side-by-side.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for SHAP / numpy / scikit-learn wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code + trained model artifacts + data
COPY . .

# Streamlit by default; docker-compose overrides command for the API container
EXPOSE 8501 8000
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

