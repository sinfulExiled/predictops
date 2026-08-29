# PredictOps — one image that builds the dashboard and serves the whole product.
# CPU only. No GPU, no database server, no API key.

FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY predictops/ ./predictops/
COPY tests/ ./tests/
COPY *.py pytest.ini README.md REPRODUCTION.md ./
COPY --from=ui /ui/dist ./frontend/dist

# Generate the dataset at build time so the image is immediately usable.
RUN python generate_data.py --machines 80 --days 30 --seed 42

EXPOSE 8000
CMD ["uvicorn", "predictops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
