FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build essentials needed for some native wheels during build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       pkg-config \
       libpq-dev \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements_render.txt ./

# Install backend requirements (lighter, pinned file)
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy project
COPY . /app

EXPOSE 8000

# Default command for the backend service
CMD ["gunicorn", "src.api.app:app", "--bind", "0.0.0.0:8000"]
