# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base image: slim = full Debian minus docs/extra tools, keeps the image
# smaller than the default python:3.10 image without losing anything we need.
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers less, so `docker logs`
# shows Streamlit's output immediately instead of holding it in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Install dependencies FIRST, copy code SECOND.
# Docker caches each instruction as a "layer." If requirements.txt hasn't
# changed, Docker reuses the cached pip-install layer instead of redoing a
# multi-minute TensorFlow install every time you change one line of app.py.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy everything else (app code + the 5 .keras model files).
COPY . .

# Documents which port the container listens on (informational — you still
# need -p on `docker run` to actually publish it to your host machine).
EXPOSE 8501

# --server.address=0.0.0.0 is required inside Docker: Streamlit's default
# (localhost) would only be reachable from INSIDE the container, not from
# your browser on the host machine.
# --server.headless=true stops Streamlit from trying to open a browser
# itself (there isn't one inside the container) and from prompting for an
# email on first run.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
