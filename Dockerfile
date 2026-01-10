FROM python:3.13-slim

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

# deps requirements
COPY requirements.txt .

# install deps
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless

# src
COPY src/ src/
COPY configs/ configs/

# pretrained model
COPY saved_models/ saved_models/

ENTRYPOINT ["python", "-m", "src.predict"]
