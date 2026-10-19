FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
# The model is downloaded on first real use (needs internet); mount a volume at /models and set ASR_MODEL_DIR=/models to keep it.
# Mock evaluation needs no model: docker run --rm --entrypoint python <image> -m asr_transcriber.evaluate --mock
ENTRYPOINT ["python", "-m", "asr_transcriber"]
