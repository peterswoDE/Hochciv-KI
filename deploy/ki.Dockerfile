# Hochciv-KI online client (the repo comes in as build context "ki", see docker-compose.yml).
# Code and Node worker are baked in; checkpoints, engine copies and the bot's data are mounted.
FROM node:24-bookworm-slim AS node

FROM python:3.13-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node

# CUDA 12.8 wheels (RTX 50 series). CPU only: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
COPY --from=ki requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=ki bridge /app/bridge
COPY --from=ki hochciv_ki /app/hochciv_ki
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-m", "hochciv_ki.online.client"]
