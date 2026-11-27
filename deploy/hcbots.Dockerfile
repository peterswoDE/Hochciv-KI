# Online client of the hardcoded bots (../hardcoded comes in as build context "hc", see docker-compose.yml).
FROM node:24-bookworm-slim AS node

FROM python:3.13-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=hc requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --from=hc bots /app/bots
COPY --from=hc online /app/online
COPY --from=hc template /app/template
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-m", "online.client"]
