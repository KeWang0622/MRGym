# The scanner service of MR Gym (tier 3: the agent runs in another container; see compose.yaml and README.md).
# Build from the repository root:  docker build -f integrations/container/Dockerfile -t mrgym-scanner .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir /src \
    && rm -rf /src \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin scanner \
    && mkdir -p /var/log/scanner /var/lib/scanner /var/run/scanner-admin \
    && chown 10001:10001 /var/log/scanner /var/lib/scanner /var/run/scanner-admin \
    && chmod 700 /var/log/scanner /var/lib/scanner /var/run/scanner-admin

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=12 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["mrgym"]
CMD ["serve", "--help"]
