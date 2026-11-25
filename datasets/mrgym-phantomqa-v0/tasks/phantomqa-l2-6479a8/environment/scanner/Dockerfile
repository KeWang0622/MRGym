FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
RUN pip install --no-cache-dir mrgym==0.1.0.dev0 && useradd -m scanner && mkdir -p /var/log/scanner /var/lib/scanner /episode && chown scanner /var/log/scanner /var/lib/scanner /episode && chmod 700 /var/log/scanner /var/lib/scanner /episode
COPY --chown=scanner --chmod=600 episode.json agent_token /episode/
ENV OMP_NUM_THREADS=1
USER scanner
CMD ["mrgym", "serve", "--episode-file", "/episode/episode.json", "--agent-token-file", "/episode/agent_token", "--log-dir", "/var/log/scanner", "--state-dir", "/var/lib/scanner", "--host", "0.0.0.0", "--port", "8000"]
