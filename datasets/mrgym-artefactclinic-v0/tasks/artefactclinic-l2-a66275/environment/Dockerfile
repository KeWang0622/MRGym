FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
RUN pip install --no-cache-dir mrgym==0.1.0.dev0 && useradd -m agent && mkdir -p /app /run/mrscan && chown agent /app
COPY --chown=agent --chmod=400 agent_token /run/mrscan/token
ENV MRSCAN_URL=http://scanner:8000 MRSCAN_TOKEN_FILE=/run/mrscan/token OMP_NUM_THREADS=1
WORKDIR /app
