# Base image pinned by digest (python:3.12-slim-bookworm, Python 3.12.16, created 2026-10-28).
FROM python@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/work/src \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 tini \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --root-user-action=ignore -r /tmp/requirements.lock

# Non-root user matching the host UID/GID so generated files stay editable on the host.
RUN (getent group "${HOST_GID}" || groupadd -g "${HOST_GID}" app) \
 && useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash app \
 && git config --system --add safe.directory /work

USER app
WORKDIR /work
# tini -g forwards SIGTERM to the whole process group (sweep workers included) so runs can checkpoint before exiting.
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["python", "-m", "delayed_local_cp.cli", "--help"]
