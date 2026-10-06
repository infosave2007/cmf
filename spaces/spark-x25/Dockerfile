# Spark Lab runs the released Cortiq runtime rather than rebuilding a model stack.
# Public CMF weights are fetched only on first use and never committed to this Space.
FROM python:3.11-slim

ARG CORTIQ_VERSION=v0.8.12
ARG CORTIQ_TARGET=x86_64-unknown-linux-gnu

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libegl1 \
        libgl1 \
        libglx0 \
        libvulkan1; \
    rm -rf /var/lib/apt/lists/*; \
    archive="cortiq-${CORTIQ_TARGET}.tar.gz"; \
    release="https://github.com/infosave2007/cmf/releases/download/${CORTIQ_VERSION}"; \
    curl --fail --location --retry 3 --output "/tmp/${archive}" "${release}/${archive}"; \
    curl --fail --location --retry 3 --output "/tmp/${archive}.sha256" "${release}/${archive}.sha256"; \
    cd /tmp; \
    sha256sum --check "${archive}.sha256"; \
    tar --extract --gzip --file "${archive}" --directory /usr/local/bin cortiq; \
    rm "/tmp/${archive}" "/tmp/${archive}.sha256"; \
    chmod +x /usr/local/bin/cortiq; \
    cortiq --version

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/home/user/.local/bin:$PATH \
    GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
