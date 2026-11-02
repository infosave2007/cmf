FROM python:3.12-slim-bookworm

ARG CORTIQ_VERSION=v0.8.53
ARG CORTIQ_SHA256=a1e468dadc633c04c175f90bf96f7e3721a6724f7a369ebef752a53e4c88ec32
ARG MODEL_REVISION=6c0c232aa3eb9cfe14fc42312ff011e71aaed2d1
ARG MODEL_SHA256=1ecbf169cd7cd5fd61b399bca70209ddd0c694b5ccc74eda81613fb884d376cc

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libvulkan1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 3 -o /tmp/cortiq.tar.gz \
       "https://github.com/infosave2007/cmf/releases/download/${CORTIQ_VERSION}/cortiq-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${CORTIQ_SHA256}  /tmp/cortiq.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/cortiq.tar.gz -C /usr/local/bin cortiq \
    && rm /tmp/cortiq.tar.gz \
    && cortiq --version \
    && useradd --create-home --uid 1000 user

USER user
WORKDIR /home/user/app
ENV HOME=/home/user PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CORTIQ_BIN=/usr/local/bin/cortiq CMF_MODEL=/home/user/app/embryo-56m.cmf PORT=7860

RUN curl --fail --location --retry 3 -o embryo-56m.cmf \
      "https://huggingface.co/infosave/Embryo-56M-cmf/resolve/${MODEL_REVISION}/embryo-56m.cmf" \
    && echo "${MODEL_SHA256}  embryo-56m.cmf" | sha256sum -c -

COPY --chown=user:user app.py index.html style.css app.js tests.json ./
EXPOSE 7860
CMD ["python", "app.py"]
