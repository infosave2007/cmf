# Sandbox of the R set: model-written commands run here, as an unprivileged
# user, with no network (docker run --network none) and a read-only root.
FROM debian:stable-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps bc jq file zip unzip gawk \
 && rm -rf /var/lib/apt/lists/*
COPY fixture.sh /fixture.sh
RUN bash /fixture.sh && rm /fixture.sh \
 && useradd -u 1000 -m user \
 && printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo
USER 1000
WORKDIR /tmp
