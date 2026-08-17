# The Space builds the renderer from source and then never touches Python
# for anything numerical: `app.py` only collects a prompt and shells out.
FROM rust:1.85-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 https://github.com/infosave2007/cmf . \
 && cargo build --release -p cortiq-cli \
 && strip target/release/cortiq

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates libvulkan1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/cortiq /usr/local/bin/cortiq
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user gradio==5.9.1 huggingface_hub>=0.26
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
