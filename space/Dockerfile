# No Rust toolchain in the image: the renderer is the released Linux binary
# from the same tag as the model card documents. `app.py` never touches a
# tensor — it collects a prompt and shells out.
FROM python:3.11-slim

ARG CORTIQ_VERSION=v0.5.85
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates curl libvulkan1 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/cortiq.tar.gz \
      "https://github.com/infosave2007/cmf/releases/download/${CORTIQ_VERSION}/cortiq-x86_64-unknown-linux-gnu.tar.gz" \
 && tar -xzf /tmp/cortiq.tar.gz -C /usr/local/bin \
 && rm /tmp/cortiq.tar.gz \
 && chmod +x /usr/local/bin/cortiq \
 && cortiq --version

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /home/user/app

# The version specifiers are quoted: an unquoted `>=` is a shell redirect,
# which silently installs nothing and leaves gradio's CLI importing a
# `requests` that is not there.
RUN pip install --no-cache-dir --user \
      "gradio>=5.44,<6" \
      "huggingface_hub>=0.26" \
      "requests>=2.31"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
