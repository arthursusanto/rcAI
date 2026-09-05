# Image for the RCA inference service (`rca serve`).
#
# Build context is the REPO ROOT, not deploy/, because the package and the model
# artifact both live there:
#
#   docker build -f deploy/Dockerfile -t rca-serve:local --build-arg MODEL_DIR=artifacts/xgb .
#
# The model is BAKED IN rather than mounted. A ConfigMap cannot carry it (Kubernetes
# caps a ConfigMap at 1 MiB; the trained models in artifacts/ are 3.8 MB (xgb) and
# 8.8 MB (rf-fpr)), and a PVC would need a separate copy-in step before the server can
# start. Baking makes the image the single versioned unit: one image = one model.
FROM python:3.12-slim

# Model directory to bake in, relative to the build context. artifacts/ is gitignored,
# so this is whatever the training run last wrote on the build machine.
ARG MODEL_DIR=artifacts/xgb

# Dependencies and package. pyproject.toml declares no `readme`, so only these three
# paths are needed for `pip install .`; rca/serve/ui/*.html comes along as package data.
# LICENSE is required because pyproject sets `license-files = ["LICENSE"]` -- without it
# the build fails rather than merely omitting the licence from the wheel.
WORKDIR /src
COPY pyproject.toml LICENSE ./
COPY rca ./rca
RUN pip install --no-cache-dir . && rm -rf /src

WORKDIR /opt/rca
COPY ${MODEL_DIR}/ /opt/rca/model/

# Real passwd entry for the UID deploy/k8s/rca-serve.yaml runs as, so HOME exists and
# `expanduser("~")` resolves. The server needs a writable /tmp too:
# rca.serve.stream._Tail writes each polled chunk to a NamedTemporaryFile.
RUN useradd --uid 10001 --user-group --create-home rca
USER 10001:10001

# Unbuffered so `kubectl logs` is live rather than block-buffered.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# `rca serve` reads JAEGER_URL / GRAFANA_URL from the environment for evidence links.
ENTRYPOINT ["rca", "serve"]
CMD ["--model", "/opt/rca/model", "--host", "0.0.0.0", "--port", "8000"]
