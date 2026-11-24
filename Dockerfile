FROM python:3.12-slim-bookworm

ARG TORCH_VARIANT=cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GNM_CACHE_DIR=/cache/gnm \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Select the PyTorch wheel explicitly. A CUDA build must not install a CPU wheel.
RUN case "$TORCH_VARIANT" in cpu|cu128) ;; *) exit 1 ;; esac \
    && python -m pip install --no-cache-dir \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
        "torch==2.7.1+${TORCH_VARIANT}"

COPY docker/constraints.txt /tmp/constraints.txt
COPY gnm/shape/ /tmp/gnm-shape/
# A regular wheel installation is required by the landmark resource test.
RUN python -m pip install --no-cache-dir \
        --constraint /tmp/constraints.txt '/tmp/gnm-shape[pytorch,jax,dev]' \
    && python -m pip check \
    && rm -rf /tmp/gnm-shape

WORKDIR /app
COPY personalab/ /app/personalab/
COPY tests/ /app/tests/
COPY docker/require-cuda.sh /usr/local/bin/require-cuda
RUN chmod 0755 /usr/local/bin/require-cuda
CMD ["python", "-m", "personalab.gnm_mesh", "--help"]
