# Exact base image digest used by the scorer (Python 3.13 + SQLite match).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

# Download sqlglot at the exact pinned commit (no git/apt needed)
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz', '/tmp/sqlglot.tar.gz')" \
    && tar -xzf /tmp/sqlglot.tar.gz -C /app \
    && rm /tmp/sqlglot.tar.gz \
    && mv /app/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /app/sqlglot \
    && test -f /app/sqlglot/sqlglot/__init__.py

COPY selector.py patch_utils.py worker_pack.py inmem_loader.py build_caches.py optimizer_checkpoints.py stress_slicing.py maps.zlib fixture_cov.zlib validator_cov.zlib db_query_cache.zlib invariants.zlib tpcds_plans.zlib opt_stages.zlib /app/

# Precompile bytecode + pre-parse read-only TPC-DS tables & fast plans at build time (zero runtime CSV/serde cost)
RUN SQLGLOT_ROOT=/app/sqlglot python /app/build_caches.py \
    && python -m compileall -q /app/sqlglot/sqlglot /app/sqlglot/tests /app/sqlglot/sqlglotc /app/*.py \
    && chmod -R a+rX /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    SKIP_INTEGRATION=0 \
    SQLGLOT_ROOT=/app/sqlglot \
    PYTHONPATH=/app/sqlglot:/app \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

CMD ["python", "/app/selector.py"]
