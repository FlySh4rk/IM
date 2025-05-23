# Multi-stage build per ottimizzare la dimensione dell'immagine
FROM python:3.9-slim as builder

# Installazione delle dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installazione di pipenv
RUN pip install --no-cache-dir pipenv

# Impostazione della directory di lavoro
WORKDIR /app

# Copia dei file di configurazione pipenv
COPY Pipfile* ./

# Fix del Pipfile per rimuovere dipendenze macOS e assicurare tensorflow standard
RUN sed -i '/tensorflow-metal/d; /tensorflow-macos/d' Pipfile || true && \
    grep -q "tensorflow = " Pipfile || echo 'tensorflow = "*"' >> Pipfile

# Installazione delle dipendenze in un virtual environment
ENV PIPENV_VENV_IN_PROJECT=1
RUN pipenv install --deploy --ignore-pipfile || \
    (rm -f Pipfile.lock && pipenv install)

# Stage finale - runtime
FROM python:3.9-slim

# Installazione delle dipendenze runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Creazione utente non-root per sicurezza
RUN useradd --create-home --shell /bin/bash app

# Impostazione della directory di lavoro
WORKDIR /app

# Copia del virtual environment dal builder stage
COPY --from=builder /app/.venv /app/.venv

# Copia del codice sorgente
COPY --chown=app:app . .

# Creazione delle directory necessarie se non esistono
RUN mkdir -p data models validation prod_data test src \
    && chown -R app:app /app

# Switch all'utente non-root
USER app

# Aggiunta del virtual environment al PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Porta di default (se il progetto ha un server web)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" || exit 1

# Comando di default - mantiene il container in esecuzione
CMD ["tail", "-f", "/dev/null"]