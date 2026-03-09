# Imagem base oficial otimizada para Python 3.12
FROM mcr.microsoft.com/vscode/devcontainers/python:3.12

# Copia o executável do uv de uma imagem oficial para o seu container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Evita ficheiros temporários e garante logs em tempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instala ferramentas essenciais e o IPython/Jupyter Kernel globalmente
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalação global de ferramentas interativas e formatadores
RUN pip install --no-cache-dir ipython ipykernel black flake8