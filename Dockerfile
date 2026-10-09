# Usei Python 3.12, a mesma versão principal do ambiente de treinamento.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalei as versões das bibliotecas usadas pelo modelo exportado.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Incluí o modelo pronto, a API e os arquivos da interface.
COPY app.py modelo.pkl ./
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000
CMD ["python", "app.py"]
