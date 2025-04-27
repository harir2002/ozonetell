FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY test.py .
COPY gunicorn_config.py .
COPY templates/index.html templates/index.html
EXPOSE 8000
ENV FLASK_ENV=production
CMD ["gunicorn", "--config", "gunicorn_config.py", "--bind", "0.0.0.0:8000", "test:app"]