FROM "bentoml/model-server"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "80"]
