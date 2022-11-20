FROM "bentoml/model-server"
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "80"]
