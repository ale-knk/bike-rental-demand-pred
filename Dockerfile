FROM python:3.12-slim

WORKDIR /app
RUN pip install --upgrade pip

COPY . .
RUN pip install --no-cache-dir .

ENTRYPOINT ["train"]
CMD ["--config_path", "/app/config_examples/config_train.json"]