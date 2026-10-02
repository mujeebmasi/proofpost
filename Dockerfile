FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY evals ./evals
# Editable install: the example and benchmark folders are found relative to the source tree.
RUN pip install --no-cache-dir -e .
RUN useradd -m app
USER app
EXPOSE 8000
CMD ["proofpost", "serve", "--host", "0.0.0.0", "--port", "8000"]
