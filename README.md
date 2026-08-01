# Fitness Assistant

A RAG (Retrieval-Augmented Generation) application that answers questions about fitness exercises — form, muscle groups, equipment, and exercise replacements — using a dataset of 1,324 exercises.

## Features

- **Search**: keyword search over exercise records (name, muscles, equipment, instructions) with `minsearch`
- **LLM answers**: Groq-powered answers grounded in retrieved exercise context
- **Evaluation**: ground-truth question generation and retrieval metrics (hit rate, MRR)
- **API** (planned): Flask interface
- **Monitoring** (planned): PostgreSQL + Grafana
- **Deployment** (planned): Docker

## Architecture

```
src/fitness_assistant/
├── ingestion/     # dataset loading and document preparation
├── search/        # minsearch index (single source of truth) and query logic
├── rag/           # RAG pipeline: retrieve -> prompt -> answer
├── llm/           # Groq client and prompt templates
├── eval/          # ground-truth generation and retrieval metrics
└── main.py        # example usage
data/
├── raw/           # source dataset (exercises.json)
├── processed/     # prepared documents consumed by the index
└── eval/          # generated ground-truth retrieval questions
```

The index is built once and cached: every component imports `get_index()` / `load_documents()` from `search/index.py`.

## Dataset

`data/raw/exercises.json` — 1,324 exercises from the [exercises-dataset](https://github.com/hasaneyldrm/exercises-dataset) project (based on ExerciseDB v1), with multilingual instructions, muscle groups, equipment, and step-by-step guidance.

## Setup

```bash
uv sync
cp .env.example .env   # set GROQ_API_KEY
```

## Usage

Run an example query through the full RAG pipeline:

```bash
uv run python src/fitness_assistant/main.py
```

Generate ground-truth retrieval questions and run evaluation:

```bash
uv run python src/fitness_assistant/eval/retreiver_eval.py
```

This writes `data/eval/ground-truth-retrieval.csv` and prints hit rate / MRR. Note that ground-truth generation calls the LLM for every exercise, so it consumes API quota.

## Roadmap

- Flask API (`/search`, `/answer`, `/health`)
- Query/answer logging to PostgreSQL and Grafana dashboards
- Docker + docker-compose for the full stack

## License

See [LICENSE](LICENSE).
