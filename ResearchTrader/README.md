# ResearchTrader

ResearchTrader is a Gradio-based web app to help quants discover, explore, and operationalize the latest ArXiv quantitative-finance research. From free-text search to structured summaries, QA over papers, and Python-strategy generation, it accelerates the journey from insight to code.

## Features

- **Discover**: Fetch the newest ArXiv q-fin papers matching a user query
- **Structure & Summarize**: Auto-extract each paper's Objective, Methods, Results, Conclusions, then generate a human-readable summary
- **Explore**: Ask natural-language questions across the fetched set
- **Generate**: Produce boilerplate Python trading strategies inspired by the research

## Architecture

This project follows a modular architecture with:

- FastAPI backend with structured routers and services
- Gradio frontend for user interface
- Integration with ArXiv API for paper discovery
- OpenAI LLM for paper analysis and strategy generation

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

### Setup with uv

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research_trader.git
cd research_trader
```

2. Create a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies with uv:
```bash
uv pip install -e ".[dev]"
```

4. Create a `.env` file based on the provided template:
```bash
cp .env.example .env
# Edit .env to add your OpenAI API key
```

## Development

### Running the Backend

Start the FastAPI server:

```bash
uvicorn research_trader.main:app --reload
```

The API will be available at http://localhost:8000

API documentation is available at:
- http://localhost:8000/redoc (ReDoc)
- http://localhost:8000/swagger (Swagger UI)

### Running the Frontend

Start the Gradio frontend:

```bash
python frontend.py
```

The frontend will be available at http://localhost:7860

### Testing

Run tests with pytest:

```bash
uv pip install -e ".[test]"
pytest
```

### Code Formatting

Format code with black and isort:

```bash
black research_trader
isort research_trader
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/` | GET | Search for papers matching a query |
| `/search/{paper_id}` | GET | Get a specific paper by ID |
| `/structure/` | POST | Extract structured sections from a paper |
| `/structure/{paper_id}` | GET | Get structured sections for a paper |
| `/summarize/` | POST | Generate a comprehensive summary of a paper |
| `/summarize/{paper_id}` | GET | Get a comprehensive summary for a paper |
| `/qa/` | POST | Ask a question about papers |
| `/strategy/` | POST | Generate a trading strategy based on papers |
| `/strategy/stream` | POST | Generate a trading strategy (streaming) |

## Project Structure

```
research_trader/
├── main.py                         # FastAPI application instance & startup
├── config.py                       # Pydantic settings
├── models/                         # Pydantic models
│   ├── paper.py                    # Paper schema
│   ├── summary.py                  # Summary schema
│   └── strategy.py                 # Strategy schema
├── router/                         # API routers
│   ├── search.py                   # Search endpoint
│   ├── structure.py                # Paper structure endpoint  
│   ├── summarize.py                # Summarization endpoint
│   ├── qa.py                       # Q&A endpoint
│   └── strategy.py                 # Strategy generation endpoint
├── services/                       # Service layer
│   ├── arxiv_client.py             # ArXiv API client
│   ├── openai_client.py            # OpenAI API client
│   └── cache.py                    # Caching service
└── utils/                          # Utility functions
    └── errors.py                   # Custom error handling
```

## Configuration

The application uses environment variables for configuration, which can be set in a `.env` file. See `.env.example` for available options.

## License

MIT

## Acknowledgments

- ArXiv API for research paper access
- OpenAI for language model capabilities