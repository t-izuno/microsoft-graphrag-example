"""Command-line interface for the vector_rag pipeline."""

import typer

from vector_rag.answer import answer_query
from vector_rag.config import load_config
from vector_rag.indexer import run_index

app = typer.Typer()


@app.command()
def index() -> None:
    """Index all documents under the configured input directory."""
    config = load_config()
    count = run_index(config)
    typer.echo(f"Indexed {count} chunks")


@app.command()
def query(question: str) -> None:
    """Answer a question using the indexed chunks."""
    config = load_config()
    answer = answer_query(question, config)
    typer.echo(answer)


if __name__ == "__main__":
    app()
