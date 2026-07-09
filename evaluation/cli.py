"""Command-line interface for the evaluation harness."""

from datetime import datetime

import typer

from evaluation.generate_qa import generate_qa_dataset
from evaluation.run import NoReviewedQAError, run_evaluation

app = typer.Typer()


def _generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@app.command("generate-qa")
def generate_qa_command(target_count: int = 40) -> None:
    """Generate candidate QA pairs from the shared input documents."""
    entries = generate_qa_dataset(target_count=target_count)
    typer.echo(f"Generated {len(entries)} QA pairs")


@app.command()
def run(
    run_id: str | None = typer.Argument(
        None, help="実行を識別する文字列。省略時は現在時刻から自動生成する"
    ),
    question_id: list[str] = typer.Option(
        [],
        "--question-id",
        "-q",
        help="このIDの質問だけ再実行する（複数指定可、省略時は全件）",
    ),
) -> None:
    """Run GraphRAG/Vector RAG, judge the answers, and write the report."""
    if run_id is None:
        run_id = _generate_run_id()

    try:
        summary = run_evaluation(run_id=run_id, question_ids=question_id or None)
    except NoReviewedQAError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"run_id: {run_id}")
    typer.echo(f"結果: {summary['output_path']}")
    typer.echo(f"採点結果: {summary['scored_path']}")
    typer.echo(f"レポート: {summary['report_path']}")


if __name__ == "__main__":
    app()
