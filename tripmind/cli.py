from __future__ import annotations

from pathlib import Path

import typer

from tripmind.llm import LLMConfigurationError
from tripmind.memory import JsonMemoryStore
from tripmind.renderer import render_markdown
from tripmind.runtime import TripMindRuntime

app = typer.Typer(help="TripMind travel-planning agent demo.")


@app.command()
def plan(
    text: str = typer.Argument(..., help="Natural language travel request."),
    user_id: str = typer.Option("demo", "--user-id", "-u", help="Memory namespace for this user."),
    memory_path: Path = typer.Option(Path(".tripmind_memory.json"), "--memory-path", help="JSON memory file."),
    model: str | None = typer.Option(None, "--model", help="LLM model. Defaults to TRIPMIND_LLM_MODEL or qwen3.6-plus."),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic local mode instead of real LLM calls."),
    review_only: bool = typer.Option(False, "--review-only", help="Stop at reviewing instead of confirmed."),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON instead of Markdown."),
) -> None:
    """Generate a travel plan from one natural-language request."""
    try:
        runtime = TripMindRuntime(memory_path=memory_path, use_llm=not mock, model=model)
    except LLMConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    run = runtime.run(text=text, user_id=user_id, auto_confirm=not review_only)
    if json_output:
        typer.echo(run.model_dump_json(indent=2))
        return
    typer.echo(render_markdown(run))


@app.command("memory")
def show_memory(
    user_id: str = typer.Option("demo", "--user-id", "-u", help="Memory namespace for this user."),
    memory_path: Path = typer.Option(Path(".tripmind_memory.json"), "--memory-path", help="JSON memory file."),
) -> None:
    """Show saved preferences for one user."""
    memory = JsonMemoryStore(memory_path).get(user_id)
    typer.echo(memory.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
