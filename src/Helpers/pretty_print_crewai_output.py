from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
import json

def display_crew_output(crew_output):
    console = Console()

    # Raw Output
    console.print(f"[bold yellow]Raw Output:[/bold yellow] {crew_output.raw}\n")

    # JSON Output
    if crew_output.json_dict:
        console.print("[bold underline]JSON Output:[/bold underline]")
        console.print(json.dumps(crew_output.json_dict, indent=2))

    # Pydantic Output
    if crew_output.pydantic:
        console.print("\n[bold underline]Pydantic Output:[/bold underline]")
        console.print(crew_output.pydantic)

    # Tasks Output
    console.print("\n[bold underline]Tasks Output:[/bold underline]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Description", style="cyan", overflow="fold")
    table.add_column("Summary", style="green", overflow="fold")
    table.add_column("Raw Output", style="white", overflow="fold")
    table.add_column("Agent", style="yellow")
    table.add_column("Output Format", style="red")

    for task in crew_output.tasks_output:
        table.add_row(
            task.description.strip(),
            task.summary.strip(),
            task.raw.strip(),
            task.agent,
            task.output_format.value
        )

    console.print(table)

    # Token Usage
    console.print("\n[bold underline]Token Usage:[/bold underline]")
    usage_table = Table(show_header=True, header_style="bold blue")
    usage_table.add_column("Metric", style="dim")
    usage_table.add_column("Value", justify="right")

    # Convert UsageMetrics to dict before iterating
    token_usage_dict = crew_output.token_usage.dict()

    for key, value in token_usage_dict.items():
        usage_table.add_row(key.replace("_", " ").capitalize(), str(value))

    console.print(usage_table)