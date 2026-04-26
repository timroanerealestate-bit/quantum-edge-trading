"""Rich terminal display for scan results."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import datetime

console = Console()


def _score_color(score: float) -> str:
    if score >= 75:
        return "bold green"
    if score >= 60:
        return "green"
    if score >= 45:
        return "yellow"
    if score >= 30:
        return "orange3"
    return "red"


def _rec_color(rec: str) -> str:
    colors = {
        "STRONG BUY": "bold green",
        "BUY": "green",
        "WATCH": "yellow",
        "NEUTRAL": "white",
        "AVOID": "red",
    }
    return colors.get(rec, "white")


def print_header():
    console.print(Panel(
        "[bold cyan]Training Bot — Institutional Flow + Technical Scanner[/bold cyan]\n"
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        box=box.DOUBLE,
        expand=False,
    ))


def print_summary_table(results: list[dict]):
    table = Table(
        title="[bold]Ranked Opportunities[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="right", width=4)
    table.add_column("Symbol", width=7)
    table.add_column("Price", justify="right", width=9)
    table.add_column("Chg%", justify="right", width=8)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Rec", width=12)
    table.add_column("Inst", justify="right", width=6)
    table.add_column("Tech", justify="right", width=6)
    table.add_column("Vol", justify="right", width=6)
    table.add_column("Sent", justify="right", width=6)

    for i, r in enumerate(results, 1):
        score = r["composite_score"]
        rec = r["recommendation"]
        scores = r["scores"]
        change = r.get("change_pct", "N/A")
        if isinstance(change, str) and change.endswith("%"):
            change_f = float(change.replace("%", ""))
            change_str = f"[{'green' if change_f >= 0 else 'red'}]{change}[/]"
        else:
            change_str = str(change)

        table.add_row(
            str(i),
            f"[bold]{r['symbol']}[/bold]",
            str(r.get("price", "N/A")),
            change_str,
            f"[{_score_color(score)}]{score}[/]",
            f"[{_rec_color(rec)}]{rec}[/]",
            str(scores.get("institutional", 0)),
            str(scores.get("technical", 0)),
            str(scores.get("volume", 0)),
            str(scores.get("sentiment", 0)),
        )

    console.print(table)


def print_detail(r: dict):
    symbol = r["symbol"]
    score = r["composite_score"]
    color = _score_color(score)

    lines = []
    lines.append(f"[bold]{symbol}[/bold]  Price: {r.get('price','N/A')}  Change: {r.get('change_pct','N/A')}")
    lines.append(f"Composite Score: [{color}]{score}/100[/]  →  [{_rec_color(r['recommendation'])}]{r['recommendation']}[/]\n")

    lines.append("[underline]Technical Signals:[/underline]")
    for d in r["tech"].get("details", []):
        lines.append(f"  • {d}")

    lines.append("\n[underline]Institutional Flow:[/underline]")
    for d in r.get("inst_details", []):
        lines.append(f"  • {d}")

    lines.append("\n[underline]Volume:[/underline]")
    for d in r.get("vol_details", []):
        lines.append(f"  • {d}")

    lines.append("\n[underline]News & Market Sentiment:[/underline]")
    sent   = r.get("sentiment", {})
    source = sent.get("source", "unknown")
    label  = sent.get("label", "N/A")
    score_s = sent.get("sentiment_score", "N/A")
    bull   = sent.get("bullish_count", 0)
    bear   = sent.get("bearish_count", 0)
    neut   = sent.get("neutral_count", 0)
    avg    = sent.get("sentiment_avg")

    label_color = "green" if label == "Bullish" else ("red" if label == "Bearish" else "yellow")
    lines.append(
        f"  Source: [dim]{source}[/dim]  "
        f"Label: [{label_color}]{label}[/]  "
        f"Score: {score_s}/100  "
        f"Articles: {sent.get('article_count', 0)}"
    )
    if bull or bear or neut:
        lines.append(f"  Breakdown: [green]{bull} bullish[/] / [red]{bear} bearish[/] / {neut} neutral"
                     + (f"  avg={avg:+.3f}" if avg is not None else ""))
    for h in sent.get("top_headlines", [])[:4]:
        lines.append(f"  [dim]• {h[:92]}[/dim]")

    # Convergence notes
    conv = r.get("convergence_notes", [])
    if conv:
        lines.append("\n[underline]Signal Convergence:[/underline]")
        for c in conv:
            lines.append(f"  {c}")

    console.print(Panel("\n".join(lines), title=f"[bold cyan]{symbol} Detail[/bold cyan]", box=box.ROUNDED))


def print_institutional_heatmap(heatmap: dict, top_n: int = 15):
    table = Table(
        title="[bold]Institutional Holdings Heatmap (13F)[/bold]",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Issuer Name", width=35)
    table.add_column("# Institutions", justify="right", width=15)
    table.add_column("Total Value ($M)", justify="right", width=18)
    table.add_column("Holders", width=40)

    for name, data in list(heatmap.items())[:top_n]:
        table.add_row(
            name[:35],
            str(data["holders"]),
            f"${data['total_value_M']:,.0f}",
            ", ".join(data["institutions"][:3]),
        )

    console.print(table)
