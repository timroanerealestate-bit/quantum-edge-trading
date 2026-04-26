"""
Training Bot — main entry point.

Usage:
    python main.py                        # Full scan of default watchlist
    python main.py --symbols AAPL TSLA   # Scan specific symbols
    python main.py --detail AAPL          # Deep dive on one symbol
    python main.py --top 5               # Show only top N results
"""
import argparse
from rich.console import Console
from config import DEFAULT_WATCHLIST

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Training Bot — Stock Signal Scanner")
    parser.add_argument("--symbols", nargs="+", help="Symbols to scan (overrides default watchlist)")
    parser.add_argument("--detail", metavar="SYMBOL", help="Deep-dive detail on a single symbol")
    parser.add_argument("--top", type=int, default=3, help="Number of detailed reports to show (default: 3)")
    args = parser.parse_args()

    from display import print_header, print_summary_table, print_detail
    from signal_engine import build_composite_score, run_scan

    print_header()

    # --- Single symbol deep-dive ---
    if args.detail:
        console.print(f"[cyan]Running deep analysis on {args.detail.upper()}...[/cyan]")
        result = build_composite_score(args.detail.upper())
        print_detail(result)
        return

    # --- Full scan ---
    watchlist = [s.upper() for s in args.symbols] if args.symbols else DEFAULT_WATCHLIST
    console.print(f"\n[cyan]Scanning {len(watchlist)} symbols...[/cyan]")
    console.print("[dim]Institutional data: yfinance  |  Technicals: yfinance + ta  |  Sentiment: AV / yfinance fallback[/dim]\n")

    results = run_scan(watchlist)

    if not results:
        console.print("[yellow]No results above score threshold — showing all:[/yellow]")
        results = []
        for sym in watchlist:
            try:
                results.append(build_composite_score(sym))
            except Exception:
                pass
        results.sort(key=lambda x: x["composite_score"], reverse=True)

    print_summary_table(results)

    console.print(f"\n[bold]Top {args.top} — Detail View[/bold]")
    for r in results[:args.top]:
        print_detail(r)


if __name__ == "__main__":
    main()
