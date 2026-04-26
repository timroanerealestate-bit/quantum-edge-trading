"""
Quick connectivity test — validates API key and pulls one symbol.
Run this first: python quick_test.py
"""
from rich.console import Console
console = Console()

def test_alpha_vantage():
    console.print("[cyan]Testing Alpha Vantage API...[/cyan]")
    try:
        from alpha_vantage_client import get_quote, get_rsi
        quote = get_quote("AAPL")
        if not quote:
            console.print("[red]ERROR: Empty quote response[/red]")
            return False
        price = quote.get("05. price", "unknown")
        console.print(f"[green]✓ Alpha Vantage OK — AAPL price: ${price}[/green]")

        console.print("[cyan]Testing RSI indicator...[/cyan]")
        rsi = get_rsi("AAPL")
        rsi_ts = rsi.get("Technical Analysis: RSI", {})
        latest_date = sorted(rsi_ts.keys(), reverse=True)[0]
        rsi_val = rsi_ts[latest_date]["RSI"]
        console.print(f"[green]✓ RSI OK — AAPL RSI(14): {rsi_val} as of {latest_date}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        return False


def test_sec_edgar():
    console.print("[cyan]Testing SEC EDGAR 13F access...[/cyan]")
    try:
        from institutional_tracker import get_submissions
        # Berkshire Hathaway
        subs = get_submissions("1067983")
        name = subs.get("name", "unknown")
        console.print(f"[green]✓ SEC EDGAR OK — fetched filings for: {name}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        return False


if __name__ == "__main__":
    console.rule("[bold]Training Bot — Connectivity Test[/bold]")
    av_ok = test_alpha_vantage()
    import time; time.sleep(13)  # respect rate limit
    sec_ok = test_sec_edgar()
    console.rule()
    if av_ok and sec_ok:
        console.print("[bold green]All systems go. Run: python main.py[/bold green]")
    else:
        console.print("[bold yellow]Some issues found — check errors above[/bold yellow]")
