"""
Institutional holdings tracker using SEC EDGAR 13F filings.
13F filings are quarterly disclosures from institutions managing >$100M.
"""
import requests
import time
from config import SEC_EDGAR_BASE_URL, SEC_HEADERS


def _get(url: str) -> dict | list:
    resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_cik_for_ticker(ticker: str) -> str | None:
    """Resolve a stock ticker to a company CIK via SEC full-text search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&forms=10-K"
    try:
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={"company": ticker, "CIK": ticker, "type": "10-K", "dateb": "",
                    "owner": "include", "count": "5", "search_text": "", "action": "getcompany",
                    "output": "atom"},
            headers=SEC_HEADERS,
            timeout=15,
        )
        # Parse CIK from the response URL redirects or content
        if resp.history:
            for r in resp.history:
                if "/cgi-bin/browse-edgar?action=getcompany&CIK=" in r.url:
                    cik = r.url.split("CIK=")[1].split("&")[0]
                    return cik.lstrip("0")
        return None
    except Exception:
        return None


def get_company_facts(cik: str) -> dict:
    """Get all SEC filings facts for a company by CIK."""
    cik_padded = str(cik).zfill(10)
    url = f"{SEC_EDGAR_BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
    return _get(url)


def get_submissions(cik: str) -> dict:
    """Get filing submissions metadata for a company."""
    cik_padded = str(cik).zfill(10)
    url = f"{SEC_EDGAR_BASE_URL}/submissions/CIK{cik_padded}.json"
    return _get(url)


def search_13f_filers() -> list[dict]:
    """
    Return a curated list of major institutional filers (known CIKs).
    These are the top hedge funds and asset managers that file 13F quarterly.
    """
    return [
        {"name": "Berkshire Hathaway", "cik": "1067983"},
        {"name": "BlackRock",          "cik": "1364742"},
        {"name": "Vanguard Group",     "cik": "102909"},
        {"name": "State Street",       "cik": "93751"},
        {"name": "Fidelity",           "cik": "315066"},
        {"name": "Citadel Advisors",   "cik": "1423053"},
        {"name": "Point72",            "cik": "1603466"},
        {"name": "D.E. Shaw",          "cik": "1009207"},
        {"name": "Renaissance Tech",   "cik": "1037389"},
        {"name": "Two Sigma",          "cik": "1179392"},
        {"name": "Millennium Mgmt",    "cik": "1273931"},
        {"name": "Bridgewater",        "cik": "1350694"},
        {"name": "Tiger Global",       "cik": "1167483"},
        {"name": "Coatue Mgmt",        "cik": "1336528"},
        {"name": "Viking Global",      "cik": "1103804"},
    ]


def get_latest_13f_holdings(institution_cik: str) -> list[dict]:
    """
    Fetch the most recent 13F-HR filing for an institution and
    return holdings as a list of dicts with {cusip, ticker_like_name, value, shares}.
    """
    subs = get_submissions(institution_cik)
    filings = subs.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    # Find most recent 13F-HR
    latest_acc_dashes = None   # e.g. "0000950170-24-010364"
    latest_acc_nodash = None   # e.g. "0000950170240103640"  (used as folder name)
    for form, acc, date in zip(forms, accessions, dates):
        if form in ("13F-HR", "13F-HR/A"):
            latest_acc_dashes = acc                        # keep original dashes
            latest_acc_nodash = acc.replace("-", "")       # folder in EDGAR archive
            break

    if not latest_acc_nodash:
        return []

    # Correct EDGAR URL: folder = no-dashes, filename = with-dashes
    idx_url = (
        f"https://www.sec.gov/Archives/edgar/data/{institution_cik}/"
        f"{latest_acc_nodash}/{latest_acc_dashes}-index.json"
    )
    try:
        idx = _get(idx_url)
    except Exception:
        return []

    # Find the infotable XML document
    info_url = None
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "").lower()
        if "infotable" in name and name.endswith(".xml"):
            info_url = (
                f"https://www.sec.gov/Archives/edgar/data/{institution_cik}/"
                f"{latest_acc_nodash}/{item['name']}"
            )
            break

    if not info_url:
        return []

    try:
        resp = requests.get(info_url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        return _parse_infotable_xml(resp.text)
    except Exception:
        return []


def _parse_infotable_xml(xml_text: str) -> list[dict]:
    """Parse 13F infotable XML into a list of holdings."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml_text, "xml")
    holdings = []
    for entry in soup.find_all("infoTable"):
        try:
            name = entry.find("nameOfIssuer")
            cusip = entry.find("cusip")
            value = entry.find("value")
            ssh_prn_amt = entry.find("sshPrnamt")
            put_call = entry.find("putCall")

            holdings.append({
                "name": name.text.strip() if name else "",
                "cusip": cusip.text.strip() if cusip else "",
                "value_thousands": int(value.text.replace(",", "")) if value else 0,
                "shares": int(ssh_prn_amt.text.replace(",", "")) if ssh_prn_amt else 0,
                "put_call": put_call.text.strip() if put_call else "None",
            })
        except Exception:
            continue
    return sorted(holdings, key=lambda x: x["value_thousands"], reverse=True)


def build_institutional_heatmap(top_n: int = 5) -> dict[str, dict]:
    """
    Query the top N institutional filers and aggregate which positions
    appear most frequently (by number of institutions holding them).
    Returns {name -> {holders, total_value_M, institutions}}.
    """
    filers = search_13f_filers()[:top_n]
    aggregated: dict[str, dict] = {}

    for filer in filers:
        print(f"  Fetching 13F for {filer['name']}...")
        try:
            holdings = get_latest_13f_holdings(filer["cik"])
            for h in holdings[:100]:  # top 100 positions per institution
                key = h["name"].upper()
                if key not in aggregated:
                    aggregated[key] = {"holders": 0, "total_value_M": 0, "institutions": []}
                aggregated[key]["holders"] += 1
                aggregated[key]["total_value_M"] += h["value_thousands"] / 1000
                aggregated[key]["institutions"].append(filer["name"])
        except Exception as e:
            print(f"  Warning: could not fetch {filer['name']}: {e}")
        time.sleep(1)

    return dict(sorted(aggregated.items(), key=lambda x: x[1]["holders"], reverse=True))
