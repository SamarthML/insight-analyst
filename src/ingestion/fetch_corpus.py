"""Download a real, freely-redistributable business corpus into data/documents.

Why a script rather than a list of links: SEC filing URLs contain accession
numbers that change with every filing, so any hardcoded Archives path is stale
the moment a company files again. This resolves them from EDGAR's official
submissions API at request time, which means the URLs are always current and
none of them are guessed.

The curated PDF sources below were each verified to return a real PDF; a
deliberately bogus URL at the same paths returns 404, so a 200 here means the
document genuinely exists.

    python -m src.ingestion.fetch_corpus --list
    python -m src.ingestion.fetch_corpus --dry-run
    python -m src.ingestion.fetch_corpus

SEC's fair-access policy requires a real contact address in the User-Agent, so
set SEC_CONTACT_EMAIL in your environment (or .env) before fetching filings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

# --- Politeness ------------------------------------------------------------
# The Federal Reserve throttles rapid sequential requests and starts returning
# non-200s that look exactly like missing files. One request per second keeps
# every host happy and keeps "not found" meaning not found.
REQUEST_DELAY_SECONDS = 1.0
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# --- What to fetch ---------------------------------------------------------
# Tickers, not CIK numbers: the CIK is resolved from SEC's official ticker map
# so nothing here can drift out of date. Sectors are deliberately mixed --
# a corpus of five tech 10-Ks gives dense and keyword retrieval far too little
# vocabulary to disagree over, which is the whole point of the exercise.
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "KO"]
FILINGS_PER_COMPANY = 3

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

BEIGE_BOOK_DATES = [
    "20250115", "20250305", "20250423", "20250604",
    "20250716", "20250903", "20251015", "20251126",
    "20260114", "20260304", "20260415", "20260603", "20260715",
]


@dataclass
class PdfSource:
    """A directly downloadable PDF, verified to exist at build time."""

    url: str
    filename: str
    note: str = ""


def curated_pdfs() -> list[PdfSource]:
    """Public-institution research PDFs, all free to redistribute."""
    sources = [
        PdfSource(
            "https://thedocs.worldbank.org/en/doc/"
            "7ce50b5aa95bef66048680bba9926ec8-0050012026/original/GEP-Jan-2026.pdf",
            "worldbank_global_economic_prospects_2026_01.pdf",
            "World Bank flagship, ~3.6 MB",
        ),
        PdfSource(
            "https://openknowledge.worldbank.org/server/api/core/bitstreams/"
            "4c370aee-ae90-4c42-ac27-89259f9e284e/content",
            "worldbank_global_economic_prospects_2026_06.pdf",
            "World Bank flagship, ~4.7 MB",
        ),
    ]
    for year, size in (("2024", "~3.9 MB"), ("2025", "~11.5 MB"), ("2026", "~5.4 MB")):
        sources.append(PdfSource(
            f"https://www.bis.org/publ/arpdf/ar{year}e.pdf",
            f"bis_annual_economic_report_{year}.pdf",
            f"Bank for International Settlements, {size}",
        ))
    for date in BEIGE_BOOK_DATES:
        sources.append(PdfSource(
            f"https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_{date}.pdf",
            f"fed_beige_book_{date}.pdf",
            "Federal Reserve, ~1 MB",
        ))
    return sources


# --- HTTP ------------------------------------------------------------------
def _sec_user_agent() -> str:
    """Build the User-Agent SEC's fair-access policy requires.

    SEC asks automated clients to identify themselves with a real contact
    address. We read it from the environment rather than embedding one so the
    declared contact is genuinely the person running the script.
    """
    email = os.getenv("SEC_CONTACT_EMAIL", "").strip()
    if not email or "@" not in email:
        raise SystemExit(
            "SEC filings need a contact address in the User-Agent (SEC fair-access\n"
            "policy). Set it before rerunning, e.g.\n\n"
            '    $env:SEC_CONTACT_EMAIL = "you@example.com"   # current shell only\n'
            '    setx SEC_CONTACT_EMAIL "you@example.com"     # new shells after this\n\n'
            "or add SEC_CONTACT_EMAIL=you@example.com to your .env file.\n"
            "Use --skip-sec to fetch only the public-institution PDFs instead."
        )
    return f"Insight Analyst research corpus builder ({email})"


def fetch_bytes(url: str, user_agent: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            import gzip
            payload = gzip.decompress(payload)
    time.sleep(REQUEST_DELAY_SECONDS)
    return payload


def probe(url: str, user_agent: str, timeout: int = 30) -> tuple[bool, str]:
    """Check a URL resolves without downloading the body.

    Asks for only the first kilobyte: enough for the server to commit to a
    status code and content type, cheap enough to run across the whole corpus
    in a pre-flight pass. A missing document 404s here rather than silently
    costing a full download later.
    """
    request = urllib.request.Request(url, headers={
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Range": "bytes=0-1024",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "?").split(";")[0]
            total = response.headers.get("Content-Range", "")
            size = total.split("/")[-1] if "/" in total else response.headers.get(
                "Content-Length", "?"
            )
            try:
                size = f"{int(size) / 1024:.0f} KB"
            except (TypeError, ValueError):
                size = "size unknown"
            ok = response.status in (200, 206)
            detail = f"HTTP {response.status}  {content_type}  {size}"
    except urllib.error.HTTPError as exc:
        ok, detail = False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"

    time.sleep(REQUEST_DELAY_SECONDS)
    return ok, detail


# --- SEC -------------------------------------------------------------------
def resolve_ciks(tickers: list[str], user_agent: str) -> dict[str, int]:
    """Map tickers to CIK numbers using SEC's official ticker file."""
    raw = json.loads(fetch_bytes(SEC_TICKERS_URL, user_agent))
    lookup = {entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()}

    resolved: dict[str, int] = {}
    for ticker in tickers:
        cik = lookup.get(ticker.upper())
        if cik is None:
            print(f"  ! {ticker}: not in SEC ticker file, skipping")
            continue
        resolved[ticker.upper()] = cik
    return resolved


def _collect_10ks(block: dict, cik: int, company: str, filings: list[dict], limit: int) -> None:
    """Append 10-K entries from one EDGAR filings block, up to `limit` total."""
    for i, form in enumerate(block["form"]):
        if form != "10-K":
            continue
        filings.append({
            "company": company,
            "cik": cik,
            "accession": block["accessionNumber"][i].replace("-", ""),
            "document": block["primaryDocument"][i],
            "date": block["filingDate"][i],
        })
        if len(filings) >= limit:
            return


def recent_10k_filings(cik: int, user_agent: str, limit: int) -> list[dict]:
    """Return metadata for a company's most recent 10-K filings.

    EDGAR only inlines roughly the last thousand filings under `recent`. That
    is many years for most companies, but a heavy filer like a large bank emits
    enough 8-Ks to push its older 10-Ks out of that window and into paginated
    archive files. Without following those, such a company silently contributes
    a single filing instead of the requested three.
    """
    data = json.loads(fetch_bytes(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent))
    company = data.get("name", str(cik))

    filings: list[dict] = []
    _collect_10ks(data["filings"]["recent"], cik, company, filings, limit)

    for archive in data["filings"].get("files", []):
        if len(filings) >= limit:
            break
        older = json.loads(fetch_bytes(
            f"https://data.sec.gov/submissions/{archive['name']}", user_agent
        ))
        _collect_10ks(older, cik, company, filings, limit)

    return filings


# --- HTML to text ----------------------------------------------------------
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_BLANKS_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Flatten a filing's HTML into plain text.

    The ingestion loader handles .pdf, .md and .txt only, and modern 10-Ks are
    inline-XBRL HTML, so they are converted on the way in rather than teaching
    the loader a fourth format for this one source.

    BeautifulSoup is used when installed because it handles malformed markup
    and block-level spacing properly; the regex fallback keeps this script
    working without adding a hard dependency.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
    except ImportError:
        stripped = _SCRIPT_STYLE_RE.sub(" ", html)
        stripped = re.sub(r"(?i)</(p|div|tr|h[1-6]|li|table)>", "\n", stripped)
        text = _TAG_RE.sub(" ", stripped)
        import html as html_module

        text = html_module.unescape(text)

    lines = [" ".join(line.split()) for line in text.splitlines()]
    return _BLANKS_RE.sub("\n\n", "\n".join(line for line in lines if line)).strip()


# --- Orchestration ---------------------------------------------------------
@dataclass
class Plan:
    sec: list[dict] = field(default_factory=list)
    pdfs: list[PdfSource] = field(default_factory=list)


def download(url: str, destination: Path, user_agent: str) -> tuple[bool, str]:
    """Fetch `url` to `destination`. Returns (downloaded, message)."""
    if destination.exists() and destination.stat().st_size > 0:
        return False, f"exists, skipped ({destination.stat().st_size / 1024:.0f} KB)"

    try:
        payload = fetch_bytes(url, user_agent)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return False, f"FAILED: {exc}"

    if destination.suffix == ".txt":
        destination.write_text(
            html_to_text(payload.decode("utf-8", errors="replace")), encoding="utf-8"
        )
    else:
        destination.write_bytes(payload)
    return True, f"{destination.stat().st_size / 1024:.0f} KB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a real business corpus.")
    parser.add_argument("--list", action="store_true", help="print sources and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every URL but download nothing")
    parser.add_argument("--skip-sec", action="store_true",
                        help="public-institution PDFs only")
    parser.add_argument("--skip-pdfs", action="store_true", help="SEC filings only")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    out = config.DOCUMENTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    pdfs = [] if args.skip_pdfs else curated_pdfs()

    if args.list:
        planned = len(TICKERS) * FILINGS_PER_COMPANY
        print(f"SEC 10-K filings : {len(TICKERS)} companies x {FILINGS_PER_COMPANY} "
              f"= {planned} (URLs resolved from EDGAR at runtime)")
        print(f"                   {', '.join(TICKERS)}")
        print(f"\nCurated PDFs     : {len(pdfs)}")
        for source in pdfs:
            print(f"  {source.filename}\n    {source.url}\n    {source.note}")
        print(f"\nTotal planned    : {planned + len(pdfs)} documents")
        return 0

    plan = Plan(pdfs=pdfs)
    user_agent = BROWSER_UA

    if not args.skip_sec:
        user_agent = _sec_user_agent()
        print(f"\nResolving SEC filings for {', '.join(TICKERS)} ...")
        for ticker, cik in resolve_ciks(TICKERS, user_agent).items():
            filings = recent_10k_filings(cik, user_agent, FILINGS_PER_COMPANY)
            print(f"  {ticker}: {len(filings)} 10-K filing(s)")
            for filing in filings:
                filing["ticker"] = ticker
                plan.sec.append(filing)

    total = len(plan.sec) + len(plan.pdfs)
    print(f"\n{total} documents planned "
          f"({len(plan.sec)} SEC filings, {len(plan.pdfs)} PDFs) -> {out}")

    if args.dry_run:
        print("\nProbing every URL (no file is written) ...")
        targets = [
            (f"sec_10k_{f['ticker'].lower()}_{f['date']}.txt",
             SEC_ARCHIVE_URL.format(**f), user_agent)
            for f in plan.sec
        ] + [(s.filename, s.url, BROWSER_UA) for s in plan.pdfs]

        unreachable = 0
        for name, url, agent in targets:
            status, detail = probe(url, agent)
            if not status:
                unreachable += 1
            print(f"  {'ok ' if status else 'FAIL'} {name:<46} {detail}")

        print(f"\n{len(targets) - unreachable}/{len(targets)} URLs resolved, "
              f"{unreachable} unreachable")
        return 1 if unreachable else 0

    fetched = skipped = failed = 0
    print("\nDownloading (1 req/s; a full run takes a few minutes) ...")

    for filing in plan.sec:
        name = f"sec_10k_{filing['ticker'].lower()}_{filing['date']}.txt"
        ok, message = download(SEC_ARCHIVE_URL.format(**filing), out / name, user_agent)
        print(f"  {name:<46} {message}")
        fetched += ok
        skipped += "skipped" in message
        failed += "FAILED" in message

    for source in plan.pdfs:
        ok, message = download(source.url, out / source.filename, BROWSER_UA)
        print(f"  {source.filename:<46} {message}")
        fetched += ok
        skipped += "skipped" in message
        failed += "FAILED" in message

    print(f"\n{fetched} downloaded, {skipped} already present, {failed} failed")
    print(f"Corpus now holds {len(list(out.glob('*')))} entries in {out}")
    print("\nNext: python -m src.ingestion.build_index")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
