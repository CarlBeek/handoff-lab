#!/usr/bin/env python
"""Create a small, deterministic Python repo for harness runs (fresh copy per run).

The package has enough independent modules that "one sub-agent per module" is the natural plan, plus a few
planted bugs and TODOs so audit/fix tasks have real work to do.

    python scripts/make_target_repo.py <dir>
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

FILES = {
"README.md": """
# ledgerlite

A tiny personal-finance ledger: import CSV bank exports, store transactions as JSON, print reports.

Modules: `ledgerlite/parse.py` (CSV import), `ledgerlite/money.py` (amounts), `ledgerlite/storage.py`
(JSON persistence), `ledgerlite/report.py` (summaries), `ledgerlite/cli.py` (entry point).
Run tests with `python -m pytest -q`.
""",
"pyproject.toml": """
[project]
name = "ledgerlite"
version = "0.1.0"
requires-python = ">=3.10"
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
""",
"ledgerlite/__init__.py": "",
"ledgerlite/money.py": '''
from decimal import Decimal, ROUND_HALF_UP


def parse_amount(text):
    """Parse '1,234.56' or '-12.5' or '(12.50)' (accounting negative) into Decimal cents."""
    t = text.strip().replace(",", "")
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1]
    value = Decimal(t)
    return -value if neg else value


def to_cents(amount):
    # TODO: rounding mode is not applied consistently with format_amount
    return int(amount * 100)


def format_amount(amount, currency="USD"):
    q = Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    sign = "-" if q < 0 else ""
    return f"{sign}{currency} {abs(q):,.2f}"


def add(a, b):
    return Decimal(a) + Decimal(b)
''',
"ledgerlite/parse.py": '''
import csv
import datetime as dt
from .money import parse_amount

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"]


def parse_date(text):
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


def read_transactions(path):
    """Read a bank CSV with columns date, description, amount (header required)."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for line in reader:
            rows.append({
                "date": parse_date(line["date"]),
                "description": line["description"].strip(),
                "amount": parse_amount(line["amount"]),
                "category": line.get("category") or "uncategorised",
            })
    # BUG: duplicates are not detected; the same export imported twice double-counts
    return rows
''',
"ledgerlite/storage.py": '''
import json
import os
from decimal import Decimal


def _encode(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(type(obj))


def save(transactions, path):
    with open(path, "w") as fh:
        json.dump(transactions, fh, default=_encode, indent=2)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    # TODO: amounts come back as str and dates as str; callers expect Decimal and date
    return data
''',
"ledgerlite/report.py": '''
from collections import defaultdict
from decimal import Decimal
from .money import format_amount


def totals_by_category(transactions):
    totals = defaultdict(Decimal)
    for t in transactions:
        totals[t["category"]] += Decimal(t["amount"])
    return dict(totals)


def monthly_totals(transactions):
    out = defaultdict(Decimal)
    for t in transactions:
        d = t["date"]
        key = f"{d.year}-{d.month:02d}" if hasattr(d, "year") else str(d)[:7]
        out[key] += Decimal(t["amount"])
    return dict(sorted(out.items()))


def render(transactions):
    lines = ["Category totals:"]
    for cat, total in sorted(totals_by_category(transactions).items()):
        lines.append(f"  {cat:20s} {format_amount(total)}")
    lines.append("Monthly totals:")
    for month, total in monthly_totals(transactions).items():
        lines.append(f"  {month} {format_amount(total)}")
    # BUG: net total line is missing
    return "\\n".join(lines)
''',
"ledgerlite/cli.py": '''
import argparse
import sys
from . import parse, report, storage


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ledgerlite")
    ap.add_argument("csv", nargs="?", help="bank export to import")
    ap.add_argument("--store", default="ledger.json")
    args = ap.parse_args(argv)
    txns = storage.load(args.store)
    if args.csv:
        txns.extend(parse.read_transactions(args.csv))
        storage.save(txns, args.store)
    print(report.render(txns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
"tests/test_money.py": '''
from decimal import Decimal
from ledgerlite.money import parse_amount, format_amount


def test_parse_amount_accounting_negative():
    assert parse_amount("(12.50)") == Decimal("-12.50")


def test_format_amount():
    assert format_amount(Decimal("1234.5")) == "USD 1,234.50"
''',
"tests/test_parse.py": '''
import datetime as dt
from ledgerlite.parse import parse_date


def test_parse_date_formats():
    assert parse_date("2026-01-02") == dt.date(2026, 1, 2)
    assert parse_date("01/02/2026") == dt.date(2026, 1, 2)
''',
"sample_export.csv": """date,description,amount,category
2026-01-02,Coffee,"(4.50)",food
2026-01-03,Salary,"3,000.00",income
2026-02-01,Rent,"(1,200.00)",housing
2026-02-14,Coffee,"(4.50)",food
""",
}


def make(dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    for rel, content in FILES.items():
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(textwrap.dedent(content).lstrip("\n"))
    subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    subprocess.run(["git", "-c", "user.email=harness@example.com", "-c", "user.name=harness", "commit", "-q", "-m", "initial"], cwd=dest, check=True)
    return dest


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "target_repo"))
