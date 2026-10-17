"""Export public Debatt-AI data from Supabase REST into a time-split, versioned dataset."""
import argparse
import base64
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PAGE_SIZE = 1000

# Explicit column allowlists: only data already public on debatt-ai.se.
# PostgREST rejects unknown columns, so a schema change fails the export.
TABLES = {
    "artiklar": ["id", "rubrik", "forfattare", "artikel", "kategori", "motivering", "arg", "ori", "rel", "tro",
                 "taggar", "kalla", "lasningar", "parent_id", "nyhetskalla", "skapad"],
    "chatt_debatter": ["id", "amne", "agenter", "inlagg", "summering", "scores", "kalla_url", "kalla_titel", "skapad"],
    "markets": ["id", "titel", "beskrivning", "deadline", "resolution_kalla", "utfall", "status", "kategori", "skapad"],
    "agent_bets": ["id", "market_id", "agent", "sannolikhet", "motivering", "skapad"],
}


def require_public_key(key: str) -> None:
    """Refuse privileged keys; the export must only see what RLS exposes publicly."""
    if key.startswith("sb_secret_"):
        raise ValueError("Refusing a Supabase secret key; use the public anon/publishable key")
    if key.startswith("sb_publishable_"):
        return
    parts = key.split(".")
    if len(parts) != 3:
        raise ValueError("Unrecognized Supabase key format")
    payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    if payload.get("role") != "anon":
        raise ValueError(f"Refusing Supabase key with role {payload.get('role')!r}; use the anon key")


def fetch_json(url: str, key: str):
    """Return (rows, total row count) for one PostgREST page."""
    request = urllib.request.Request(url, headers={
        "apikey": key, "Authorization": f"Bearer {key}", "Prefer": "count=exact",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        return json.loads(response.read().decode("utf-8")), int(total)


def fetch_table(base_url: str, key: str, table: str, columns: list, fetch=fetch_json) -> list:
    rows = []
    while True:
        query = urllib.parse.urlencode({
            "select": ",".join(columns), "order": "id.asc", "limit": PAGE_SIZE, "offset": len(rows),
        })
        page, total = fetch(f"{base_url}/rest/v1/{table}?{query}", key)
        rows.extend(page)
        if not page or len(rows) >= total:
            break
    # A server-side row cap below PAGE_SIZE must not silently truncate the export.
    if len(rows) != total:
        raise ValueError(f"{table}: fetched {len(rows)} rows but the server reports {total}")
    return rows


def utc_date(created: str) -> date:
    moment = datetime.fromisoformat(created)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.date()


def month_of(created: str) -> str:
    return utc_date(created).strftime("%Y-%m")


def monthly(months) -> dict:
    return dict(sorted(Counter(months).items()))


def split_for(created: str, test_from: date) -> str:
    return "test" if utc_date(created) >= test_from else "train"


def assign_splits(data: dict, test_from: date) -> dict:
    for table in ("artiklar", "chatt_debatter", "markets"):
        for row in data[table]:
            row["split"] = split_for(row["skapad"], test_from)
    # Bets follow their market, so a test market never leaks into training via its bets.
    market_split = {row["id"]: row["split"] for row in data["markets"]}
    for row in data["agent_bets"]:
        if row["market_id"] not in market_split:
            raise ValueError(f"Bet {row['id']} references unknown market {row['market_id']}")
        row["split"] = market_split[row["market_id"]]
    return data


def write_export(data: dict, output: Path, test_from: date, source: str) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    market_month = {row["id"]: month_of(row["skapad"]) for row in data["markets"]}
    for table, rows in data.items():
        if not rows:
            raise ValueError(f"Table {table} returned no rows; refusing an incomplete export")
        body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        (output / f"{table}.jsonl").write_text(body, encoding="utf-8")
        files[table] = {
            "rows": len(rows),
            "train_rows": sum(row["split"] == "train" for row in rows),
            "test_rows": sum(row["split"] == "test" for row in rows),
            # Per-month counts make it possible to choose test_from without downloading the data.
            # Bets are counted under their market's month, matching how assign_splits splits them.
            "rows_by_month": monthly(market_month[row["market_id"]] if table == "agent_bets" else month_of(row["skapad"])
                                     for row in rows),
            "columns": TABLES[table],
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
    files["markets"]["resolved_by_split"] = {
        split: sum(row["split"] == split and row["status"] == "avgjord" for row in data["markets"])
        for split in ("train", "test")
    }
    files["markets"]["resolved_by_month"] = monthly(
        month_of(row["skapad"]) for row in data["markets"] if row["status"] == "avgjord")
    manifest = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "test_from": test_from.isoformat(),
        "split_rule": "test if skapad >= test_from (UTC date); agent_bets follow their market",
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-from", type=date.fromisoformat,
                        help="First UTC date of the held-out test split (default: 90 days ago)")
    args = parser.parse_args()
    base_url = os.environ["SUPABASE_URL"].rstrip("/")
    if not re.fullmatch(r"https://[a-z0-9]+\.supabase\.co", base_url):
        raise ValueError("SUPABASE_URL must be https://<project>.supabase.co")
    key = os.environ["SUPABASE_ANON_KEY"]
    require_public_key(key)
    test_from = args.test_from or (datetime.now(timezone.utc).date() - timedelta(days=90))
    data = {table: fetch_table(base_url, key, table, columns) for table, columns in TABLES.items()}
    print(write_export(assign_splits(data, test_from), args.output, test_from, base_url))


if __name__ == "__main__":
    main()
