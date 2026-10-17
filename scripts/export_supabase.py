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

# Columns that legitimately change after a row is created (read counts, market resolution).
# They are left out of the frozen test-set fingerprint so the set stays verifiable over time.
MUTABLE_COLUMNS = {"artiklar": {"lasningar"}, "markets": {"resolution_kalla", "status", "utfall"}}


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


def split_for(created: str, test_from: date, test_until: date | None = None) -> str:
    """Train before test_from, test through test_until (inclusive), excluded afterwards."""
    day = utc_date(created)
    if day < test_from:
        return "train"
    if test_until is None or day <= test_until:
        return "test"
    # Rows newer than a frozen test window are kept out of training too, so they can never leak.
    return "excluded"


def assign_splits(data: dict, test_from: date, test_until: date | None = None) -> dict:
    for table in ("artiklar", "chatt_debatter", "markets"):
        for row in data[table]:
            row["split"] = split_for(row["skapad"], test_from, test_until)
    # Bets follow their market, so a test market never leaks into training via its bets.
    market_split = {row["id"]: row["split"] for row in data["markets"]}
    for row in data["agent_bets"]:
        if row["market_id"] not in market_split:
            raise ValueError(f"Bet {row['id']} references unknown market {row['market_id']}")
        row["split"] = market_split[row["market_id"]]
        # Agents may still bet on an open market after the window closes; such late bets stay out.
        if test_until and utc_date(row["skapad"]) > test_until:
            row["split"] = "excluded"
    return data


def fingerprint(table: str, rows: list) -> str:
    """Order-independent hash of the immutable content of rows."""
    skip = MUTABLE_COLUMNS.get(table, set()) | {"split"}
    lines = sorted(
        (row["id"], json.dumps({k: v for k, v in row.items() if k not in skip}, ensure_ascii=False, sort_keys=True))
        for row in rows
    )
    return hashlib.sha256("".join(f"{row_id}\t{body}\n" for row_id, body in lines).encode("utf-8")).hexdigest()


def frozen_test_set(data: dict, test_from: date, test_until: date) -> dict:
    tables = {}
    for table, rows in data.items():
        test = [row for row in rows if row["split"] == "test"]
        tables[table] = {"ids": sorted(row["id"] for row in test), "sha256": fingerprint(table, test)}
    return {
        "schema_version": 1,
        "test_from": test_from.isoformat(),
        "test_until": test_until.isoformat(),
        "mutable_columns": {table: sorted(columns) for table, columns in MUTABLE_COLUMNS.items()},
        "tables": tables,
    }


def verify_frozen(data: dict, frozen: dict) -> None:
    """Refuse to export if the frozen test rows were added, removed or edited."""
    current = frozen_test_set(data, date.fromisoformat(frozen["test_from"]), date.fromisoformat(frozen["test_until"]))
    for table, expected in frozen["tables"].items():
        got = current["tables"][table]
        missing = sorted(set(expected["ids"]) - set(got["ids"]))
        added = sorted(set(got["ids"]) - set(expected["ids"]))
        if missing or added:
            raise ValueError(f"{table}: frozen test set changed (missing ids {missing[:10]}, new ids {added[:10]})")
        if got["sha256"] != expected["sha256"]:
            raise ValueError(f"{table}: content of frozen test rows changed")


def write_export(data: dict, output: Path, test_from: date, source: str, test_until: date | None = None) -> Path:
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
            "excluded_rows": sum(row["split"] == "excluded" for row in rows),
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
        "test_until": test_until.isoformat() if test_until else None,
        "split_rule": "train if skapad < test_from, test through test_until, excluded after (UTC dates); "
                      "agent_bets follow their market",
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-from", type=date.fromisoformat,
                        help="First UTC date of the held-out test split (default: 90 days ago)")
    parser.add_argument("--test-until", type=date.fromisoformat,
                        help="Last UTC date of the test split; newer rows are excluded (default: no limit)")
    parser.add_argument("--frozen", type=Path,
                        help="Frozen test-set file: take the window from it and verify the test rows are unchanged")
    parser.add_argument("--write-frozen", type=Path, help="Write the test set's ids and fingerprints here")
    args = parser.parse_args()
    frozen = None
    if args.frozen:
        if args.test_from or args.test_until or args.write_frozen:
            parser.error("--frozen fixes the test window; do not combine it with --test-from/--test-until/--write-frozen")
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        args.test_from = date.fromisoformat(frozen["test_from"])
        args.test_until = date.fromisoformat(frozen["test_until"])
    if args.write_frozen and not (args.test_from and args.test_until):
        parser.error("--write-frozen requires an explicit --test-from and --test-until")
    if args.test_until and args.test_until >= datetime.now(timezone.utc).date():
        parser.error("--test-until must be in the past, or rows created later that day would change the test set")
    base_url = os.environ["SUPABASE_URL"].rstrip("/")
    if not re.fullmatch(r"https://[a-z0-9]+\.supabase\.co", base_url):
        raise ValueError("SUPABASE_URL must be https://<project>.supabase.co")
    key = os.environ["SUPABASE_ANON_KEY"]
    require_public_key(key)
    test_from = args.test_from or (datetime.now(timezone.utc).date() - timedelta(days=90))
    data = {table: fetch_table(base_url, key, table, columns) for table, columns in TABLES.items()}
    assign_splits(data, test_from, args.test_until)
    if frozen:
        verify_frozen(data, frozen)
    print(write_export(data, args.output, test_from, base_url, args.test_until))
    if args.write_frozen:
        frozen = frozen_test_set(data, test_from, args.test_until)
        args.write_frozen.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Compact copy in the log, so the file can be committed without downloading the artifact.
        print("FROZEN_TEST_SET " + json.dumps(frozen, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
