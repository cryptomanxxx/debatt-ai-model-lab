"""Export Debatt-AI data from Supabase REST into a time-split, versioned dataset. Read-only: GET requests only."""
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
                 "taggar", "kalla", "lasningar", "parent_id", "nyhetskalla", "filmrecension", "skapad"],
    "chatt_debatter": ["id", "amne", "agenter", "inlagg", "summering", "scores", "kalla_url", "kalla_titel", "skapad"],
    "markets": ["id", "titel", "beskrivning", "deadline", "resolution_kalla", "utfall", "status", "kategori", "skapad"],
    "agent_bets": ["id", "market_id", "agent", "sannolikhet", "motivering", "skapad"],
}

# Hidden from the anon key by RLS, so read with the service role key. Only AI-agent rows (kalla = 'ai') are
# requested and kept: human submissions can be unpublished texts with real names and never leave the database.
# Older rows without kalla are skipped rather than guessed from agent names.
PRIVATE_TABLES = {
    "inlamningar": {
        "columns": ["id", "rubrik", "forfattare", "artikel", "kategori", "motivering", "beslut", "arg", "ori", "rel",
                    "tro", "kalla", "status", "filmrecension", "skapad"],
        "filters": {"kalla": "eq.ai"},
    },
}
COLUMNS = {**TABLES, **{table: spec["columns"] for table, spec in PRIVATE_TABLES.items()}}

# Boolean columns that decide which evaluation task a row belongs to. The ids of test rows where they are true are
# frozen too, so a corrected flag cannot silently move a row between tasks under the same frozen-set version.
FLAG_COLUMNS = {"artiklar": ["filmrecension"], "inlamningar": ["filmrecension"]}

# Columns that legitimately change after a row is created (read counts, market resolution).
# They are left out of the frozen test-set fingerprint so the set stays verifiable over time.
MUTABLE_COLUMNS = {"artiklar": {"lasningar"}, "markets": {"resolution_kalla", "status", "utfall"},
                   # A submission moves from inkorg to publicerad when it is published.
                   "inlamningar": {"status"}}


def key_role(key: str) -> str:
    if key.startswith("sb_secret_"):
        return "service_role"
    if key.startswith("sb_publishable_"):
        return "anon"
    parts = key.split(".")
    if len(parts) != 3:
        raise ValueError("Unrecognized Supabase key format")
    return json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))).get("role")


def require_public_key(key: str) -> None:
    """Refuse privileged keys for the public tables; they must only see what RLS exposes publicly."""
    role = key_role(key)
    if role != "anon":
        raise ValueError(f"Refusing Supabase key with role {role!r} for public tables; use the anon key")


def require_service_key(key: str) -> None:
    """The service role key is only for PRIVATE_TABLES; it must really be one, or RLS returns nothing."""
    role = key_role(key)
    if role != "service_role":
        raise ValueError(f"SUPABASE_SERVICE_ROLE_KEY has role {role!r}; expected the service role key")


def read_only_request(url: str, key: str) -> urllib.request.Request:
    """Build the only kind of request this script sends: a GET. The service role key could write, so nothing else."""
    headers = {"apikey": key, "Prefer": "count=exact"}
    # Opaque sb_publishable_/sb_secret_ keys go in apikey only; Supabase rejects them as bearer JWTs.
    if not key.startswith("sb_"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, method="GET", headers=headers)
    if request.get_method() != "GET" or request.data is not None:
        raise RuntimeError("Refusing a non-GET request to Supabase")
    return request


def fetch_json(url: str, key: str):
    """Return (rows, total row count) for one PostgREST page."""
    request = read_only_request(url, key)
    with urllib.request.urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        return json.loads(response.read().decode("utf-8")), int(total)


def fetch_table(base_url: str, key: str, table: str, columns: list, fetch=fetch_json, filters=None) -> list:
    rows = []
    while True:
        query = urllib.parse.urlencode({
            **(filters or {}), "select": ",".join(columns), "order": "id.asc", "limit": PAGE_SIZE, "offset": len(rows),
        })
        page, total = fetch(f"{base_url}/rest/v1/{table}?{query}", key)
        rows.extend(page)
        if not page or len(rows) >= total:
            break
    # A server-side row cap below PAGE_SIZE must not silently truncate the export.
    if len(rows) != total:
        raise ValueError(f"{table}: fetched {len(rows)} rows but the server reports {total}")
    return rows


def fetch_private_table(base_url: str, key: str, table: str, fetch=fetch_json) -> list:
    spec = PRIVATE_TABLES[table]
    rows = fetch_table(base_url, key, table, spec["columns"], fetch, spec["filters"])
    # Defence in depth: never write a non-AI submission, even if the server-side filter were ignored.
    if any(row.get("kalla") != "ai" for row in rows):
        raise ValueError(f"{table}: received rows with kalla other than 'ai'; refusing to export them")
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
    for table in ("artiklar", "chatt_debatter", "markets", *PRIVATE_TABLES):
        for row in data.get(table, ()):
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


def fingerprint(table: str, rows: list, mutable_columns: dict, columns=None) -> str:
    """Order-independent hash of the immutable content of rows, limited to columns when given."""
    skip = set(mutable_columns.get(table, ())) | {"split"}
    keep = (lambda k: k in columns and k not in skip) if columns is not None else (lambda k: k not in skip)
    lines = sorted(
        (row["id"], json.dumps({k: v for k, v in row.items() if keep(k)}, ensure_ascii=False, sort_keys=True))
        for row in rows
    )
    return hashlib.sha256("".join(f"{row_id}\t{body}\n" for row_id, body in lines).encode("utf-8")).hexdigest()


def flag_ids(rows: list, columns: list) -> dict:
    return {column: sorted(row["id"] for row in rows if row.get(column) is True) for column in columns}


def frozen_test_set(data: dict, test_from: date, test_until: date, mutable_columns: dict = MUTABLE_COLUMNS,
                    columns: dict | None = None) -> dict:
    """Ids and fingerprints of the test rows; columns (default: the export's) records what the hash covers."""
    tables = {}
    for table, rows in data.items():
        test = [row for row in rows if row["split"] == "test"]
        cols = (columns or {}).get(table, COLUMNS.get(table))
        tables[table] = {"ids": sorted(row["id"] for row in test), "columns": cols,
                         "sha256": fingerprint(table, test, mutable_columns, cols)}
        if table in FLAG_COLUMNS:
            tables[table]["flags"] = flag_ids(test, FLAG_COLUMNS[table])
    return {
        "schema_version": 1,
        "test_from": test_from.isoformat(),
        "test_until": test_until.isoformat(),
        "mutable_columns": {table: sorted(columns) for table, columns in mutable_columns.items()},
        "tables": tables,
    }


def verify_frozen(data: dict, frozen: dict) -> None:
    """Refuse to export if the frozen test rows were added, removed or edited."""
    # Hash with the exclusions and columns recorded at freeze time, so later edits to MUTABLE_COLUMNS or a newly
    # exported column cannot break v1, while edits to the frozen columns are still caught.
    current = frozen_test_set(data, date.fromisoformat(frozen["test_from"]), date.fromisoformat(frozen["test_until"]),
                              frozen["mutable_columns"],
                              {table: spec.get("columns") for table, spec in frozen["tables"].items()})
    for table, expected in frozen["tables"].items():
        if table not in current["tables"]:
            hint = " (set SUPABASE_SERVICE_ROLE_KEY)" if table in PRIVATE_TABLES else ""
            raise ValueError(f"{table}: part of the frozen test set but not exported{hint}")
        got = current["tables"][table]
        missing = sorted(set(expected["ids"]) - set(got["ids"]))
        added = sorted(set(got["ids"]) - set(expected["ids"]))
        if missing or added:
            raise ValueError(f"{table}: frozen test set changed (missing ids {missing[:10]}, new ids {added[:10]})")
        if got["sha256"] != expected["sha256"]:
            raise ValueError(f"{table}: content of frozen test rows changed")
        test = [row for row in data[table] if row["split"] == "test"]
        for column, ids in expected.get("flags", {}).items():
            if flag_ids(test, [column])[column] != ids:
                raise ValueError(f"{table}: frozen {column} flags changed")


def frozen_flag_additions(data: dict, frozen: dict) -> dict | None:
    """Flag ids of frozen tables whose frozen file does not record them yet."""
    additions = {}
    for table, spec in frozen["tables"].items():
        missing = [c for c in FLAG_COLUMNS.get(table, []) if c not in spec.get("flags", {})]
        if missing and table in data:
            additions[table] = flag_ids([row for row in data[table] if row["split"] == "test"], missing)
    return additions or None


def frozen_additions(data: dict, frozen: dict) -> dict | None:
    """Fingerprints, for the frozen window, of exported tables the frozen file does not cover yet."""
    new = {table: rows for table, rows in data.items() if table not in frozen["tables"]}
    if not new:
        return None
    return frozen_test_set(new, date.fromisoformat(frozen["test_from"]), date.fromisoformat(frozen["test_until"]),
                           {table: MUTABLE_COLUMNS.get(table, set()) for table in new})


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
            "columns": COLUMNS[table],
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
    files["markets"]["resolved_by_split"] = {
        split: sum(row["split"] == split and row["status"] == "avgjord" for row in data["markets"])
        for split in ("train", "test")
    }
    files["markets"]["resolved_by_month"] = monthly(
        month_of(row["skapad"]) for row in data["markets"] if row["status"] == "avgjord")
    for table in ("artiklar", "inlamningar"):
        if table in files:
            # Film reviews are a different genre; counted so they can be kept apart from debate writing.
            files[table]["filmrecension_by_split"] = {
                split: sum(row["split"] == split and row.get("filmrecension") is True for row in data[table])
                for split in ("train", "test")
            }
    if "inlamningar" in files:
        # Accept/reject balance per split, needed to judge whether the editor test can measure rejections.
        files["inlamningar"]["beslut_by_split"] = {
            split: monthly(row["beslut"] for row in data["inlamningar"] if row["split"] == split)
            for split in ("train", "test")
        }
    manifest = {
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "test_from": test_from.isoformat(),
        "test_until": test_until.isoformat() if test_until else None,
        "split_rule": (
            "train if skapad < test_from, test through test_until, excluded after (UTC dates); "
            "agent_bets follow their market, except bets placed after test_until are excluded"
            if test_until else
            "train if skapad < test_from, test from test_from onward (UTC dates); agent_bets follow their market"
        ),
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
    args.test_from = args.test_from or (datetime.now(timezone.utc).date() - timedelta(days=90))
    if args.test_until and args.test_until < args.test_from:
        parser.error("--test-until must not be earlier than --test-from")
    base_url = os.environ["SUPABASE_URL"].rstrip("/")
    if not re.fullmatch(r"https://[a-z0-9]+\.supabase\.co", base_url):
        raise ValueError("SUPABASE_URL must be https://<project>.supabase.co")
    key = os.environ["SUPABASE_ANON_KEY"]
    require_public_key(key)
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    # Fail before any download when the frozen set needs a table only the service role key can read.
    needed = sorted(set(PRIVATE_TABLES) & set(frozen["tables"])) if frozen else []
    if needed and not service_key:
        raise SystemExit(f"{', '.join(needed)} is part of {args.frozen}; set SUPABASE_SERVICE_ROLE_KEY")
    test_from = args.test_from
    data = {table: fetch_table(base_url, key, table, columns) for table, columns in TABLES.items()}
    # Without it (and without a frozen set that needs it) the export holds public tables only.
    if service_key:
        require_service_key(service_key)
        for table in PRIVATE_TABLES:
            data[table] = fetch_private_table(base_url, service_key, table)
    else:
        print("SUPABASE_SERVICE_ROLE_KEY not set; skipping " + ", ".join(PRIVATE_TABLES))
    assign_splits(data, test_from, args.test_until)
    if frozen:
        verify_frozen(data, frozen)
    print(write_export(data, args.output, test_from, base_url, args.test_until))
    additions = frozen and frozen_additions(data, frozen)
    if additions:
        # Not yet frozen: printed so the tables can be added to the frozen file for the same window.
        print("FROZEN_TEST_SET_ADDITIONS " + json.dumps(additions, ensure_ascii=False, separators=(",", ":")))
    flag_additions = frozen and frozen_flag_additions(data, frozen)
    if flag_additions:
        print("FROZEN_FLAG_ADDITIONS " + json.dumps(flag_additions, ensure_ascii=False, separators=(",", ":")))
    if args.write_frozen:
        frozen = frozen_test_set(data, test_from, args.test_until)
        args.write_frozen.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Compact copy in the log, so the file can be committed without downloading the artifact.
        print("FROZEN_TEST_SET " + json.dumps(frozen, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
