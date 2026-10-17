"""Tests for the public, time-split Supabase export."""
import base64
import json
from datetime import date

import pytest

from scripts import export_supabase
from scripts.export_supabase import assign_splits, fetch_table, require_public_key, split_for, write_export


def jwt(role):
    payload = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def sample():
    return {
        "artiklar": [{"id": 1, "skapad": "2026-06-01T10:00:00+00:00"}, {"id": 2, "skapad": "2026-09-01T00:00:00+00:00"}],
        "chatt_debatter": [{"id": 1, "skapad": "2026-08-31T23:59:59+00:00"}],
        "markets": [{"id": 10, "skapad": "2026-05-01T00:00:00+00:00", "status": "avgjord"},
                    {"id": 11, "skapad": "2026-09-02T00:00:00+00:00", "status": "öppen"}],
        # Bet 2 is placed before test_from but on a test market: it must stay in test.
        "agent_bets": [{"id": 1, "market_id": 10, "skapad": "2026-05-02T00:00:00+00:00"},
                       {"id": 2, "market_id": 11, "skapad": "2026-08-01T00:00:00+00:00"}],
    }


def test_anon_and_publishable_keys_accepted():
    require_public_key(jwt("anon"))
    require_public_key("sb_publishable_abc")


@pytest.mark.parametrize("key", [jwt("service_role"), "sb_secret_abc", "not-a-key"])
def test_privileged_or_unknown_keys_rejected(key):
    with pytest.raises(ValueError):
        require_public_key(key)


def test_split_uses_utc_date():
    test_from = date(2026, 9, 1)
    assert split_for("2026-09-01T00:00:00+00:00", test_from) == "test"
    assert split_for("2026-09-01T01:00:00+02:00", test_from) == "train"


def test_bets_follow_market_split():
    data = assign_splits(sample(), date(2026, 9, 1))
    assert [row["split"] for row in data["artiklar"]] == ["train", "test"]
    assert data["chatt_debatter"][0]["split"] == "train"
    assert [row["split"] for row in data["agent_bets"]] == ["train", "test"]


def test_bet_on_unknown_market_rejected():
    data = sample()
    data["agent_bets"].append({"id": 3, "market_id": 99, "skapad": "2026-05-02T00:00:00+00:00"})
    with pytest.raises(ValueError, match="unknown market"):
        assign_splits(data, date(2026, 9, 1))


def fake_server(pages, total):
    urls = []

    def fetch(url, key):
        urls.append(url)
        return (pages[len(urls) - 1] if len(urls) <= len(pages) else []), total
    return fetch, urls


def test_pagination_reads_all_rows(monkeypatch):
    monkeypatch.setattr(export_supabase, "PAGE_SIZE", 2)
    fetch, urls = fake_server([[{"id": 1}, {"id": 2}], [{"id": 3}]], 3)
    rows = fetch_table("https://x.supabase.co", "k", "markets", ["id"], fetch=fetch)
    assert [row["id"] for row in rows] == [1, 2, 3]
    assert "offset=0" in urls[0] and "offset=2" in urls[1]
    assert "order=id.asc" in urls[0]


def test_server_row_cap_below_page_size_still_reads_everything():
    # The server caps pages at 2 rows although PAGE_SIZE is 1000.
    fetch, _ = fake_server([[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}], [{"id": 5}]], 5)
    rows = fetch_table("https://x.supabase.co", "k", "markets", ["id"], fetch=fetch)
    assert len(rows) == 5


def test_missing_rows_rejected():
    fetch, _ = fake_server([[{"id": 1}]], 3)
    with pytest.raises(ValueError, match="server reports 3"):
        fetch_table("https://x.supabase.co", "k", "markets", ["id"], fetch=fetch)


def test_manifest_records_counts_and_hashes(tmp_path):
    data = assign_splits(sample(), date(2026, 9, 1))
    out = write_export(data, tmp_path / "export", date(2026, 9, 1), "https://x.supabase.co")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_from"] == "2026-09-01"
    assert manifest["files"]["agent_bets"]["test_rows"] == 1
    assert len((out / "artiklar.jsonl").read_text().splitlines()) == 2
    assert manifest["files"]["artiklar"]["rows_by_month"] == {"2026-06": 1, "2026-09": 1}
    assert manifest["files"]["markets"]["resolved_by_split"] == {"train": 1, "test": 0}


def test_empty_table_rejected(tmp_path):
    data = assign_splits(sample(), date(2026, 9, 1))
    data["markets"] = []
    with pytest.raises(ValueError, match="no rows"):
        write_export(data, tmp_path / "export", date(2026, 9, 1), "https://x.supabase.co")
