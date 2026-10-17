"""Tests for the public, time-split Supabase export."""
import base64
import json
from datetime import date

import pytest

from scripts import export_supabase
from scripts.export_supabase import (assign_splits, fetch_table, frozen_test_set, require_public_key, split_for,
                                     verify_frozen, write_export)


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
    assert manifest["files"]["markets"]["resolved_by_month"] == {"2026-05": 1}
    # Bet 2 was placed in August but belongs to the September market, like its split.
    assert manifest["files"]["agent_bets"]["rows_by_month"] == {"2026-05": 1, "2026-09": 1}


def test_empty_table_rejected(tmp_path):
    data = assign_splits(sample(), date(2026, 9, 1))
    data["markets"] = []
    with pytest.raises(ValueError, match="no rows"):
        write_export(data, tmp_path / "export", date(2026, 9, 1), "https://x.supabase.co")


def window():
    return date(2026, 9, 1), date(2026, 9, 30)


def test_rows_after_test_until_are_excluded():
    assert split_for("2026-09-30T23:59:59+00:00", *window()) == "test"
    assert split_for("2026-10-01T00:00:00+00:00", *window()) == "excluded"


def test_late_bet_on_test_market_is_excluded():
    data = sample()
    data["agent_bets"].append({"id": 3, "market_id": 11, "skapad": "2026-10-05T00:00:00+00:00"})
    assign_splits(data, *window())
    assert [row["split"] for row in data["agent_bets"]] == ["train", "test", "excluded"]


def test_excluded_rows_reported_in_manifest(tmp_path):
    data = sample()
    data["artiklar"].append({"id": 3, "skapad": "2026-10-02T00:00:00+00:00"})
    out = write_export(assign_splits(data, *window()), tmp_path / "export", window()[0], "https://x.supabase.co",
                       window()[1])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_until"] == "2026-09-30"
    assert "bets placed after test_until are excluded" in manifest["split_rule"]
    assert manifest["files"]["artiklar"]["excluded_rows"] == 1


def frozen_sample():
    data = sample()
    data["markets"][1].update({"titel": "Fråga?", "utfall": None})
    data["artiklar"][1].update({"rubrik": "Rubrik", "lasningar": 5})
    return assign_splits(data, *window())


def test_frozen_set_lists_test_ids():
    frozen = frozen_test_set(frozen_sample(), *window())
    assert frozen["test_until"] == "2026-09-30"
    assert frozen["tables"]["artiklar"]["ids"] == [2]
    assert frozen["tables"]["agent_bets"]["ids"] == [2]


def test_frozen_set_tolerates_resolution_and_reads():
    frozen = frozen_test_set(frozen_sample(), *window())
    data = frozen_sample()
    data["markets"][1].update({"status": "avgjord", "utfall": "ja"})
    data["artiklar"][1]["lasningar"] = 50
    data["artiklar"].append({"id": 4, "skapad": "2026-10-10T00:00:00+00:00"})
    verify_frozen(assign_splits(data, *window()), frozen)


def test_frozen_set_detects_edits_additions_and_deletions():
    frozen = frozen_test_set(frozen_sample(), *window())
    edited = frozen_sample()
    edited["artiklar"][1]["rubrik"] = "Ändrad"
    with pytest.raises(ValueError, match="content of frozen test rows changed"):
        verify_frozen(edited, frozen)
    added = frozen_sample()
    added["chatt_debatter"].append({"id": 9, "skapad": "2026-09-15T00:00:00+00:00", "split": "test"})
    with pytest.raises(ValueError, match="new ids \\[9\\]"):
        verify_frozen(added, frozen)
    deleted = frozen_sample()
    deleted["artiklar"].pop()
    with pytest.raises(ValueError, match="missing ids \\[2\\]"):
        verify_frozen(deleted, frozen)


def test_inverted_window_rejected(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["export_supabase.py", "--output", "out",
                                     "--test-from", "2026-09-01", "--test-until", "2026-08-01"])
    with pytest.raises(SystemExit):
        export_supabase.main()
    assert "must not be earlier than --test-from" in capsys.readouterr().err
