"""
Integration test for sql/03_features.sql and sql/04_rules.sql on a tiny,
hand-made card history where every expected value is known.
Needs a PostgreSQL database (PG* environment variables); skipped otherwise.
"""

import shutil

import pandas as pd
import pytest

from src.db import query, run_sql_file

pytestmark = pytest.mark.skipif(shutil.which("psql") is None, reason="psql not installed")

SETUP = """
DROP SCHEMA IF EXISTS fraud CASCADE;
CREATE SCHEMA fraud;
CREATE TABLE fraud.transactions (
    trans_num TEXT PRIMARY KEY, trans_ts TIMESTAMP, card_id BIGINT, merchant TEXT,
    category TEXT, amt NUMERIC(12,2), gender CHAR(1), city TEXT, state CHAR(2),
    lat DOUBLE PRECISION, long DOUBLE PRECISION, city_pop INT, dob DATE,
    merch_lat DOUBLE PRECISION, merch_long DOUBLE PRECISION, is_fraud SMALLINT, dataset TEXT);
INSERT INTO fraud.transactions VALUES
 ('t1','2020-01-01 10:00',1,'Shop A','grocery_pos',  10,'F','X','TX',30,-97,1000,'1990-06-01',30,-97,0,'train'),
 ('t2','2020-01-01 10:30',1,'Shop A','grocery_pos',  20,'F','X','TX',30,-97,1000,'1990-06-01',30,-97,0,'train'),
 ('t3','2020-01-01 12:00',1,'Shop B','grocery_pos', 300,'F','X','TX',30,-97,1000,'1990-06-01',31,-97,1,'train'),
 ('t4','2020-01-02 11:00',1,'Shop A','grocery_pos',   5,'F','X','TX',30,-97,1000,'1990-06-01',30,-97,0,'test'),
 ('t5','2020-01-01 23:30',2,'Shop C','grocery_pos', 400,'M','Y','NY',40,-74,5000,'1970-01-01',40,-74,1,'train');
"""


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    setup = tmp_path_factory.mktemp("sql") / "setup.sql"
    setup.write_text(SETUP)
    run_sql_file(setup)
    run_sql_file("sql/03_features.sql")
    run_sql_file("sql/04_rules.sql")
    f = query("SELECT * FROM fraud.features ORDER BY trans_num").set_index("trans_num")
    r = query("SELECT * FROM fraud.rule_flags ORDER BY trans_num").set_index("trans_num")
    return f, r


def test_card_sequence_and_time_gap(features):
    f, _ = features
    assert list(f.loc[["t1", "t2", "t3", "t4"], "card_txn_number"]) == [1, 2, 3, 4]
    assert pd.isna(f.loc["t1", "secs_since_prev_txn"])          # no history yet
    assert f.loc["t2", "secs_since_prev_txn"] == 1800


def test_velocity_windows_look_back_only(features):
    f, _ = features
    assert f.loc["t2", "txn_count_1h"] == 2       # t1 (10:00) + t2 (10:30)
    assert f.loc["t3", "txn_count_1h"] == 1       # 12:00 - nothing in the previous hour
    assert f.loc["t3", "txn_count_24h"] == 3
    assert f.loc["t4", "txn_count_24h"] == 2      # t3 (12:00 day before) + itself; t1/t2 are > 24h ago
    assert f.loc["t3", "amt_sum_24h"] == 330


def test_amount_vs_card_history(features):
    f, _ = features
    assert pd.isna(f.loc["t1", "card_avg_amt_prior"])
    assert f.loc["t3", "card_avg_amt_prior"] == 15      # avg of 10 and 20
    assert f.loc["t3", "amt_to_card_avg"] == 20


def test_new_merchant_distance_and_time_flags(features):
    f, _ = features
    assert f.loc["t1", "is_new_merchant"] == "t" and f.loc["t2", "is_new_merchant"] == "f"
    assert f.loc["t1", "distance_km"] == 0
    assert 110 < f.loc["t3", "distance_km"] < 112       # 1 degree of latitude ~ 111 km
    assert f.loc["t5", "is_night"] == "t" and f.loc["t1", "is_night"] == "f"
    assert f.loc["t1", "customer_age"] == 29


def test_rules(features):
    _, r = features
    assert r.loc["t5", "r1_night_high_amount"] == "t"    # $400 at 23:30
    assert r.loc["t3", "r1_night_high_amount"] == "f"    # $300 but at noon
