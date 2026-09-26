-- =====================================================================
-- 02_load.sql: load both Kaggle files, then build a clean transactions
-- table. The source data is synthetic, but customer names and street
-- addresses are dropped anyway (data minimisation: the model doesn't need them).
-- Run from the project root:  psql -d fraud -f sql/02_load.sql
-- =====================================================================
\copy fraud.raw_transactions FROM 'data/raw/fraudTrain.csv' WITH (FORMAT csv, HEADER true)
CREATE TEMP TABLE train_ids AS SELECT trans_num FROM fraud.raw_transactions;
\copy fraud.raw_transactions FROM 'data/raw/fraudTest.csv' WITH (FORMAT csv, HEADER true)

CREATE TABLE fraud.transactions AS
SELECT
    r.trans_num,
    r.trans_date_trans_time                              AS trans_ts,
    r.cc_num                                             AS card_id,
    regexp_replace(r.merchant, '^fraud_', '')            AS merchant,  -- simulator prefix on every merchant
    r.category,
    r.amt,
    r.gender,
    r.city,
    r.state,
    r.lat, r.long,
    r.city_pop,
    r.dob,
    r.merch_lat, r.merch_long,
    r.is_fraud,
    CASE WHEN t.trans_num IS NOT NULL THEN 'train' ELSE 'test' END AS dataset
FROM fraud.raw_transactions r
LEFT JOIN train_ids t USING (trans_num);

ALTER TABLE fraud.transactions ADD PRIMARY KEY (trans_num);
CREATE INDEX ON fraud.transactions (card_id, trans_ts);
DROP TABLE fraud.raw_transactions;
ANALYZE fraud.transactions;

SELECT dataset,
       COUNT(*)                              AS transactions,
       SUM(is_fraud)                         AS fraud_cases,
       ROUND(100.0 * AVG(is_fraud), 3)       AS fraud_rate_pct,
       MIN(trans_ts)::date                   AS first_day,
       MAX(trans_ts)::date                   AS last_day
FROM fraud.transactions
GROUP BY dataset ORDER BY first_day;
