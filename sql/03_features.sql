-- =====================================================================
-- 03_features.sql: fraud features built with SQL window functions.
--
-- Every card-history feature looks BACKWARDS only (preceding rows), so a
-- transaction is scored with exactly the information a real system would
-- have at that moment. No future data leaks into the features.
-- =====================================================================

DROP TABLE IF EXISTS fraud.features;

CREATE TABLE fraud.features AS
WITH category_baseline AS (
    -- "Normal" spend per category, learned from the TRAINING period only
    SELECT category,
           AVG(amt)         AS cat_avg_amt,
           STDDEV_SAMP(amt) AS cat_std_amt
    FROM fraud.transactions
    WHERE dataset = 'train'
    GROUP BY category
),
card_history AS (
    SELECT
        t.*,
        -- position in this card's history
        ROW_NUMBER() OVER card_w                                             AS card_txn_number,
        -- time since the card's previous transaction
        EXTRACT(EPOCH FROM t.trans_ts - LAG(t.trans_ts) OVER card_w)         AS secs_since_prev_txn,
        -- velocity: how many transactions / how much spend in the last 1h and 24h
        COUNT(*)   OVER (card_w RANGE BETWEEN INTERVAL '1 hour'   PRECEDING AND CURRENT ROW) AS txn_count_1h,
        COUNT(*)   OVER (card_w RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) AS txn_count_24h,
        SUM(t.amt) OVER (card_w RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) AS amt_sum_24h,
        -- the card's typical spend BEFORE this transaction
        AVG(t.amt) OVER (card_w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)          AS card_avg_amt_prior,
        -- first time this card has used this merchant?
        ROW_NUMBER() OVER (PARTITION BY t.card_id, t.merchant ORDER BY t.trans_ts, t.trans_num) = 1
                                                                             AS is_new_merchant
    FROM fraud.transactions t
    WINDOW card_w AS (PARTITION BY t.card_id ORDER BY t.trans_ts)
)
SELECT
    h.trans_num,
    h.trans_ts,
    h.card_id,
    h.dataset,
    h.is_fraud,
    h.category,
    h.amt,
    h.gender,
    h.state,
    EXTRACT(HOUR FROM h.trans_ts)::INT                                  AS hour,
    EXTRACT(ISODOW FROM h.trans_ts)::INT                                AS day_of_week,
    (EXTRACT(HOUR FROM h.trans_ts) >= 22 OR EXTRACT(HOUR FROM h.trans_ts) < 4)
                                                                        AS is_night,
    DATE_PART('year', AGE(h.trans_ts::date, h.dob))::INT                AS customer_age,
    h.city_pop,
    -- great-circle (haversine) distance between customer home and merchant, km
    ROUND((2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(h.merch_lat - h.lat) / 2), 2) +
        COS(RADIANS(h.lat)) * COS(RADIANS(h.merch_lat)) *
        POWER(SIN(RADIANS(h.merch_long - h.long) / 2), 2)
    )))::NUMERIC, 1)                                                    AS distance_km,
    h.card_txn_number,
    h.secs_since_prev_txn,
    h.txn_count_1h,
    h.txn_count_24h,
    h.amt_sum_24h,
    ROUND(h.card_avg_amt_prior, 2)                                      AS card_avg_amt_prior,
    ROUND(h.amt / NULLIF(h.card_avg_amt_prior, 0), 3)                   AS amt_to_card_avg,
    ROUND((h.amt - b.cat_avg_amt) / NULLIF(b.cat_std_amt, 0), 3)        AS amt_zscore_in_category,
    h.is_new_merchant
FROM card_history h
JOIN category_baseline b USING (category);

ALTER TABLE fraud.features ADD PRIMARY KEY (trans_num);
ANALYZE fraud.features;
