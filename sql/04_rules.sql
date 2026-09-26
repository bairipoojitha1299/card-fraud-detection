-- =====================================================================
-- 04_rules.sql: explainable fraud rules, the kind a bank's fraud team
-- writes before (and alongside) any model. Each flag has a plain-English
-- reason an analyst can act on.
-- =====================================================================
DROP VIEW IF EXISTS fraud.rule_flags;

CREATE VIEW fraud.rule_flags AS
SELECT
    trans_num,
    dataset,
    is_fraud,
    amt,
    -- R1: large purchase in the middle of the night
    (is_night AND amt >= 300)                                          AS r1_night_high_amount,
    -- R2: spending far above this card's own history
    (card_txn_number > 5 AND amt_to_card_avg >= 5 AND amt >= 200)       AS r2_spike_vs_card_history,
    -- R3: a burst of spend within 24 hours
    (amt_sum_24h >= 1500 AND txn_count_24h >= 3)                        AS r3_24h_spend_burst,
    -- R4: unusually large amount for the category (3+ standard deviations)
    (amt_zscore_in_category >= 3)                                       AS r4_category_outlier
FROM fraud.features;
