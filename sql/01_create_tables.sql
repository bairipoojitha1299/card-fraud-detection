-- =====================================================================
-- 01_create_tables.sql: raw landing table (one row per card transaction)
-- =====================================================================
DROP SCHEMA IF EXISTS fraud CASCADE;
CREATE SCHEMA fraud;

-- Column order matches fraudTrain.csv / fraudTest.csv exactly.
CREATE TABLE fraud.raw_transactions (
    row_idx                BIGINT,
    trans_date_trans_time  TIMESTAMP,
    cc_num                 BIGINT,
    merchant               TEXT,
    category               TEXT,
    amt                    NUMERIC(12,2),
    first                  TEXT,
    last                   TEXT,
    gender                 CHAR(1),
    street                 TEXT,
    city                   TEXT,
    state                  CHAR(2),
    zip                    TEXT,
    lat                    DOUBLE PRECISION,
    long                   DOUBLE PRECISION,
    city_pop               INT,
    job                    TEXT,
    dob                    DATE,
    trans_num              TEXT,
    unix_time              BIGINT,
    merch_lat              DOUBLE PRECISION,
    merch_long             DOUBLE PRECISION,
    is_fraud               SMALLINT
);
