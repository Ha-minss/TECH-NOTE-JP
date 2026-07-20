-- Build the H07 investigation base frame using DuckDB SQL.
--
-- Required registered tables:
-- customers, card_contracts, merchant_master, card_purchases,
-- expected_rewards, reward_ledger, reward_batch_logs, complaints
--
-- Required runtime table:
-- runtime_params(product_id, product_config_id, rule_template_id)
--
-- card_purchases is the base table. expected_rewards and reward_ledger must
-- remain LEFT JOINs because missing rows are error signals.

CREATE OR REPLACE TEMP VIEW h07_base_frame AS
WITH params AS (
    SELECT product_id, product_config_id, rule_template_id
    FROM runtime_params
    LIMIT 1
),
filtered_purchases AS (
    SELECT
        p.*,
        rp.product_config_id,
        rp.rule_template_id
    FROM card_purchases p
    INNER JOIN params rp
        ON p.product_id = rp.product_id
)
SELECT
    p.rule_template_id,
    p.product_config_id,
    p.purchase_id,
    p.card_id,
    p.customer_id,
    p.product_id,
    p.purchase_date,
    p.purchase_month,
    p.amount,
    p.merchant_id,
    p.status AS purchase_status,
    p.cancelled_at,
    p.processing_route,
    p.source_system AS purchase_source_system,

    c.customer_name_masked,
    c.customer_segment,
    c.age_band,
    c.home_region,
    c.is_synthetic,

    cc.product_name AS contract_product_name,
    cc.card_opened_at,
    cc.status AS card_contract_status,
    cc.settlement_account_id,

    m.merchant_name,
    m.merchant_category,
    m.is_cashback_excluded,
    m.merchant_region,

    e.reward_id,
    e.expected_payment_date AS bank_expected_payment_date,
    e.reward_batch_id,
    e.purchase_amount AS bank_purchase_amount,
    e.merchant_category AS bank_merchant_category,
    e.purchase_status AS bank_purchase_status,
    e.reward_rate AS bank_reward_rate,
    e.gross_reward_amount AS bank_gross_reward_amount,
    e.expected_reward_amount AS bank_expected_reward_amount,
    e.monthly_cap_amount AS bank_monthly_cap_amount,
    e.eligibility_reason AS bank_eligibility_reason,
    e.included_in_monthly_cap AS bank_included_in_monthly_cap,
    e.processing_route AS bank_expected_processing_route,
    e.source_system AS bank_expected_source_system,

    l.ledger_id,
    l.paid_reward_amount,
    l.paid_at,
    l.payment_status,
    l.incident_id,
    l.source_system AS ledger_source_system
FROM filtered_purchases p
LEFT JOIN customers c
    ON p.customer_id = c.customer_id
LEFT JOIN card_contracts cc
    ON p.card_id = cc.card_id
   AND p.customer_id = cc.customer_id
   AND p.product_id = cc.product_id
LEFT JOIN merchant_master m
    ON p.merchant_id = m.merchant_id
LEFT JOIN expected_rewards e
    ON p.purchase_id = e.purchase_id
   AND p.customer_id = e.customer_id
   AND p.card_id = e.card_id
   AND p.product_id = e.product_id
LEFT JOIN reward_ledger l
    ON e.reward_id = l.reward_id
   AND e.purchase_id = l.purchase_id
   AND e.customer_id = l.customer_id
   AND e.card_id = l.card_id
   AND e.product_id = l.product_id
   AND e.reward_batch_id = l.reward_batch_id
;
