-- Independently recalculate expected cashback from the product config.
--
-- Required view:
-- h07_base_frame
--
-- Required config-derived runtime tables:
-- runtime_policy(minimum_eligible_amount_krw, monthly_cap_amount_krw, payment_day)
-- rate_table(min_amount_inclusive, max_amount_exclusive, cashback_rate)
-- excluded_merchant_categories(merchant_category)
-- eligible_transaction_statuses(status)

CREATE OR REPLACE TEMP VIEW h07_policy_calculated AS
WITH policy AS (
    SELECT
        minimum_eligible_amount_krw,
        monthly_cap_amount_krw,
        payment_day
    FROM runtime_policy
    LIMIT 1
),
rated AS (
    SELECT
        b.*,
        r.cashback_rate AS policy_applied_rate,
        CASE
            WHEN b.merchant_id IS NOT NULL AND b.merchant_category IS NULL
                THEN TRUE
            ELSE FALSE
        END AS merchant_master_missing,
        CASE
            WHEN b.card_id IS NOT NULL AND b.card_contract_status IS NULL
                THEN TRUE
            ELSE FALSE
        END AS card_contract_missing,
        CASE
            WHEN b.merchant_category IN (
                SELECT merchant_category FROM excluded_merchant_categories
            )
                THEN TRUE
            ELSE FALSE
        END AS excluded_by_product_config,
        CASE
            WHEN b.is_cashback_excluded IS TRUE
             AND b.merchant_category NOT IN (
                SELECT merchant_category FROM excluded_merchant_categories
             )
                THEN TRUE
            WHEN COALESCE(b.is_cashback_excluded, FALSE) IS FALSE
             AND b.merchant_category IN (
                SELECT merchant_category FROM excluded_merchant_categories
             )
                THEN TRUE
            ELSE FALSE
        END AS merchant_exclusion_conflict
    FROM h07_base_frame b
    LEFT JOIN rate_table r
        ON b.amount >= r.min_amount_inclusive
       AND (
            r.max_amount_exclusive IS NULL
            OR b.amount < r.max_amount_exclusive
       )
),
eligibility AS (
    SELECT
        rated.*,
        policy.minimum_eligible_amount_krw,
        policy.monthly_cap_amount_krw,
        policy.payment_day,
        CASE
            WHEN merchant_master_missing THEN 'REVIEW_REQUIRED'
            WHEN card_contract_missing THEN 'REVIEW_REQUIRED'
            WHEN merchant_exclusion_conflict THEN 'REVIEW_REQUIRED'
            WHEN purchase_status NOT IN (
                SELECT status FROM eligible_transaction_statuses
            ) THEN 'NOT_ELIGIBLE'
            WHEN amount < policy.minimum_eligible_amount_krw THEN 'NOT_ELIGIBLE'
            WHEN excluded_by_product_config THEN 'NOT_ELIGIBLE'
            WHEN policy_applied_rate IS NULL THEN 'NOT_ELIGIBLE'
            ELSE 'ELIGIBLE'
        END AS policy_eligibility_status,
        CASE
            WHEN merchant_master_missing THEN 'MERCHANT_MASTER_MISSING'
            WHEN card_contract_missing THEN 'CARD_CONTRACT_MISSING'
            WHEN merchant_exclusion_conflict
                THEN 'MERCHANT_EXCLUSION_CONFLICT_REVIEW'
            WHEN purchase_status NOT IN (
                SELECT status FROM eligible_transaction_statuses
            ) THEN 'PURCHASE_STATUS_NOT_ELIGIBLE'
            WHEN amount < policy.minimum_eligible_amount_krw
                THEN 'BELOW_MINIMUM_TRANSACTION_AMOUNT'
            WHEN excluded_by_product_config THEN 'EXCLUDED_MERCHANT_CATEGORY'
            WHEN policy_applied_rate IS NULL THEN 'NO_RATE_TABLE_MATCH'
            ELSE 'ELIGIBLE'
        END AS policy_eligibility_reason,
        CASE
            WHEN purchase_status IN (
                    SELECT status FROM eligible_transaction_statuses
                 )
             AND amount >= policy.minimum_eligible_amount_krw
             AND excluded_by_product_config = FALSE
             AND policy_applied_rate IS NOT NULL
             AND merchant_master_missing = FALSE
             AND card_contract_missing = FALSE
             AND merchant_exclusion_conflict = FALSE
                THEN FLOOR(amount * policy_applied_rate)
            ELSE 0
        END AS policy_gross_reward_amount
    FROM rated
    CROSS JOIN policy
),
cap_running AS (
    SELECT
        e.*,
        COALESCE(
            SUM(policy_gross_reward_amount) OVER (
                PARTITION BY customer_id, product_id, purchase_month
                ORDER BY CAST(purchase_date AS DATE), purchase_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS policy_monthly_reward_before_transaction
    FROM eligibility e
),
cap_applied AS (
    SELECT
        cr.*,
        GREATEST(
            monthly_cap_amount_krw - policy_monthly_reward_before_transaction,
            0
        ) AS policy_monthly_cap_remaining_before_transaction,
        LEAST(
            policy_gross_reward_amount,
            GREATEST(
                monthly_cap_amount_krw - policy_monthly_reward_before_transaction,
                0
            )
        ) AS policy_expected_reward_amount
    FROM cap_running cr
),
payment_date_raw AS (
    SELECT
        ca.*,
        CAST(
            STRFTIME(
                STRPTIME(purchase_month || '-01', '%Y-%m-%d') + INTERVAL '1 month',
                '%Y-%m'
            )
            || '-'
            || LPAD(CAST(payment_day AS VARCHAR), 2, '0')
            AS DATE
        ) AS raw_policy_expected_payment_date
    FROM cap_applied ca
),
payment_date_adjusted AS (
    SELECT
        pdr.*,
        CASE
            WHEN STRFTIME(raw_policy_expected_payment_date, '%w') = '6'
                THEN raw_policy_expected_payment_date + INTERVAL '2 day'
            WHEN STRFTIME(raw_policy_expected_payment_date, '%w') = '0'
                THEN raw_policy_expected_payment_date + INTERVAL '1 day'
            ELSE raw_policy_expected_payment_date
        END AS policy_expected_payment_date
    FROM payment_date_raw pdr
)
SELECT
    *,
    GREATEST(
        monthly_cap_amount_krw
        - policy_monthly_reward_before_transaction
        - policy_expected_reward_amount,
        0
    ) AS policy_monthly_cap_remaining_after_transaction
FROM payment_date_adjusted
;
