-- Reconcile the independently calculated expected cashback with the bank
-- expected_rewards table and the actual reward_ledger.
--
-- Required view:
-- h07_policy_calculated

CREATE OR REPLACE TEMP VIEW h07_reconciled_result AS
WITH flags AS (
    SELECT
        pc.*,
        CASE
            WHEN policy_expected_reward_amount > 0 AND reward_id IS NULL
                THEN TRUE
            ELSE FALSE
        END AS is_expected_reward_row_missing,
        CASE
            WHEN reward_id IS NOT NULL
             AND COALESCE(bank_expected_reward_amount, 0) > 0
             AND ledger_id IS NULL
                THEN TRUE
            ELSE FALSE
        END AS is_reward_ledger_row_missing,
        CASE
            WHEN reward_id IS NOT NULL
             AND policy_expected_reward_amount
                 != COALESCE(bank_expected_reward_amount, 0)
                THEN TRUE
            ELSE FALSE
        END AS is_policy_calculation_error,
        CASE
            WHEN reward_id IS NOT NULL
             AND ledger_id IS NOT NULL
             AND policy_expected_reward_amount
                 = COALESCE(bank_expected_reward_amount, 0)
             AND COALESCE(paid_reward_amount, 0)
                 < COALESCE(bank_expected_reward_amount, 0)
                THEN TRUE
            ELSE FALSE
        END AS is_payment_execution_error,
        CASE
            WHEN policy_expected_reward_amount > 0
                THEN GREATEST(
                    policy_expected_reward_amount - COALESCE(paid_reward_amount, 0),
                    0
                )
            ELSE 0
        END AS harm_amount
    FROM h07_policy_calculated pc
),
typed AS (
    SELECT
        f.*,
        NULLIF(
            CONCAT_WS(
                '|',
                CASE
                    WHEN is_expected_reward_row_missing
                        THEN 'EXPECTED_REWARD_ROW_MISSING'
                    ELSE NULL
                END,
                CASE
                    WHEN is_reward_ledger_row_missing
                        THEN 'REWARD_LEDGER_ROW_MISSING'
                    ELSE NULL
                END,
                CASE
                    WHEN is_policy_calculation_error
                        THEN 'POLICY_CALCULATION_ERROR'
                    ELSE NULL
                END,
                CASE
                    WHEN is_payment_execution_error
                        THEN 'PAYMENT_EXECUTION_ERROR'
                    ELSE NULL
                END,
                CASE
                    WHEN policy_eligibility_status = 'REVIEW_REQUIRED'
                        THEN 'REVIEW_REQUIRED'
                    ELSE NULL
                END
            ),
            ''
        ) AS detected_error_types,
        CASE
            WHEN is_expected_reward_row_missing
              OR is_reward_ledger_row_missing
              OR is_policy_calculation_error
              OR is_payment_execution_error
                THEN TRUE
            ELSE FALSE
        END AS is_affected,
        CASE
            WHEN policy_eligibility_status = 'NOT_ELIGIBLE'
             AND policy_expected_reward_amount = 0
                THEN TRUE
            ELSE FALSE
        END AS is_normal_exclusion,
        CASE
            WHEN policy_eligibility_status = 'REVIEW_REQUIRED'
                THEN TRUE
            ELSE FALSE
        END AS requires_human_review
    FROM flags f
)
SELECT
    rule_template_id,
    product_config_id,
    purchase_id,
    customer_id,
    card_id,
    product_id,
    purchase_date,
    purchase_month,
    amount,
    merchant_id,
    merchant_name,
    merchant_category,
    is_cashback_excluded,
    purchase_status,
    processing_route,
    reward_id,
    ledger_id,
    reward_batch_id,
    policy_eligibility_status,
    policy_eligibility_reason,
    policy_applied_rate,
    policy_gross_reward_amount,
    policy_expected_reward_amount,
    policy_expected_payment_date,
    policy_monthly_cap_remaining_before_transaction,
    policy_monthly_cap_remaining_after_transaction,
    bank_expected_reward_amount,
    bank_expected_payment_date,
    bank_reward_rate,
    bank_gross_reward_amount,
    bank_monthly_cap_amount,
    bank_eligibility_reason,
    paid_reward_amount,
    paid_at,
    payment_status,
    incident_id,
    is_expected_reward_row_missing,
    is_reward_ledger_row_missing,
    is_policy_calculation_error,
    is_payment_execution_error,
    detected_error_types,
    is_affected,
    is_normal_exclusion,
    requires_human_review,
    harm_amount
FROM typed
;
