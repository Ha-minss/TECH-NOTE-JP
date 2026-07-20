# Data Audit

## ab_test.csv
- Rows: 404,770
- Unique users: 404,770
- Duplicate rows: 0
- Duplicate user_id values: 0
- Users assigned to multiple groups: 0
- Missing values: {'user_id': 0, 'revenue': 0, 'testgroup': 0}
- Negative revenue rows: 0
- Overall zero-revenue share: 99.08%
- Group counts: {'a': 202103, 'b': 202667}
- Payers by group: {'a': 1928, 'b': 1805}

CSV files use semicolon delimiters (`;`). The A/B analysis uses only `ab_test.csv` because no documented user-level join key ties this experiment to registration/auth logs.

## Cross-file Check
- A/B users present in `reg_data.uid`: 90.06%
- A/B user count: 404,770
- Matching registration users: 364,555
This overlap is descriptive only. It is not used to combine A/B and retention results because the prompt does not establish that both files use the same user identity namespace.
