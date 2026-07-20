from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import statsmodels.formula.api as smf


def run_twfe_ols(
    df: pd.DataFrame,
    y: str,
    x_terms: List[str],
    controls: List[str],
    cluster_col: str = "Country",
) -> Tuple[object, str]:
    fe = "C(Country) + C(Year)"
    rhs = " + ".join([*x_terms, *controls, fe])
    formula = f"{y} ~ {rhs}"
    res = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
    )
    return res, formula


def tidy_term(res, term: str) -> Dict[str, float]:
    return {
        "b": float(res.params.get(term, float("nan"))),
        "se": float(res.bse.get(term, float("nan"))),
        "p": float(res.pvalues.get(term, float("nan"))),
        "N": int(res.nobs),
        "R2": float(res.rsquared),
    }
