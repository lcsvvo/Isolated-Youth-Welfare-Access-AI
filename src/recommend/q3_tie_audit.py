"""Ranking-order diagnostic only; retain the published servId-ascending baseline."""
import pandas as pd


def audit(full: dict, k: int = 10) -> pd.DataFrame:
    """Compare ascending vs descending servId within identical (score, tf).

    This deliberately extreme alternative measures membership sensitivity, not
    relevance or ranking quality. Full rankings are required, not saved Top-50.
    """
    rows = []
    for (qid, cid), ranked in full.items():
        original = set(ranked.servId.head(k))
        reverse = ranked.sort_values(['score', 'tf', 'servId'], ascending=[False, False, False])
        alternative = set(reverse.servId.head(k))
        rows.append(dict(query_id=qid, cond_id=cid, n_top_baseline=len(original),
                         n_common=len(original & alternative), n_replaced=len(original-alternative),
                         membership_changed=original != alternative,
                         method='score/tf fixed; servId ascending vs descending (diagnostic only)'))
    return pd.DataFrame(rows)
