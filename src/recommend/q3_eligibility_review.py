"""Human eligibility audit of relevant Top-10 results, after intent/relevance consensus.

Rates use relevant (relevance >= 2 by default) returned services as denominator,
not ten padded slots. A query/condition rate is withheld until every such service
has human judgments for both age and region. Unknown is a valid judgment, never
an ineligibility error. These checks do not determine income/household eligibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import q3_eval as ev

AGE = 'age_judgment(일치/명백불일치/판단불가)'
REGION = 'region_judgment(일치/명백불일치/판단불가)'
VALUES = ('일치', '명백불일치', '판단불가')
REQUIRED = ['pair_id', 'servId', 'profile_id', AGE, REGION, 'reviewer', 'reviewed_at', 'judge_type']


def load_human_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    d = ev._human_rows(df, REQUIRED)
    if d.duplicated(['pair_id', 'servId']).any():
        raise ValueError('자격 합의본 pair_id·servId 중복')
    for col in (AGE, REGION):
        if (~d[col].isin(('',) + VALUES)).any():
            raise ValueError(f'자격 코드북 밖 값: {col}')
    if not d.profile_id.isin(ev.PROFILES).all():
        raise ValueError('정의되지 않은 자격 프로필')
    return d[d[AGE].isin(VALUES) & d[REGION].isin(VALUES)].copy()


def aggregate(queries, full, gold, intent, judgments, conditions, min_rel=ev.RELEVANCE_PRIMARY_MIN):
    """Return one row/query/condition; never treat missing or AI labels as human.

    mismatch = age OR region explicitly mismatches; unknown = no mismatch AND
    either axis unknown. They are disjoint. No mismatch does not certify eligibility.
    Incomplete human review yields counts/coverage but NaN rates (not zeros).
    """
    intent = ev.load_human_intent(intent)
    gold = ev.load_human_relevance(gold)
    judged = load_human_eligibility(judgments)
    usable = set(intent.loc[intent.intent_equivalence.ne('불일치'), 'pair_id'])
    gold = gold[gold.pair_id.isin(usable) & gold.relevance.ge(min_rel)]
    allowed = set(zip(gold.pair_id, gold.servId))
    if not set(zip(judged.pair_id, judged.servId)) <= allowed:
        raise ValueError('자격 판정이 확정 관련 서비스 집합 밖에 있음')
    profiles = queries.set_index('pair_id').profile_id.to_dict()
    if any(r.profile_id != profiles.get(r.pair_id) for r in judged.itertuples()):
        raise ValueError('평가셋과 자격 판정의 프로필 불일치')
    judged = judged.set_index(['pair_id', 'servId'])
    rows = []
    for q in queries.itertuples():
        if q.pair_id not in usable:
            continue
        relevant = set(gold.loc[gold.pair_id.eq(q.pair_id), 'servId'])
        for cond in conditions:
            ids = set(full[(q.query_id, cond.cond_id)].servId.head(ev.K)) & relevant
            keys = [(q.pair_id, sid) for sid in sorted(ids) if (q.pair_id, sid) in judged.index]
            d = judged.loc[keys] if keys else judged.iloc[:0]
            mismatch = d[AGE].eq('명백불일치') | d[REGION].eq('명백불일치')
            unknown = ~mismatch & (d[AGE].eq('판단불가') | d[REGION].eq('판단불가'))
            den, n = len(ids), len(d)
            complete = den > 0 and n == den
            rows.append(dict(query_id=q.query_id, pair_id=q.pair_id, query_type=q.query_type,
                             cond_id=cond.cond_id, n_relevant_top10=den, n_human_reviewed=n,
                             n_unreviewed=den-n, n_mismatch=int(mismatch.sum()), n_unknown=int(unknown.sum()),
                             n_no_mismatch=int((~mismatch & ~unknown).sum()),
                             n_age_unknown=int(d[AGE].eq('판단불가').sum()),
                             n_region_unknown=int(d[REGION].eq('판단불가').sum()),
                             judged_coverage=n/den if den else np.nan,
                             mismatch_rate=int(mismatch.sum())/den if complete else np.nan,
                             unknown_rate=int(unknown.sum())/den if complete else np.nan,
                             status='산출' if complete else '산출 보류(검수 미완료 또는 분모 0)',
                             denominator='관련 판정 >= %s인 실제 Top-10 반환 서비스 수' % min_rel,
                             income_household='확인 필요'))
    return pd.DataFrame(rows)
