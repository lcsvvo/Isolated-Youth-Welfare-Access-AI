"""Read-only Q2 input audit; never rebuild matching or rewrite its metadata."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

MATCH_COLUMNS = ['servId', 'source', 'servNm', 'org', 'region_sido', 'region_sgg']
COMPARE_COLUMNS = ['servId', 'source', 'summary_text', 'overview_text', 'target_text',
                   'criteria_text', 'benefit_text', 'apply_text', 'apply_cat', 'contact_json', 'law_json']
REVIEW_COLUMNS = ['org_dept']


def audit(root: Path) -> tuple[pd.DataFrame, dict]:
    """Find the exact historical input by SHA-256 and compare values keyed by servId.

    Only an exact byte hash locates the historical Git blob. The recorded input
    hash itself is never edited; parsed field values are compared separately.
    """
    path = 'data/processed/services_master.csv'
    meta = json.loads((root / 'data/processed/q2_match.meta.json').read_text(encoding='utf-8'))
    expected = meta['inputs_sha256'][path]
    current = (root / path).read_bytes()
    refs = subprocess.check_output(['git', 'log', '--format=%H', '--', path], cwd=root, text=True).splitlines()
    old = None
    for ref in refs:
        blob = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=root)
        if hashlib.sha256(blob).hexdigest() == expected:
            old = blob
            break
    if old is None:
        raise ValueError('Q2 matching input hash not found in local Git history; do not rebuild matching')
    read = lambda b: pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False, encoding='utf-8-sig')
    a, b = read(old), read(current)
    if a.servId.duplicated().any() or b.servId.duplicated().any() or set(a.servId) != set(b.servId):
        raise ValueError('Historical/current master servId sets differ or contain duplicates')
    a, b = a.set_index('servId').sort_index(), b.set_index('servId').sort_index()
    rows = []
    for col in sorted(set(a) | set(b)):
        role = ('matching' if col in MATCH_COLUMNS else 'comparison' if col in COMPARE_COLUMNS
                else 'review_display' if col in REVIEW_COLUMNS else 'unused')
        n = int(a[col].ne(b[col]).sum()) if col in a and col in b else len(b)
        rows.append(dict(column=col, role=role, old_exists=col in a, current_exists=col in b, n_changed=n))
    table = pd.DataFrame(rows)
    info = dict(historical_commit=ref, matching_created_at=meta['created_at'],
                historical_sha256=expected, current_sha256=hashlib.sha256(current).hexdigest(),
                rows_old=len(a), rows_current=len(b), servId_symmetric_difference=0,
                old_columns=len(a.columns)+1, current_columns=len(b.columns)+1,
                matching_code_sha256=meta['code_sha256'],
                current_code_sha256=hashlib.sha256((root/'src/preprocess/q2_match.py').read_bytes()).hexdigest(),
                other_inputs_unchanged=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h
                                           for p,h in meta['inputs_sha256'].items() if p != path))
    info['used_columns_unchanged'] = not table.loc[table.role.ne('unused'), 'n_changed'].any()
    if not (info['used_columns_unchanged'] and info['other_inputs_unchanged']
            and info['matching_code_sha256'] == info['current_code_sha256']):
        raise ValueError('Q2 analysis inputs/code differ; investigation required before proceeding')
    return table, info
