"""Trace request loss and contextual evidence without changing search intent."""
import re

_PHYSICAL = re.compile(r'(?:몸|신체|거동|다리|허리|무릎).{0,12}(?:불편|통증|아프|다쳐|다친)')
_MOTION = re.compile(r'집\s*밖|방\s*밖|외출|밖(?:에|으로|을)|나가')


def physical_context(session, channels_in):
    """Collect an explicit physical cause as metadata for a confirmed health area.

This never creates a health need or an eligibility/diagnosis assertion.
"""
    if session.get('need_changes') or '건강' not in {n['domain'] for n in session['confirmed_profile']['needs']}:
        return []
    result = []
    for c in session['confirmed_profile'].get('constraints', []):
        evidence = c.get('evidence') or ''
        if c.get('channel') != '방문':
            continue
        motion = _MOTION.search(evidence)
        prefix = evidence[:motion.start()].strip() if motion else ''
        if prefix and _PHYSICAL.search(prefix) and not channels_in(prefix) and prefix not in result:
            result.append(prefix)
    return result


def request_trace(session, source, bm25_tokens, dense, tokenize, normalize, excluded):
    """Diagnostic missing tokens are not relevance labels or new inferred needs."""
    bm, de = set(bm25_tokens), set(tokenize(dense))
    rows = []
    for n in session['confirmed_profile']['needs']:
        evidence = (n.get('evidence') or '').strip("'\" ")
        tokens = list(dict.fromkeys(tokenize(evidence)))
        rows.append({'domain': n['domain'], 'evidence': evidence or None,
            'activity_need_evidence_overlap': bool(evidence and normalize(evidence) == normalize(session['extracted'].get('activity_evidence') or '')),
            'original_evidence_found': bool(evidence and normalize(evidence) in normalize(session['original_text'])),
            'source_evidence_retained': bool(evidence and normalize(evidence) in normalize(source)),
            'missing_bm25_tokens': [t for t in tokens if t not in bm and t not in excluded],
            'policy_excluded_tokens': [t for t in tokens if t in excluded],
            'missing_dense_tokens': [t for t in tokens if t not in de],
            'status': 'no_text_evidence' if not evidence else 'user_need_changed' if session.get('need_changes') else 'traced'})
    return rows
