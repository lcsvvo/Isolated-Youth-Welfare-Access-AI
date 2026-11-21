"""Q2 전처리: SSIS(복지로)와 보조금24의 같은 서비스 매칭, 항목별 명시 수준 비교.

입력
  - data/processed/services_master.csv  (SSIS 중앙 461 + 지자체 4,827, python -m src.preprocess.ssis_master)
  - data/raw/gov24/gov24_serviceList.csv, gov24_serviceDetail.csv  (보조금24 10,942, 원자료를 직접 읽는다)
출력
  - data/processed/q2_match.csv          SSIS 서비스마다 보조금24 최상위 후보 1건, 점수, 매칭 등급
  - data/processed/q2_field_compare.csv  매칭 쌍(A·B·C 등급)의 항목별 명시 수준과 정보 항목 명시 여부
  - data/processed/q2_match.meta.json    생성 시각, 입력 해시, 임계값
  - logs/validation/q2_match/            매칭 검수 시트, 전화번호 진단표

원칙
  - 공통 ID가 없다(서비스ID 겹침 0건, 복지로 ID가 들어 있는 보조금24 신청 URL 1건). 매칭은 서비스명과 지역·기관만 쓴다.
  - Q2에서 비교할 본문(지원대상·선정기준·지원내용·신청방법)은 매칭에 쓰지 않는다. 본문 유사도로 매칭하면
    본문이 비슷한 쌍만 남아 플랫폼 간 차이가 작게 측정된다.
  - 문의 전화번호는 매칭 점수에 넣지 않고 등급별 일치율 진단에만 쓴다.
  - 팀 결정(2026-11-05): 본문 임베딩도 매칭 점수에 쓰지 않는다. 비교 출처는 SSIS와 보조금24로 한정(온통청년 제외).
    정책 원문은 q2_codebook.SOURCES / MATCHING_POLICY.
  - 임계값(DICE_HIGH, MARGIN, DICE_REVIEW)은 잠정값이다. 검수 시트 판정으로 등급별 정밀도를 확인한 뒤 확정한다.

실행: python -m src.preprocess.q2_match
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, RAW, ROOT

MASTER = PROCESSED / "services_master.csv"
GOV_LIST = RAW / "gov24" / "gov24_serviceList.csv"
GOV_DETAIL = RAW / "gov24" / "gov24_serviceDetail.csv"
VALID = ROOT / "logs" / "validation" / "q2_match"

DICE_HIGH = 0.8      # B 등급 하한
MARGIN = 0.15        # B 등급: 1위와 2위 후보 점수 차 하한
DICE_REVIEW = 0.5    # C(검수) 등급 하한, 미만은 U(미매칭)
SEED = 20261105
REVIEW_N = {"A": 40, "B": 40, "C": 60, "U": 30}

# ------------------------------------------------------------------ 서비스명 정규화
PUNCT = re.compile(r"[\s()\[\]{}<>〈〉《》「」『』【】·ㆍ\-_,.:;/'\"‘’“”~!?+&]")
PREFIX = re.compile(r"^(자체사업|신규|시비|도비|구비|군비)")
SUFFIX = re.compile(r"(지원사업|사업|지원금지급|지원|지급|서비스|제공|운영)$")
SIDO_SUFFIX = re.compile(r"(특별자치도|특별자치시|특별시|광역시|도)$")


def core_name(name: str, sido: str = "", sgg: str = "") -> str:
    """비교용 서비스명: 기호·공백 제거, 앞의 지역명·예산 구분 제거, 끝의 '사업·지원·지급' 류 제거."""
    s = PUNCT.sub("", name)
    regions = [PUNCT.sub("", r) for r in (sido, sgg, SIDO_SUFFIX.sub("", sido), sgg[:-1] if len(sgg) > 2 else "") if r]
    for _ in range(2):
        for r in regions:
            if s.startswith(r) and len(s) > len(r) + 1:
                s = s[len(r):]
        s = PREFIX.sub("", s)
    for _ in range(3):
        t = SUFFIX.sub("", s)
        if len(t) >= 2:
            s = t
    return s


def bigrams(s: str) -> frozenset:
    return frozenset(s[i:i + 2] for i in range(len(s) - 1)) or frozenset([s])


def dice(a: frozenset, b: frozenset) -> float:
    return 2 * len(a & b) / (len(a) + len(b))


# ------------------------------------------------------------------ 입력
def read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def load_ssis() -> pd.DataFrame:
    m = read(MASTER)
    m["sgg_key"] = m["region_sgg"].replace("-", "")       # '-'는 시·도 본청 사업
    m["core"] = [core_name(n, a, b) for n, a, b in zip(m["servNm"], m["region_sido"], m["sgg_key"])]
    return m


def load_gov24() -> pd.DataFrame:
    lst = read(GOV_LIST)
    det = read(GOV_DETAIL)[["서비스ID", "서비스목적", "신청방법", "구비서류", "문의처", "법령", "자치법규", "행정규칙",
                            "온라인신청사이트URL"]]
    g = lst.rename(columns={"신청방법": "신청방법_분류"}).merge(det, on="서비스ID", how="left", validate="1:1")
    typ, org = g["소관기관유형"], g["소관기관명"]
    local = typ.isin(["시군구", "광역시도"])
    g["sido"] = np.where(local, org.str.split(" ").str[0], "")
    g["sgg_key"] = np.where(typ.eq("시군구"), org.str.split(" ").str[1].fillna(""), "")
    g["core"] = [core_name(n, a, b) for n, a, b in zip(g["서비스명"], g["sido"], g["sgg_key"])]
    return g


# ------------------------------------------------------------------ 매칭
def match(m: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """SSIS 서비스마다 같은 블록 안에서 서비스명 유사도 최상위 보조금24 후보를 고른다.

    블록: 지자체 = 같은 시·도 + 시·군·구(본청 사업은 시·군·구 빈칸), 중앙 = 보조금24 중앙행정기관 중 같은 부처명.
    블록이 없으면(교육청·행정구역 개편 지역, 보조금24에 없는 부처) 지자체는 같은 시·도 전체, 중앙은 보조금24 전체로 넓히고
    `block`에 표시한다. 넓힌 블록의 결과는 A·B로 올리지 않는다.
    """
    gb = g["core"].map(bigrams)
    local = g["소관기관유형"].isin(["시군구", "광역시도"])
    central = g["소관기관유형"].eq("중앙행정기관")
    by_sgg = {k: v.index.to_numpy() for k, v in g[local].groupby(["sido", "sgg_key"])}
    by_sido = {k: v.index.to_numpy() for k, v in g[local].groupby("sido")}
    by_org = {k: v.index.to_numpy() for k, v in g[central].groupby("소관기관명")}
    all_idx = g.index.to_numpy()

    rows = []
    for r in m.itertuples(index=False):
        if r.source == "local":
            idx, block = by_sgg.get((r.region_sido, r.sgg_key)), "sgg"
            if idx is None:
                idx, block = by_sido.get(r.region_sido, all_idx[:0]), "sido_fallback"
        else:
            idx, block = by_org.get(r.org), "org"
            if idx is None:
                idx, block = all_idx, "all_fallback"
        mb = bigrams(r.core)
        sc = np.array([dice(mb, gb.iat[i]) for i in idx]) if len(idx) else np.array([])
        if not len(sc):
            rows.append({"servId": r.servId, "block": "none", "n_candidates": 0, "gov_id": "", "score": 0.0,
                         "score_2nd": 0.0, "n_ties": 0})
            continue
        order = np.argsort(-sc, kind="stable")
        top = sc[order[0]]
        rows.append({"servId": r.servId, "block": block, "n_candidates": len(idx), "gov_id": g.at[idx[order[0]], "서비스ID"],
                     "score": round(float(top), 4), "score_2nd": round(float(sc[order[1]]), 4) if len(sc) > 1 else 0.0,
                     "n_ties": int((sc == top).sum())})
    res = pd.DataFrame(rows)

    strict = res["block"].isin(["sgg", "org"])
    unique = res["n_ties"].eq(1)
    tier = np.select(
        [strict & unique & res["score"].eq(1.0),
         strict & unique & res["score"].ge(DICE_HIGH) & (res["score"] - res["score_2nd"]).ge(MARGIN),
         res["score"].ge(DICE_REVIEW)],
        ["A", "B", "C"], default="U")
    res["tier"] = tier
    # 같은 보조금24 서비스를 둘 이상의 SSIS 서비스가 가리키는 경우 (SSIS 중복 등록 또는 오매칭)
    linked = res["tier"].ne("U") & res["gov_id"].ne("")
    res["gov_shared"] = linked & res["gov_id"].map(res.loc[linked, "gov_id"].value_counts()).fillna(0).gt(1)

    cols_m = ["servId", "source", "servNm", "org", "region_sido", "region_sgg", "core"]
    cols_g = ["서비스ID", "서비스명", "소관기관명", "소관기관유형", "core"]
    out = (res.merge(m[cols_m], on="servId", how="left")
              .merge(g[cols_g].rename(columns={"서비스ID": "gov_id", "서비스명": "gov_name", "소관기관명": "gov_org",
                                               "소관기관유형": "gov_org_type", "core": "gov_core"}), on="gov_id", how="left"))
    return out[["servId", "source", "servNm", "org", "region_sido", "region_sgg", "core", "block", "n_candidates",
                "gov_id", "gov_name", "gov_org", "gov_org_type", "gov_core", "score", "score_2nd", "n_ties", "tier",
                "gov_shared"]]


# ------------------------------------------------------------------ 항목 비교
# 01_eda 3절(s3_information_level)과 같은 수준 정의
PLACEHOLDER = re.compile(r"^\s*[-–·.]*\s*(해당\s*(사항)?\s*없음|없음|상동|별도\s*공고|공고\s*(참조|참고)|홈페이지\s*(참조|참고)"
                         r"|.*내용을\s*참고(해\s*주시기\s*바랍니다)?|문의|추후\s*공지)\s*[.]?\s*$")
LEVELS = ["필드 없음", "비어 있음", "형식 문구", "20자 미만", "20자 이상"]


def level(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    return pd.Series(np.select([s.eq(""), s.str.fullmatch(PLACEHOLDER), s.str.len().lt(20)], LEVELS[1:4], LEVELS[4]),
                     index=s.index)


def json_text(s: str, keys=("name", "content")) -> str:
    if not s:
        return ""
    try:
        items = json.loads(s)
    except json.JSONDecodeError:
        return s
    return "\n".join(" ".join(str(it.get(k) or "") for k in keys).strip() for it in items).strip()


# 필드 대응. SSIS에 없는 필드는 None ('필드 없음'으로 기록, 정보 누락과 구분)
FIELDS = {
    "summary": (None, "서비스목적요약"),        # SSIS 요약은 중앙·지자체 출처가 달라 아래에서 따로 만든다
    "target": ("target_text", "지원대상"),
    "criteria": ("criteria_text", "선정기준"),
    "benefit": ("benefit_text", "지원내용"),
    "apply": ("apply_text", "신청방법"),
    "contact": ("contact_text", "문의처"),
    "law": ("law_text", "law_text"),
    "apply_cat": ("apply_cat", "신청방법_분류"),  # 신청방법 분류값 (SSIS는 지자체만)
    "documents": (None, "구비서류"),
    "deadline": (None, "신청기한"),
}

# 정보 항목: 사용자가 신청 여부를 판단하는 데 필요한 내용이 있는지 (같은 정규식을 양쪽에 적용).
# 범위 text = 양쪽에 모두 있는 본문 필드만, all = 각 플랫폼이 보여 주는 필드 전체(분류값·구비서류·신청기한 포함).
# all의 차이에는 필드 구조 차이가 섞이므로, '정보 누락'은 text 범위로 먼저 읽는다.
SCOPES = {"text": ["summary", "target", "criteria", "benefit", "apply"],
          "all": ["summary", "target", "criteria", "benefit", "apply", "contact", "law", "apply_cat", "documents", "deadline"]}
ITEMS = {
    "age": re.compile(r"\d{1,2}\s?세|\d{1,2}\s?개월|연령|나이"),
    "income": re.compile(r"중위소득|소득인정액|소득\s?기준|수급자|차상위|저소득|건강보험료\s?(?:본인부담|납부)"),
    "residence": re.compile(r"거주|주민등록|전입"),
    # 신청 경로: 방법(방문·온라인·우편 등) 또는 신청 장소(주민센터·시군구청 등)가 적혀 있으면 명시로 본다
    "channel": re.compile(r"방문|온라인|인터넷|홈페이지|누리집|정부24|복지로|우편|팩스|이메일|전자우편|전화\s?(?:신청|접수)|모바일"
                          r"|주민센터|행정복지센터|읍\s?[·./]?\s?면\s?[·./]?\s?동|시\s?[·./]?\s?군\s?[·./]?\s?구|시청|군청|구청|도청|보건소"
                          r"|센터|사무소|접수처|담당\s?부서|직접\s?(?:신청|제출)"),
    "period": re.compile(r"상시|연중|수시|\d{1,2}\s?월|\d{4}\s?[.년]|기간|기한|마감|까지"),
    "documents": re.compile(r"구비\s?서류|제출\s?서류|신청서|증명서|등본|신분증|통장\s?사본|서식"),
    "contact": re.compile(r"(?:0\d{1,2}|1\d{3})[-)\s]?\d{3,4}-\d{4}|1\d{3}-\d{4}|☎"),
}


# v2 (2026-11-05, 검증 전 잠정): 항목마다 그 정보가 있어야 할 필드에서만 찾는다.
# v1은 본문 전체에서 찾아, 예를 들어 SSIS 매칭 쌍에서 신청 기간 v1 판정 958건 중 741건이 신청방법이 아닌
# 지원대상·지원내용 문장('2024년 이후 출생', '지원기간 12개월')에서만 걸렸다.
# text = 두 플랫폼 공통 본문 필드, all = 각 플랫폼 전용 필드(신청방법 분류·신청기한·구비서류·문의처)까지 포함.
ITEM_FIELDS_V2 = {
    "age": {"text": ["target", "criteria"]},
    "income": {"text": ["target", "criteria"]},
    "residence": {"text": ["target", "criteria"]},
    "channel": {"text": ["apply"], "all": ["apply_cat"]},
    "period": {"text": ["apply"], "all": ["deadline"]},
    "documents": {"text": ["apply"], "all": ["documents"]},
    "contact": {"text": ["apply"], "all": ["contact"]},
}
ITEMS_V2 = dict(ITEMS)
ITEMS_V2["period"] = re.compile(
    r"(?:신청|접수|모집|공모|청구)\s?(?:기간|기한|일정|마감|시기)|상시|연중|수시|선착순|예산\s?소진|마감"
    r"|(?:\d{1,2}\s?월|\d{1,2}\s?[./]\s?\d{1,2}|\d{4}\s?[.년]).{0,20}(?:까지|부터|~|신청|접수)")
# 전용 필드는 키워드 대신 내용 유무로 본다 (비어 있음·형식 문구가 아니면 명시)
APPLY_CAT_CHANNEL = re.compile(r"방문|온라인|인터넷|우편|팩스|전화|이메일|모바일")
DEDICATED = {"apply_cat": lambda s: s.fillna("").str.contains(APPLY_CAT_CHANNEL)}


def items_present_v2(f: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = {}
    for k, spec in ITEM_FIELDS_V2.items():
        cols = [c for c in spec["text"] if c in f]
        hit = f[cols].fillna("").agg("\n".join, axis=1).str.contains(ITEMS_V2[k]) if cols else pd.Series(False, index=f.index)
        if scope == "all":
            for c in spec.get("all", []):
                if c in f:
                    extra = DEDICATED[c](f[c]) if c in DEDICATED else level(f[c]).isin(["20자 미만", "20자 이상"])
                    hit = hit | extra
        out[f"item2_{scope}_{k}"] = hit
    return pd.DataFrame(out, index=f.index)


def ssis_fields(m: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"servId": m["servId"]})
    f["summary"] = np.where(m["source"].eq("central"), m["overview_text"], m["summary_text"])
    for k in ["target", "criteria", "benefit", "apply", "apply_cat"]:
        f[k] = m[FIELDS[k][0]]
    f["contact"] = m["contact_json"].map(json_text)
    f["law"] = m["law_json"].map(lambda s: json_text(s, ("name",)))
    return f


def gov_fields(g: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"gov_id": g["서비스ID"]})
    for k, (_, gc) in FIELDS.items():
        if gc != "law_text":
            f[k] = g[gc].fillna("")
    f["contact"] = (g["문의처"].fillna("") + "\n" + g["전화문의"].fillna("")).str.strip()
    f["law"] = g[["법령", "자치법규", "행정규칙"]].fillna("").agg("\n".join, axis=1).str.strip()
    return f


def items_present(f: pd.DataFrame, scope: str) -> pd.DataFrame:
    text = f[[c for c in SCOPES[scope] if c in f]].fillna("").agg("\n".join, axis=1)
    return pd.DataFrame({f"item_{scope}_{k}": text.str.contains(p) for k, p in ITEMS.items()}, index=f.index)


def compare(res: pd.DataFrame, m: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    pairs = res.loc[res["tier"].ne("U"), ["servId", "gov_id", "tier", "gov_shared", "source", "score"]]
    sf, gf = ssis_fields(m), gov_fields(g)
    s = pairs[["servId"]].merge(sf, on="servId", how="left")
    q = pairs[["gov_id"]].merge(gf, on="gov_id", how="left")
    out = pairs.reset_index(drop=True).copy()
    for k, (sc, _) in FIELDS.items():
        out[f"ssis_{k}"] = level(s[k]) if k in s else LEVELS[0]
        out[f"gov_{k}"] = level(q[k])
    out.loc[out["source"].eq("central"), "ssis_apply_cat"] = LEVELS[0]   # 중앙 SSIS에는 신청방법 분류 필드가 없다
    for scope in SCOPES:
        for side, f in [("ssis", s), ("gov", q)]:
            it = items_present(f, scope)
            for c in it:
                out[f"{side}_{c}"] = it[c].to_numpy()
        for k in ITEMS:
            a, b = out[f"ssis_item_{scope}_{k}"], out[f"gov_item_{scope}_{k}"]
            out[f"diff_item_{scope}_{k}"] = np.select([a & b, a & ~b, ~a & b], ["둘 다", "SSIS만", "보조금24만"], "둘 다 없음")
        for side, f in [("ssis", s), ("gov", q)]:
            it = items_present_v2(f, scope)
            for c in it:
                out[f"{side}_{c}"] = it[c].to_numpy()
        for k in ITEM_FIELDS_V2:
            a, b = out[f"ssis_item2_{scope}_{k}"], out[f"gov_item2_{scope}_{k}"]
            out[f"diff_item2_{scope}_{k}"] = np.select([a & b, a & ~b, ~a & b], ["둘 다", "SSIS만", "보조금24만"], "둘 다 없음")
    return out


# ------------------------------------------------------------------ 검증
PHONE = re.compile(r"(?:0\d{1,2}|1\d{3})[-)\s]?\d{3,4}-\d{4}|1\d{3}-\d{4}")


def phones(s: str) -> set:
    return {re.sub(r"\D", "", x)[-8:] for x in PHONE.findall(s or "")}


def phone_check(res: pd.DataFrame, m: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """등급별 문의 전화번호 일치율. 매칭 정밀도의 보조 진단이다(같은 서비스도 담당 부서 번호가 달라 일치율이 1이 되지 않는다)."""
    sp = m.set_index("servId")["contact_json"].map(phones)
    gp = (g["문의처"].fillna("") + " " + g["전화문의"].fillna("")).map(phones)
    gp.index = g["서비스ID"]
    a = res["servId"].map(sp)
    b = res["gov_id"].map(gp).map(lambda x: x if isinstance(x, set) else set())
    both = a.map(bool) & b.map(bool)
    hit = pd.Series([bool(x & y) for x, y in zip(a, b)], index=res.index)
    t = (pd.DataFrame({"tier": res["tier"], "both": both, "hit": hit & both})
         .groupby("tier").agg(n=("both", "size"), n_both_phone=("both", "sum"), n_phone_hit=("hit", "sum")).reset_index())
    t["phone_hit_rate"] = (t["n_phone_hit"] / t["n_both_phone"]).round(3)
    return t


def review_sheet(res: pd.DataFrame, m: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """등급별 층화 표본. 등급·점수는 시트에 넣지 않는다(q2_match.csv에서 servId로 다시 붙인다)."""
    pool = res[res["gov_id"].ne("")]
    parts = [grp.sample(min(len(grp), REVIEW_N[t]), random_state=SEED) for t, grp in pool.groupby("tier") if t in REVIEW_N]
    smp = pd.concat(parts)[["servId", "gov_id"]]
    ms = m.set_index("servId")
    gs = g.set_index("서비스ID")
    clip = lambda s: s.fillna("").str.replace(r"\s+", " ", regex=True).str[:200]
    sheet = pd.DataFrame({
        "servId": smp["servId"].to_numpy(), "gov_id": smp["gov_id"].to_numpy(),
        "ssis_name": ms.loc[smp["servId"], "servNm"].to_numpy(),
        "ssis_org": (ms.loc[smp["servId"], "org"] + " " + ms.loc[smp["servId"], "org_dept"]).str.strip().to_numpy(),
        "ssis_summary": clip(pd.Series(np.where(ms.loc[smp["servId"], "source"].eq("central"),
                                                ms.loc[smp["servId"], "overview_text"],
                                                ms.loc[smp["servId"], "summary_text"]))).to_numpy(),
        "ssis_target": clip(ms.loc[smp["servId"], "target_text"]).to_numpy(),
        "gov_name": gs.loc[smp["gov_id"], "서비스명"].to_numpy(),
        "gov_org": (gs.loc[smp["gov_id"], "소관기관명"] + " " + gs.loc[smp["gov_id"], "부서명"].fillna("")).str.strip().to_numpy(),
        "gov_summary": clip(gs.loc[smp["gov_id"], "서비스목적요약"]).to_numpy(),
        "gov_target": clip(gs.loc[smp["gov_id"], "지원대상"]).to_numpy(),
    })
    sheet["h_same"] = ""        # 같음 / 다름 / 불명
    sheet["h_note"] = ""
    sheet["labeler"] = ""
    return sheet.sample(frac=1, random_state=SEED).reset_index(drop=True)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    m, g = load_ssis(), load_gov24()
    res = match(m, g)
    cmp_ = compare(res, m, g)
    res.to_csv(PROCESSED / "q2_match.csv", index=False, encoding="utf-8-sig")
    cmp_.to_csv(PROCESSED / "q2_field_compare.csv", index=False, encoding="utf-8-sig")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"q2_match": len(res), "q2_field_compare": len(cmp_)},
        "params": {"DICE_HIGH": DICE_HIGH, "MARGIN": MARGIN, "DICE_REVIEW": DICE_REVIEW, "SEED": SEED},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [MASTER, GOV_LIST, GOV_DETAIL]},
        "code_sha256": sha256(ROOT / "src" / "preprocess" / "q2_match.py"),
    }
    (PROCESSED / "q2_match.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    VALID.mkdir(parents=True, exist_ok=True)
    pc = phone_check(res, m, g)
    pc.to_csv(VALID / "phone_check_by_tier.csv", index=False, encoding="utf-8-sig")
    sheet_path = VALID / "match_review_sheet.csv"
    sheet = review_sheet(res, m, g)
    if sheet_path.exists():
        prev = read(sheet_path)
        if prev[["h_same", "h_note", "labeler"]].ne("").any().any():
            raise ValueError(f"판정값이 있어 덮어쓰지 않습니다: {sheet_path}")
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")

    print("== 매칭 등급 (SSIS 서비스 단위)")
    print(pd.crosstab(res["source"], res["tier"], margins=True).to_string())
    print("== 블록:", res["block"].value_counts().to_dict())
    print("== 같은 보조금24 서비스를 공유하는 SSIS 서비스(A·B):", int((res["gov_shared"] & res["tier"].isin(["A", "B"])).sum()))
    print("== 전화번호 진단\n" + pc.to_string(index=False))
    print(f"== 검수 시트 {len(sheet)}건 → {sheet_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
