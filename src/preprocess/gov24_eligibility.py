"""보조금24 자격·신청 정보 결측 보완 (보조금24 내부 자료만 사용).

입력: data/processed/gov24_merged.csv  (python -m src.preprocess.gov24_youth 결과)
출력: data/processed/gov24_eligibility.csv  (서비스ID + 아래 파생 컬럼)

원칙
  - 구조화 필드(JA 코드) → 텍스트 규칙(지원대상·선정기준) → '명시 없음' 순서로 채운다.
  - 모델로 추정해서 채우지 않는다. 각 값의 출처를 *_src 컬럼에 남긴다.
  - 사회보장정보원 자료와의 교차 보완, LLM 추출은 이 스크립트 범위 밖이다 (팀 합의 후 진행).

검증 기록 (2026-10-31). 판정 상세와 표본 ID: logs/validation/gov24_eligibility_review_2026-10-31.json
  - 신청 채널: 범주 ∪ 본문 합집합. 이전 회차와 겹치지 않는 표본으로 회차별 측정
      범주 15/15, 범주+본문 23/25, 본문 19/20, 기관 경유·신청 불필요 신설분 16/20(이후 문의 번호 규칙 수정, 재측정 안 함)
  - 소득: 텍스트 % 방향(이하=상한, 초과=하한) 구분. JA와 비교해 일치 / 구간 경계 / 불일치로 분류
      불일치 46건 전수 판정: JA 과소 33, 서로 다른 조건 8, JA 과대 3, 추출 오류 2
      JA가 0~50% 구간만 표시한 서비스 중 본문 상한이 있는 116건의 62%는 본문 상한이 50% 초과 → 소득 필터에 JA 단독 사용 금지
  - 연령: 텍스트 연령은 여러 대상군·자녀 나이가 섞여 판정에 쓰지 않음(elig_age_text_differs는 참고용)
      떨어진 범위 4건 전수 판정(JA 오류 3, 복수 대상군 1). 자동 검출 안 되는 JA 오류 추가 표본 4/20, 이전 표본 2/25
  - 판정은 1인이 수행했다. 교차 판정은 아직 하지 않았다.

실행: python -m src.preprocess.gov24_eligibility
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, RAW

NONE = "명시 없음"
RULE_VERSION = "2026-11-05.1"
INCOME_BANDS = [  # (JA 컬럼 접두, 상한 %)
    ("JA0201", 50), ("JA0202", 75), ("JA0203", 100), ("JA0204", 200), ("JA0205", 999),
]
STATUS = {"JA0317": "초등학생", "JA0318": "중학생", "JA0319": "고등학생", "JA0320": "대학생/대학원생",
          "JA0326": "근로자/직장인", "JA0327": "구직자/실업자", "JA0328": "장애인",
          "JA0329": "국가보훈대상자", "JA0330": "질병/질환자"}
HOUSEHOLD = {"JA0401": "다문화가족", "JA0402": "북한이탈주민", "JA0403": "한부모/조손",
             "JA0404": "1인가구", "JA0411": "다자녀", "JA0412": "무주택", "JA0413": "신규전입",
             "JA0414": "확대가족"}

BAND_LOWER = {50: 0, 75: 51, 100: 76, 200: 101, 999: 201}
RE_INCOME_SPAN = re.compile(r"중위소득[^\n|○ㅇ\-]{0,45}")
RE_PCT_DIR = re.compile(r"(\d{2,3})\s?%\s?(이하|미만|초과|이상)?")
NEXT_BAND_UPPER = {50: 75, 75: 100, 100: 200, 200: 999}
RE_LOW_INCOME = re.compile(r"기초생활|수급자|차상위|저소득")
RE_AGE_RANGE = re.compile(r"만?\s?(\d{1,2})\s?세\s?(?:이상)?\s?[~～-]\s?만?\s?(\d{1,2})\s?세")
RE_AGE_MIN = re.compile(r"만\s?(\d{1,2})\s?세\s?이상")
RE_AGE_MAX = re.compile(r"만\s?(\d{1,2})\s?세\s?(?:이하|미만)")
RE_AGE_DISJOINT = re.compile(r"(\d{1,2})\s?세\s?(?:이하|미만)\s?(?:및|또는|,|과|와)\s?만?\s?(\d{1,2})\s?세\s?이상")
RE_RESIDENCE = re.compile(r"(?:주민등록|거주)")


def col(df: pd.DataFrame, code: str) -> pd.Series:
    name = next(c for c in df.columns if c.startswith(code + "_"))
    return df[name].eq("Y")


def income(df: pd.DataFrame, text: pd.Series) -> pd.DataFrame:
    flags = pd.concat([col(df, c).rename(c) for c, _ in INCOME_BANDS], axis=1)
    n = flags.sum(axis=1)
    upper = pd.Series(np.nan, index=df.index)
    for c, u in INCOME_BANDS:
        upper = upper.where(~flags[c], u)          # 가장 높은 Y 구간의 상한
    ja = np.select([n == 5, n.between(1, 4)], ["무관", "제한"], default="")
    # '중위소득' 뒤 구간에서 %를 방향과 함께 읽는다. 이하·미만(또는 방향 없음) = 상한, 초과·이상 = 하한.
    # 상한이 여러 개면 최댓값(가장 넓은 자격)을 쓴다. 예: 청년독립가구 60% 이하, 원가구 100% 이하 → 100
    def parse(s: str) -> tuple[float, float]:
        ups, lows = [], []
        for span in RE_INCOME_SPAN.findall(s):
            for num, d in RE_PCT_DIR.findall(span):
                (lows if d in ("초과", "이상") else ups).append(float(num))
        return (max(ups) if ups else np.nan, min(lows) if lows else np.nan)
    parsed = text.apply(parse)
    spans = text.apply(lambda s: " / ".join(" ".join(x.split()) for x in RE_INCOME_SPAN.findall(s))[:300])
    pct = parsed.str[0]
    pct_low = parsed.str[1]
    low = text.str.contains(RE_LOW_INCOME)
    txt = np.select([pct.notna(), pct_low.notna(), low], ["제한", "하한 있음", "제한(저소득)"], default="")

    out = pd.DataFrame(index=df.index)
    out["elig_income"] = np.where(ja != "", ja, np.where(txt != "", txt, NONE))
    out["elig_income_src"] = np.where(ja != "", "JA", np.where(txt != "", "텍스트", NONE))
    out["elig_income_max_pct"] = np.where(ja == "제한", upper, pct)
    out["elig_income_text_pct"] = pct          # 텍스트 상한 %
    out["elig_income_text_low_pct"] = pct_low  # 텍스트 하한 % (초과·이상)
    out["elig_income_evidence"] = spans        # 근거: 본문의 '중위소득 …' 구절
    out["elig_income_ja_upper"] = np.where(ja == "제한", upper, np.nan)
    ja_up = out["elig_income_ja_upper"]
    lower = ja_up.map(BAND_LOWER)
    both = pct.notna() & ja_up.notna()
    inside = pct.between(lower, ja_up)
    # JA는 '전부 해당하는 구간'만 표시하는 경향 → 텍스트 상한이 바로 위 구간 안에 있으면 '구간 경계'
    edge = (pct > ja_up) & (pct <= ja_up.map(NEXT_BAND_UPPER))
    out["elig_income_check"] = np.select(
        [~both, inside, edge], ["비교 불가", "일치", "구간 경계"], default="불일치")
    out["elig_income_conflict"] = out["elig_income_check"].eq("불일치")
    return out


def age(df: pd.DataFrame, text: pd.Series) -> pd.DataFrame:
    a0 = pd.to_numeric(df["JA0110_대상연령(시작)"], errors="coerce")
    a1 = pd.to_numeric(df["JA0111_대상연령(종료)"], errors="coerce")
    rng = text.str.extract(RE_AGE_RANGE).astype(float)
    tmin = rng[0].fillna(text.str.extract(RE_AGE_MIN)[0].astype(float))
    tmax = rng[1].fillna(text.str.extract(RE_AGE_MAX)[0].astype(float))
    out = pd.DataFrame(index=df.index)
    out["elig_age_min"] = a0.fillna(tmin)
    out["elig_age_max"] = a1.fillna(tmax)
    out["elig_age_src"] = np.select(
        [a0.notna() & a1.notna(), tmin.notna() | tmax.notna()], ["JA", "텍스트"], default=NONE)
    out["elig_age_text_min"], out["elig_age_text_max"] = tmin, tmax   # 검증용
    # 텍스트 연령은 여러 대상군·자녀 나이가 섞여 불일치가 흔하다 (표본 25건 중 21건이 이 경우).
    # 참고용 표시만 남기고, 검토 대상은 '떨어진 두 범위'(예: 24세 이하 및 65세 이상)를 JA가 연속 범위로 입력한 경우로 한정한다.
    out["elig_age_text_differs"] = (a0.notna() & tmin.notna() & ((a0 - tmin).abs() > 1)) | \
                                   (a1.notna() & tmax.notna() & ((a1 - tmax).abs() > 1))
    dj = text.str.extract(RE_AGE_DISJOINT).astype(float)
    out["elig_age_disjoint"] = dj[0].notna()
    out["elig_age_evidence"] = evidence(text, RE_AGE_DISJOINT, 20)
    out["elig_age_disjoint_young_max"] = dj[0]   # 젊은 쪽 범위 상한 (예: 24)
    out["elig_age_disjoint_old_min"] = dj[1]     # 고령 쪽 범위 하한 (예: 65)
    out["elig_age_conflict"] = out["elig_age_disjoint"] & a0.notna() & (a0 >= dj[0] - 1)
    return out


def listed(df: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    flags = pd.concat([col(df, c).rename(v) for c, v in mapping.items()], axis=1)
    return flags.apply(lambda r: "|".join(r.index[r]), axis=1).replace("", NONE)


# '홈페이지·누리집'은 신청·접수와 함께 쓰일 때만 온라인 신청으로 본다 (안내·공지용 언급이 많음)
RE_ONLINE = re.compile(r"온라인|인터넷\s?(?:신청|접수|청구)|정부24|혜택알리미|복지로|QR|모바일|앱|어플|카카오톡|네이버\s?폼|네이폼|구글\s?폼|"
                       r"(?:홈페이지|누리집|사이트)[^\n.]{0,15}(?:신청(?!서)|접수|등록|청구|예매)")
RE_VISIT = re.compile(r"방문|주민센터|행정복지센터|내방|직접 제출|현장\s?접수|현장\s?신청")
# 우편·팩스·이메일은 제출 경로로 쓰인다. 전화·유선은 문의 번호 표기가 많아 '신청·접수'와 함께일 때만 인정한다.
RE_REMOTE = re.compile(r"우편|등기|이메일|전자우편|e-?mail|팩스|FAX|(?:전화|유선)[^\n,.]{0,6}(?:신청|접수|청구)", re.I)
# 범주가 없는 서비스는 신청방법 칸에 경로만 적는 경우가 많아("○ 기타 - 전화") 전화·유선 언급 자체를 인정한다.
# 번호·☎ 표기는 대부분 문의처라서, 번호 바로 뒤에 접수·예약·신청이 오는 경우만 인정한다 (예: "1644-9666 접수").
RE_REMOTE_LOOSE = re.compile(r"우편|등기|이메일|전자우편|e-?mail|팩스|FAX|전화|유선|"
                             r"\d{2,4}-\d{3,4}(?:-\d{4})?\)?\s{0,6}(?:으로\s?)?(?:접수|예약|신청)", re.I)
# 본인이 직접 신청하지 않고 학교·시설·기관이 추천·신청·청구하는 서비스 (범주가 없는 서비스에만 적용)
RE_VIA_INSTITUTION = re.compile(r"추천|학교(?:장|측)?[^\n]{0,8}(?:신청|제출|취합)|시설(?:장)?[^\n]{0,8}(?:청구|신청)|"
                                r"기관[^\n]{0,8}(?:의뢰|신청|청구)|공문|보육통합정보시스템|에듀파인|문서\s?24|이첩|농협에서")
RE_NOT_NEEDED_TEXT = re.compile(
    r"신청\s?(?:절차\s?)?(?:없음|불필요)|신청서비스 아님|"
    r"별도(?:의)?\s?신청(?:절차)?\s?(?:없음|불필요|없이|하지\s?않)|자동\s?신청")


def clean_method_text(detail_method: pd.Series) -> pd.Series:
    """채널 판정 전에 근거가 될 수 없는 표현을 지운다."""
    return (detail_method.fillna("")
            # 부정 표현 (예: "*방문접수 불가", "온라인 신청 불가 시")
            .str.replace(r"방문\s?(?:접수|신청)?\s?(?:불가|불가능)", "", regex=True)
            .str.replace(r"온라인\s?(?:접수|신청)?\s?(?:불가|불가능)", "", regex=True)
            # 안내·공지용 웹사이트 언급 (예: "홈페이지 공고문 참조")
            .str.replace(r"(?:홈페이지|누리집)\s?(?:공고|공고문|게시판)\S*\s?참조", "", regex=True)
            # 웹사이트 '방문'은 오프라인 방문이 아니다 (예: "홈페이지(www...) 방문")
            .str.replace(r"(홈페이지|누리집|사이트|www\.\S+|https?://\S+)\)?[ \t]?방문", r"\1", regex=True)
            # "방문신청, 우편, 기타 : 해당없음" 처럼 '해당없음'으로 끝나는 항목 나열
            .str.replace(r"[^\n○ㅇ❍\-]*:\s?해당\s?없음", "", regex=True))


def evidence(text: pd.Series, pattern: re.Pattern, width: int = 25) -> pd.Series:
    """패턴이 처음 걸린 위치 앞뒤 구절 (근거 기록용)."""
    def one(t: str) -> str:
        m = pattern.search(t)
        return "" if not m else " ".join(t[max(0, m.start() - width): m.end() + width].split())
    return text.apply(one)


def application(df: pd.DataFrame, detail_method: pd.Series) -> pd.DataFrame:
    """신청 채널 = 목록 범주형 신청방법 ∪ 상세 서술형 신청방법 키워드.

    범주만 쓰면 온라인 가능 여부가 과소 추정된다 (범주 '방문만' 6,321건 중 422건이 본문에 온라인 신청 언급, 2026-10-31).
    범주값(*_cat), 본문값(*_text), 최종값을 따로 보존한다.
    """
    m = df["신청방법"].fillna("")
    cat_known = m.str.contains("온라인|방문|신청불필요")
    d = clean_method_text(detail_method)
    c_on, c_vis = m.str.contains("온라인"), m.str.contains("방문")
    t_on, t_vis = d.str.contains(RE_ONLINE), d.str.contains(RE_VISIT)
    # '전화 문의'·'☎ 문의' 등 문의 안내 구절은 신청 경로 근거에서 뺀다
    d_noinq = d.str.replace(r"[^\n.]{0,30}문의[^\n.]{0,20}", "", regex=True)
    t_rem = d.str.contains(RE_REMOTE) | (~cat_known & d_noinq.str.contains(RE_REMOTE_LOOSE))
    online, visit, remote = c_on | t_on, c_vis | t_vis, t_rem
    c_not_needed = m.str.contains("신청불필요")
    not_needed = (c_not_needed
                  | df["신청기한"].fillna("").str.contains(r"신청\s?불필요|별도의 신청절차가 없음")
                  | (~cat_known & d.str.contains(RE_NOT_NEEDED_TEXT)))
    via_inst = ~cat_known & ~not_needed & d.str.contains(RE_VIA_INSTITUTION)
    docs = df["구비서류"].fillna("").str.strip()

    out = pd.DataFrame(index=df.index)
    out["apply_category_raw"] = m                                   # 원본 범주값
    out["apply_online_cat"], out["apply_visit_cat"] = c_on, c_vis    # 범주 기준
    out["apply_online_text"], out["apply_visit_text"], out["apply_remote_text"] = t_on, t_vis, t_rem  # 본문 기준
    out["apply_online"], out["apply_visit"], out["apply_remote_other"] = online, visit, remote       # 최종
    out["apply_not_needed"] = not_needed
    out["apply_via_institution"] = via_inst
    out["apply_channel"] = np.select(
        [not_needed, online & visit, online & remote, online, visit & remote, visit, remote, via_inst],
        ["신청 불필요", "온라인+방문", "온라인+전화·우편", "온라인", "방문+전화·우편", "방문만", "전화·우편·이메일",
         "기관 경유(본인 신청 아님)"],
        default=NONE)
    added = (t_on & ~c_on) | (t_vis & ~c_vis) | t_rem
    out["apply_channel_src"] = np.select(
        [out["apply_channel"] == NONE, not_needed & c_not_needed, not_needed,
         cat_known & ~added, cat_known & added],
        [NONE, "범주", "본문", "범주", "범주+본문"], default="본문")
    ev_on, ev_vis, ev_rem = evidence(d, RE_ONLINE), evidence(d, RE_VISIT), evidence(d, RE_REMOTE)
    ev_rem = ev_rem.where(ev_rem.ne(""), evidence(d_noinq, RE_REMOTE_LOOSE).where(~cat_known, ""))
    out["apply_evidence"] = (("온라인: " + ev_on).where(ev_on != "", "") + " / "
                             + ("방문: " + ev_vis).where(ev_vis != "", "") + " / "
                             + ("기타: " + ev_rem).where(ev_rem != "", "")).str.strip(" /")
    ev_none = evidence(d, RE_NOT_NEEDED_TEXT).where(not_needed, "")
    ev_inst = evidence(d, RE_VIA_INSTITUTION).where(via_inst, "")
    out["apply_evidence"] = (out["apply_evidence"] + " / "
                             + ("신청 불필요: " + ev_none).where(ev_none.ne(""), "") + " / "
                             + ("기관 경유: " + ev_inst).where(ev_inst.ne(""), "")).str.strip(" /")
    out["docs_required"] = np.select([docs.isin(["해당없음", "없음", "-"]), docs != ""], ["불필요", "필요"], default=NONE)
    out["always_open"] = df["신청기한유형"].eq("상시신청")
    return out


def build_eligibility(df: pd.DataFrame, detail_method: pd.Series) -> pd.DataFrame:
    """입력 행 순서를 유지하는 순수 변환. 파일 저장은 호출자가 결정한다."""
    text = (df["지원대상"].fillna("") + "\n" + df["선정기준"].fillna(""))
    out = pd.concat([df[["서비스ID", "서비스명", "소관기관명", "소관기관유형", "youth_overlap", "youth_focused"]],
                     age(df, text), income(df, text), application(df, detail_method)], axis=1)
    out["elig_status"] = listed(df, STATUS)
    out["elig_household"] = listed(df, HOUSEHOLD)
    out["residence_mentioned"] = text.str.contains(RE_RESIDENCE)
    out["needs_review"] = out["elig_income_conflict"] | out["elig_age_conflict"]
    out["personal"] = df["사용자구분"].fillna("").str.contains("개인")
    out["info_known_count"] = (
        (out["elig_age_src"] != NONE).astype(int) + (out["elig_income_src"] != NONE).astype(int)
        + (out["apply_channel"] != NONE).astype(int) + (out["docs_required"] != NONE).astype(int))
    return out


def main() -> None:
    df = pd.read_csv(PROCESSED / "gov24_merged.csv", dtype=str, low_memory=False)
    detail = pd.read_csv(RAW / "gov24" / "gov24_serviceDetail.csv", dtype=str).set_index("서비스ID")["신청방법"]
    out = build_eligibility(df, df["서비스ID"].map(detail))
    out.to_csv(PROCESSED / "gov24_eligibility.csv", index=False, encoding="utf-8-sig")

    print("== 출처별 건수")
    for c in ["elig_age_src", "elig_income_src", "apply_channel", "apply_channel_src", "docs_required"]:
        print(c, out[c].value_counts().to_dict())

    print("== 소득 JA·텍스트 대조:", out["elig_income_check"].value_counts().to_dict())
    print(f"== 연령 떨어진 범위(예: 24세 이하 및 65세 이상): {int(out['elig_age_disjoint'].sum())}건, JA 연속범위 입력 의심 {int(out['elig_age_conflict'].sum())}건")
    print(f"== 검토 필요 합계 {int(out['needs_review'].sum())} (소득 불일치 {int(out['elig_income_conflict'].sum())} + 연령 떨어진 범위 {int(out['elig_age_conflict'].sum())})")
    p = out[out["personal"]]
    print(f"== 개인 대상 {len(p):,}건 중 명시 없음: 연령 {int((p.elig_age_src=='명시 없음').sum())}, 소득 {int((p.elig_income_src=='명시 없음').sum())}, 신청채널 {int((p.apply_channel=='명시 없음').sum())}")
    y = out[out["youth_overlap"].astype(str).eq("True")]
    print(f"== 청년 포함 {len(y):,}건 중 검토 필요 {int(y.needs_review.sum())}, 신청채널 명시 없음 {int((y.apply_channel=='명시 없음').sum())}")


if __name__ == "__main__":
    main()
