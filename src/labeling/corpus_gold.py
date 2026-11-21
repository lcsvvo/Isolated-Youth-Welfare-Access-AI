"""사용자 입력 말뭉치의 문장별 정답 판정 양식 (2026-11-05).

말뭉치(data/raw/user_input_corpus/user_input_corpus_300_v2.xlsx)의 `정답_영역`·`정답_단계`·`정답_위기`·`기대행동`은
유형마다 30문장이 모두 같은 값이다(페르소나 유형의 기본값). 문장에 없는 필요를 덧붙인 모델이 점수를 받지 않도록
문장 단위 정답을 사람이 따로 단다. 원자료는 수정하지 않는다.

- 유형 기본값은 키 파일에 `유형_기본영역` 등으로 이름을 바꿔 두고, 층화·비교에만 쓴다.
- 키 파일 끝의 `유형_기본영역_v3`, `유형_기본활동범위`는 persona_common_v3 `페르소나_유형`의 유형 기본값이다.
  층화·비교용 참고 열이며 문장별 사람 정답이 아니다. 기존 열은 바꾸지 않는다.
- 판정 시트는 이미 있으면 다시 만들지 않는다(없을 때만 생성).
- 판정 시트에는 유형·기본값·제약 라벨을 보여주지 않는다(blind).
- 판정 규칙: 문장에 직접 드러난 것만. 드러나지 않으면 '판단 불가'. 부담과 불가능을 구분한다.

실행: python -m src.labeling.corpus_gold
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..collect._common import RAW, ROOT
from ..preprocess import keywords as K

SRC = RAW / "user_input_corpus" / "user_input_corpus_300_v2.xlsx"
PERSONA = RAW / "user_input_corpus" / "persona_cards" / "persona_common_v3.xlsx"
OUT = ROOT / "outputs" / "02_labeling" / "tables"
SEED_DOUBLE = 20261111
DOUBLE_SHARE = 0.2
RENAME = {"정답_영역": "유형_기본영역", "정답_단계": "유형_기본단계", "정답_위기": "유형_기본위기", "기대행동": "유형_기대행동"}
HUMAN_COLS = [
    "h_필요영역",          # 앱 9개 영역(K.APP_DOMAINS_V3) 중 문장에 드러난 것만, ';' 구분 / 판단 불가
    "h_필요영역_근거",      # 해당 표현 그대로
    "h_활동범위",          # 방 안 / 집 안 / 집 근처 / 필요하면 외출 / 판단 불가 (조사 F7 기준, 평소 패턴이 드러날 때만)
    "h_활동범위_근거",
    "h_제약",              # 예: 전화 부담 / 방문 불가 / 비용 불가 / 없음 / 판단 불가 (부담·불가능 구분)
    "h_제약_근거",
    "h_위기",              # Y / N
    "h_note", "labeler", "labeled_at",
]
KEY_COLS = ["말뭉치ID", "유형번호", "페르소나 유형", *RENAME.values(), "제약_표현", "double_label"]
V3_COLS = ["유형_기본영역_v3", "유형_기본활동범위"]
# 활동 범위 값 (조사 F7 기준, 기존 외출 단계 0~3/N 대체). keywords.py는 마스터 메타 해시 대상이라 여기 둔다
ACTIVITY_RANGES = ["방 안", "집 안", "집 근처", "필요하면 외출", "판단 불가"]
CSV_OPTIONS = {"index": False, "encoding": "utf-8-sig", "lineterminator": "\n"}
DOMAIN_8TO9 = K.APP_DOMAIN_CORPUS8_TO_V3      # 말뭉치 8개 영역 → v3 이름 (영역 이름은 keywords.py에서만 정의)


def load() -> pd.DataFrame:
    return pd.read_excel(SRC, sheet_name="입력말뭉치_300", dtype=str)


def type_default_check(c: pd.DataFrame) -> pd.DataFrame:
    """유형별로 각 정답 열의 고유값 수. 1이면 유형 기본값이다."""
    cols = [k for k in RENAME if k in c] + ["제약_표현"]
    return c.groupby("유형번호")[cols].nunique().reset_index()


def load_persona_types() -> pd.DataFrame:
    return pd.read_excel(PERSONA, sheet_name="페르소나_유형", dtype=str)


def type_defaults_v3(p: pd.DataFrame) -> pd.DataFrame:
    """카드의 정답 영역·활동범위를 이름 대응 없이 그대로 읽고 허용 값인지 검증한다."""
    domains_ok = {*K.APP_DOMAINS_V3, "판단 불가"}
    rows = []
    for number, name, rng, dom in p[["유형번호", "유형", "정답_활동범위", "정답 영역(기본)"]].itertuples(index=False, name=None):
        domains = [x.strip() for x in dom.split(";")]
        ranges = [x.strip() for x in rng.replace(" (카드별 교대)", "").split("/")]
        bad = [x for x in domains if x not in domains_ok] + [x for x in ranges if x not in ACTIVITY_RANGES]
        if bad:
            raise ValueError(f"유형 {number}: 허용 목록에 없는 값 {bad}")
        rows.append({"유형번호": number, "페르소나 유형": name,
                     "유형_기본영역_v3": "; ".join(domains), "유형_기본활동범위": " / ".join(ranges)})
    out = pd.DataFrame(rows)
    if out["유형번호"].duplicated().any():
        raise ValueError("카드 유형번호가 중복됨")
    return out


def domain_8to9(c: pd.DataFrame, p: pd.DataFrame | None = None) -> pd.DataFrame:
    """유형 기본값의 변환 후보를 만들고 의미 범위가 바뀌는 영역을 표시한다.

    p(페르소나 카드)가 있으면 카드 값과 후보의 차이, 이전 단계값, 활동범위를 비교 열로 붙인다.
    """
    type_cols = ["유형번호", "페르소나 유형", "정답_영역"]
    if p is not None:
        type_cols.append("정답_단계")
    rows = []
    for number, persona, old, *stage in c[type_cols].drop_duplicates().itertuples(index=False, name=None):
        domains, notes = [], []
        for value in old.split(";"):
            domain = value.strip()
            proposed, note = DOMAIN_8TO9.get(domain, (domain, ""))
            domains.append(proposed)
            if note:
                notes.append(note)
        row = {
            "유형번호": number,
            "페르소나 유형": persona,
            "유형_기본영역_v2": old,
            "유형_기본영역_v3_후보": "; ".join(domains),
            "변환_상태": "확인 필요" if notes else "1:1",
            "note": " / ".join(notes),
        }
        if p is not None:
            card = p_v3(p).loc[number]
            card_domains = card["유형_기본영역_v3"].split("; ")
            row["유형_기본영역_v3"] = card["유형_기본영역_v3"]
            row["카드_추가"] = "; ".join(x for x in card_domains if x not in domains)
            row["카드_제외"] = "; ".join(x for x in domains if x not in card_domains)
            row["유형_기본단계_v2"] = stage[0]
            row["유형_기본활동범위"] = card["유형_기본활동범위"]
        rows.append(row)
    return pd.DataFrame(rows)


def p_v3(p: pd.DataFrame) -> pd.DataFrame:
    return type_defaults_v3(p).set_index("유형번호")


def build(c: pd.DataFrame, p: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if p is None:
        p = load_persona_types()
    key = c.rename(columns=RENAME)[["말뭉치ID", "유형번호", "페르소나 유형", *RENAME.values(), "제약_표현"]].copy()
    rng = np.random.default_rng(SEED_DOUBLE)
    key["double_label"] = rng.random(len(key)) < DOUBLE_SHARE
    sheet = key[["말뭉치ID", "double_label"]].merge(c[["말뭉치ID", "사용자 입력"]], on="말뭉치ID")
    for col in HUMAN_COLS:
        sheet[col] = ""
    defaults = type_defaults_v3(p)
    names = defaults.set_index("유형번호")["페르소나 유형"]
    mapped = key["유형번호"].map(names)
    if mapped.isna().any() or (mapped != key["페르소나 유형"]).any():
        raise ValueError("말뭉치 유형 이름이 persona_common_v3 유형과 다름")
    key = key.merge(defaults.drop(columns="페르소나 유형"), on="유형번호", how="left", validate="m:1")
    if key[V3_COLS].isna().any().any():
        raise ValueError("v3 기본값이 비어 있는 행이 있음")
    return key, sheet.sample(frac=1, random_state=SEED_DOUBLE).reset_index(drop=True)


def write_key(key: pd.DataFrame, path) -> None:
    """키 파일을 쓴다. 기존 9열은 바뀌면 중단하고, v3 열은 손으로 고친 값을 남긴다."""
    if not path.exists():
        key.to_csv(path, **CSV_OPTIONS)
        print(f"{path.name} 새로 작성 ({len(key)}행)")
        return
    prev = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    new = key.astype(str)
    if (list(prev.columns[:len(KEY_COLS)]) != KEY_COLS or len(prev) != len(new)
            or not prev[KEY_COLS].equals(new[KEY_COLS])):
        raise RuntimeError(f"{path.name}: 기존 열·행·double_label이 달라 쓰지 않음")
    final = key.copy()
    kept = 0
    for col in V3_COLS:
        if col in prev:
            keep = prev[col].ne("") & prev[col].ne(new[col])
            final.loc[keep.to_numpy(), col] = prev.loc[keep, col].to_numpy()
            kept += int(keep.sum())
    if kept:
        print(f"{path.name}: 손으로 고친 v3 값 {kept}칸 유지")
    if list(prev.columns) == list(final.columns) and prev.equals(final.astype(str)):
        print(f"{path.name} 변경 없음")
        return
    final.to_csv(path, **CSV_OPTIONS)
    print(f"{path.name} 갱신 ({len(final)}행)")


def main() -> None:
    c = load()
    p = load_persona_types()
    OUT.mkdir(parents=True, exist_ok=True)
    key, sheet = build(c, p)
    type_default_check(c).to_csv(OUT / "corpus_type_default_check.csv", **CSV_OPTIONS)
    write_key(key, OUT / "corpus_gold_key.csv")
    domain_8to9(c, p).to_csv(OUT / "corpus_domain_8to9.csv", **CSV_OPTIONS)
    path = OUT / "corpus_gold_sheet.csv"
    if path.exists():
        print("corpus_gold_sheet.csv가 이미 있어 다시 만들지 않습니다")
        return
    sheet.to_csv(path, **CSV_OPTIONS)
    print(f"corpus_gold_sheet.csv {len(sheet)}행 (이중 판정 {int(sheet.double_label.sum())})")


if __name__ == "__main__":
    main()
