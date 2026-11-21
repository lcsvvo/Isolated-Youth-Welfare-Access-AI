"""1단계(위기 확인 + 욕구·조건 추출) 개발 세트를 만든다.

입력 (모두 레포 안 파일, 수정하지 않음)
- 말뭉치 300건 사람 정답: logs/validation/corpus_gold_review/human/corpus_review_R1~R4.csv
  + 배정표 review_assignment.csv(주 검수자) + 불일치 합의안 disagreement_resolution.csv
- 페르소나 개발 카드: data/raw/user_input_corpus/persona_cards/persona_cards_<이름>.xlsx `작성카드` (사람별 파일)
  세트가 '개발'이고 `작성자메모`가 빈 카드(검수 완료)만. 평가 카드 80장(홀드아웃)은 읽지 않는다.

출력
- ai/data/dev_corpus.jsonl: 말뭉치 300건. gold(평가용) + label(few-shot 예시용 extract-v1, 위기 문장은 null)
- ai/data/dev_cards.jsonl: 검수 끝난 개발 카드. gold만 (근거 칸이 없어 few-shot에는 쓰지 않음)
- ai/data/holdout_cards.jsonl (--holdout일 때만): 평가 카드 80장. 최종 평가 직전, 설정 동결 후 한 번 만든다.
  80장 모두 작성글이 있어야 만들어지며 few-shot 예시 풀(dev_corpus)과 섞이지 않는다.

실행 (ai/ 폴더에서)
  python -m eval.build_devset              # 개발 세트
  python -m eval.build_devset --holdout    # 평가 카드 80장 (최종 평가 직전)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

from core.utils import evidence_in
from extract.schema import GOLD_CONSTRAINT, GOLD_CONSTRAINT_ORDER, SCHEMA_VERSION

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
REVIEW = ROOT / "logs" / "validation" / "corpus_gold_review"
CARDS_DIR = ROOT / "data" / "raw" / "user_input_corpus" / "persona_cards"
OUT_CORPUS = AI / "data" / "dev_corpus.jsonl"
OUT_CARDS = AI / "data" / "dev_cards.jsonl"
OUT_HOLDOUT = AI / "data" / "holdout_cards.jsonl"

DEV_SET = "개발(LLM 생성 후 사람 수정)"
HOLDOUT_SET = "평가(사람 작성)"
HOLDOUT_N = 80
FIELD = {"필요영역": "h_필요영역", "활동범위": "h_활동범위", "제약": "h_제약", "위기": "h_위기"}
EVIDENCE = {"h_필요영역": "h_필요영역_근거", "h_활동범위": "h_활동범위_근거", "h_제약": "h_제약_근거"}
RANGE_ALIAS = {"집안": "집 안", "방안": "방 안"}

# 말뭉치 h_제약은 자유 기재다. 구분(';')마다 채널(전화·방문)과 정도(부담·불가)를 읽는다.
# 채널·정도가 둘 다 읽히는 구분만 쓰고 나머지(문자 선호, 가족에게 말하기, 일정 등)는 버린다.
_PHONE = re.compile(r"전화|통화")
_VISIT = re.compile(r"방문|대면|외출|이동|모임|참석|참여|센터|기관")
_HOME = re.compile(r"가정\s*방문")                       # 집으로 찾아오는 방문은 사용자 이동 제약이 아님
_SKIP = re.compile(r"가능|필요$|희망$|선호$")             # '방문 가능', '이동 지원 차량 필요' 등
_BLOCK = re.compile(r"불가|제외|못")
_BURDEN = re.compile(r"부담|어려움|기피|회피|제한|대신|보다")


def parse_constraint(value: str | None) -> list[tuple[str, str]]:
    """'전화 부담; 문자 선호' → [('전화', '부담')]. 같은 채널은 불가 우선."""
    out: dict[str, str] = {}
    for seg in str(value or "").split(";"):
        seg = seg.strip()
        if not seg or seg in ("없음", "판단 불가"):
            continue
        if _HOME.search(seg):
            continue
        ch = "전화" if _PHONE.search(seg) else "방문" if _VISIT.search(seg) else None
        if ch == "전화" and re.search(r"대신|보다", seg):   # '전화 대신 채팅', '전화보다 글 상담'
            lv = "부담"
        elif _SKIP.search(seg) and not _BLOCK.search(seg) and not _BURDEN.search(seg):
            continue
        else:
            lv = "불가" if _BLOCK.search(seg) else "부담" if _BURDEN.search(seg) else None
        if ch and lv and (out.get(ch) != "불가"):
            out[ch] = lv
    return [(ch, out[ch]) for ch in ("전화", "방문") if ch in out]


def strongest(cons: list[tuple[str, str]]) -> str:
    return next((GOLD_CONSTRAINT[k] for k in GOLD_CONSTRAINT_ORDER if k in cons), "없음")


def split_domains(value: str | None, domains: list[str]) -> list[str]:
    out = []
    for d in str(value or "").split(";"):
        d = d.strip()
        if d and d != "판단 불가" and d != "nan":
            if d not in domains:
                raise ValueError(f"영역 목록에 없는 값: {d}")
            if d not in out:
                out.append(d)
    return out


def span(evidence: str | None, text: str) -> str | None:
    """근거가 원문에 그대로 있으면 그대로, 쉼표로 이은 여러 조각이면 원문에서 처음~끝을 덮는 구간."""
    ev = str(evidence or "").strip()
    if not ev or ev == "nan":
        return None
    if evidence_in(ev, text):
        return ev
    pos = []
    for seg in re.split(r"[,，/]\s*", ev):
        seg = seg.strip()
        i = text.find(seg) if seg else -1
        if i >= 0:
            pos.append((i, i + len(seg)))
    if not pos:
        return None
    return text[min(p[0] for p in pos):max(p[1] for p in pos)]


def load_reviews() -> pd.DataFrame:
    rows = []
    for f in sorted((REVIEW / "human").glob("corpus_review_R*.csv")):
        x = pd.read_csv(f, dtype=str, encoding="utf-8-sig").dropna(subset=["말뭉치ID"])
        rows.append(x.assign(reviewer=f.stem.split("_")[-1]))
    return pd.concat(rows, ignore_index=True)


def corpus_gold(domains: list[str]) -> pd.DataFrame:
    """주 검수자 판정에 불일치 합의안을 덮어쓴 300건 정답."""
    h = load_reviews()
    assign = pd.read_csv(REVIEW / "review_assignment.csv", dtype=str, encoding="utf-8-sig")
    base = assign.merge(h, left_on=["말뭉치ID", "reviewer_primary"], right_on=["말뭉치ID", "reviewer"],
                        how="left", validate="1:1", suffixes=("", "_h"))
    if base["사용자 입력"].isna().any():
        missing = base.loc[base["사용자 입력"].isna(), "말뭉치ID"].tolist()
        raise ValueError(f"주 검수자 판정이 없는 문장: {missing}")
    base = base.set_index("말뭉치ID")
    base["resolved"] = ""

    res = pd.read_csv(REVIEW / "disagreement_resolution.csv", dtype=str, encoding="utf-8-sig")
    for _, r in res.iterrows():
        col = FIELD[r["불일치 항목"]]
        base.loc[r["말뭉치ID"], col] = r["합의안"]
        if col in EVIDENCE:
            base.loc[r["말뭉치ID"], EVIDENCE[col]] = r["원문 근거"]
        base.loc[r["말뭉치ID"], "resolved"] += r["불일치 항목"] + ";"

    base["h_활동범위"] = base["h_활동범위"].str.strip().replace(RANGE_ALIAS)
    base["domains"] = base["h_필요영역"].map(lambda v: split_domains(v, domains))
    base["cons"] = base["h_제약"].map(parse_constraint)
    return base.reset_index()


def few_shot_label(r) -> dict | None:
    """사람 정답을 extract-v1 예시 출력으로 바꾼다. 위기 문장은 1단계를 실행하지 않으므로 None.

    - 영역은 검수 칸에 적힌 순서대로 첫 영역 main, 나머지 sub (검수 양식에 main/sub 구분이 없음)
    - 근거는 원문 구간(span)으로 맞추고, 원문에서 찾지 못하면 그 항목을 예시에서 뺀다
    """
    if r["h_위기"] == "Y":
        return None
    text = r["사용자 입력"]
    need_ev = span(r["h_필요영역_근거"], text)
    needs = [{"domain": d, "level": "main" if i == 0 else "sub", "evidence": need_ev}
             for i, d in enumerate(r["domains"])] if need_ev else []
    if r["domains"] and not needs:
        return None
    ar, ar_ev = r["h_활동범위"], span(r["h_활동범위_근거"], text)
    if ar != "판단 불가" and not ar_ev:
        return None
    con_ev = span(r["h_제약_근거"], text)
    if r["cons"] and not con_ev:
        return None
    return {"needs": needs,
            "activity_range": ar if ar_ev else "판단 불가",
            "activity_evidence": ar_ev if ar != "판단 불가" else None,
            "constraints": [{"channel": ch, "level": lv, "evidence": con_ev} for ch, lv in r["cons"]],
            "age": None, "region_sido": None}


def corpus_records(domains: list[str]) -> list[dict]:
    g = corpus_gold(domains)
    out = []
    for _, r in g.sort_values("말뭉치ID").iterrows():
        out.append({
            "id": r["말뭉치ID"], "source": "corpus", "type_no": int(r["말뭉치ID"][1:3]),
            "text": r["사용자 입력"],
            "gold": {"영역": sorted(r["domains"]), "활동범위": r["h_활동범위"], "제약": strongest(r["cons"]),
                     "제약_원문": r["h_제약"], "위기": r["h_위기"]},
            "resolved": r["resolved"].rstrip(";") or None,
            "label": few_shot_label(r),
        })
    return out


def load_cards(cards_dir: Path = CARDS_DIR) -> pd.DataFrame:
    """사람별 카드 파일(persona_cards_<이름>.xlsx)의 `작성카드`를 합친다. 카드ID가 겹치면 오류."""
    files = sorted(cards_dir.glob("persona_cards_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"카드 파일 없음: {cards_dir}")
    c = pd.concat([pd.read_excel(f, sheet_name="작성카드", dtype=str) for f in files], ignore_index=True)
    dup = c["카드ID"][c["카드ID"].duplicated()].tolist()
    if dup:
        raise ValueError(f"카드ID 중복: {dup}")
    return c


def card_records(domains: list[str], holdout: bool = False) -> list[dict]:
    """개발 카드(작성자메모가 빈 검수 완료분) 또는 평가 카드 80장(전부 작성돼야 함)."""
    c = load_cards()
    if holdout:
        dev = c[c["세트"] == HOLDOUT_SET]
        empty = dev.loc[dev["작성글"].isna() | dev["작성글"].fillna("").str.strip().eq(""), "카드ID"].tolist()
        if len(dev) != HOLDOUT_N or empty:
            raise ValueError(f"평가 카드 {len(dev)}장, 작성글 빈 카드 {len(empty)}장: {empty[:10]}")
    else:
        dev = c[(c["세트"] == DEV_SET) & c["작성글"].notna() & c["작성자메모"].isna()]
    out = []
    for _, r in dev.iterrows():
        out.append({
            "id": r["카드ID"], "source": "card", "type_no": int(r["유형번호"]), "author": r["작성자"],
            "text": r["작성글"].strip(),
            "gold": {"영역": sorted(split_domains(r["정답_영역"], domains)), "활동범위": r["정답_활동범위"],
                     "제약": r["정답_제약"], "나이언급": r["정답_나이언급"], "지역언급": r["정답_지역언급"],
                     "위기": r["정답_위기"], "기대행동": r["기대행동"]},
            "label": None,
        })
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="평가 카드 80장만 만든다 (최종 평가 직전)")
    domains = yaml.safe_load(open(AI / "configs" / "domains.yaml", encoding="utf-8"))["domains"]
    if ap.parse_args().holdout:
        hold = card_records(domains, holdout=True)
        write_jsonl(hold, OUT_HOLDOUT)
        print(f"{OUT_HOLDOUT.name}: {len(hold)}장 (위기 Y {sum(r['gold']['위기'] == 'Y' for r in hold)})")
        return
    corpus, cards = corpus_records(domains), card_records(domains)
    write_jsonl(corpus, OUT_CORPUS)
    write_jsonl(cards, OUT_CARDS)
    n_label = sum(r["label"] is not None for r in corpus)
    print(f"{OUT_CORPUS.name}: {len(corpus)}건 (few-shot 예시 가능 {n_label}, 위기 Y {sum(r['gold']['위기'] == 'Y' for r in corpus)})")
    print(f"{OUT_CARDS.name}: {len(cards)}장 (작성자: {', '.join(sorted({r['author'] for r in cards}))})")


if __name__ == "__main__":
    main()
