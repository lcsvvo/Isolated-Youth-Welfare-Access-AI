"""Q1 추가 검토의 표본과 원문을 고정한다. 모델은 호출하지 않는다."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from ..collect._common import PROCESSED, ROOT
from ..gap_index.q1_supply import in_scenario, load_master
from .q1_label_audit import FOCUS, existing_labels, source_only

OUT = ROOT / "private" / "q1_followup"
SEED = 20261105
CRITERIA_VERSION = "q1-followup-20261105-v1"
CRITERIA = """원문에 실제 제공하는 내용과 수혜자를 확인한다. 기존 라벨은 판정 후 대조한다.
①은 선정형 숙박·긴급 보호와 구분되는 자유 방문·체류 공간이다.
② 핵심은 생활 리듬·자기관리·독립생활·사회 적응의 회복 활동·훈련 또는 명시된 목적 지원이다.
일반 간병·가사·식사·동행·안부확인은 care 확장으로 기록한다. 보험료의 복귀 목적 문구만으로 ②를 부여하지 않는다.
신체·장애 재활도 실제 생활 기능·사회 적응 훈련이 있으면 핵심 후보가 될 수 있다.
③ 혼자 하는 활동, ④ 함께 하는 취미·문화·체육, ⑤ 전문가 상담·치료, ⑥ 또래·멘토 지지를 실제 지원에서 확인한다.
⑦은 실제 교육·학습 또는 그 목적의 지원이며 본인·자녀 수혜자를 구분한다. 시험 응시료만 지원하면 ⑧이다.
⑧은 취업 정보·구직·직업훈련·일경험 지원이다. ⑨는 공동생활·규칙적인 생활을 실제 지원한다.
⑩ main은 생계·생활비 또는 반복 필수비용 지원, candidate는 저축·보상·일시금·용도 미상 지급이다.
명시된 주거·교육·취업·의료 등의 지정 용도 지급은 해당 영역으로 기록한다.
청년 범위는 19~34세 기준 A/B/U/C이며, 지원대상 전체를 가리키는 누구나 표현은 조건·충돌을 대조해 B 근거로 기록한다.
숫자 연령을 추정하지 않고 universal_target 근거를 구분한다. 소득 무관 누구나를 연령 무관으로 확대하지 않는다.
목적·홍보·제공 인력·신청 대리인·자녀 연령을 수혜자나 실제 지원으로 혼동하지 않는다.
판정이 부족하면 unknown 또는 팀 결정 필요를 남기고 근거 인용을 보존한다."""


def previous_ids() -> set[str]:
    doc = (ROOT / "docs/q1_label_audit.md").read_text(encoding="utf-8")
    cases = doc.split("### 사례별 상태", 1)[1].split("### 공식 결과", 1)[0]
    ids = set(re.findall(r"WLF\d+", cases))
    path = ROOT / "private/q1_label_audit/audit.sqlite3"
    if path.is_file():
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as db:
            ids.update(sid for (sid,) in db.execute("SELECT DISTINCT serv_id FROM decisions"))
    return ids


def select(m: pd.DataFrame, reviewed: set[str]) -> pd.DataFrame:
    eligible = in_scenario(m, {"A", "B"}, {"age_overlap"})
    texts = m[["servNm", "summary_text", "benefit_text"]].fillna("").agg(" ".join, axis=1)
    groups: dict[str, set[str]] = {}
    for area, pattern in FOCUS.items():
        positive = m.need_area_q21.str.split("|").map(lambda labels: area in labels)
        for sid in m.loc[eligible & ~positive & texts.str.contains(pattern, regex=True), "servId"]:
            groups.setdefault(sid, set()).add(f"target_{area}")
    target_ids = set(groups)
    available = m.loc[eligible & ~m.servId.isin(reviewed | target_ids)].copy()
    for stratum, mask in (("random_labeled", available.need_area_q21.ne("N")),
                          ("random_unlabeled", available.need_area_q21.eq("N"))):
        pool = available.loc[mask].sort_values("servId")
        if len(pool) < 20:
            raise ValueError(f"{stratum}: 표본 후보가 20건 미만입니다.")
        for sid in pool.sample(n=20, random_state=SEED).servId:
            groups.setdefault(sid, set()).add(stratum)

    # 엄격 시나리오 밖의 명시적인 전체 대상 표현을 소수 점검한다.
    targets = m.target_text.fillna("")
    universal = targets.str.contains(
        r"(?:시민|주민|국민|성인|전국민|전\s*국민).{0,12}누구나|연령\s*제한\s*(?:없|무)|모든\s*연령",
        regex=True,
    )
    outside = m.loc[universal & m.youth_scope.isin(["U", "C"])].sort_values("servId")
    for sid in outside.loc[~outside.servId.isin(reviewed)].head(5).servId:
        groups.setdefault(sid, set()).add("universal_outside")
    rows = [{"servId": sid, "review_groups": "|".join(sorted(labels)), "already_reviewed": sid in reviewed}
            for sid, labels in sorted(groups.items())]
    return pd.DataFrame(rows)


def prepare() -> None:
    master_path = PROCESSED / "services_master.csv"
    m = load_master()
    if not m.servId.is_unique or m.servId.isna().any() or len(m) != 5288:
        raise ValueError("Unexpected service IDs or count")
    reviewed = previous_ids()
    selection = select(m, reviewed)
    manifest = {
        "criteria_version": CRITERIA_VERSION, "criteria": CRITERIA, "seed": SEED,
        "master_sha256": hashlib.sha256(master_path.read_bytes()).hexdigest(),
        "previous_ids": sorted(reviewed), "selection": selection.to_dict(orient="records"),
        "sampling": "S2_AB_strict, 20 labeled + 20 unlabeled, excluding prior and targeted cases",
        "new_model_service_limit": 30,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing follow-up manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    selection.to_csv(OUT / "selection.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    new = selection.loc[~selection.already_reviewed]
    with (OUT / "sources.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for sid in new.servId:
            row = m.loc[m.servId.eq(sid)].iloc[0]
            file.write(json.dumps({"servId": sid, "source": source_only(row)}, ensure_ascii=False) + "\n")
    print(selection.groupby(["review_groups", "already_reviewed"]).size().to_string())
    print(f"New unique services: {len(new)}; prior exclusion: {len(reviewed)}")


def comparison(m: pd.DataFrame) -> pd.DataFrame:
    """고정한 원문 판정과 기존 라벨을 대조한다. 미결정은 오류로 세지 않는다."""
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((PROCESSED / "services_master.csv").read_bytes()).hexdigest() != manifest["master_sha256"]:
        raise ValueError("표본 고정 후 마스터가 변경됐습니다.")
    selected = {r["servId"]: r for r in manifest["selection"] if not r["already_reviewed"]}
    decisions = [json.loads(line) for line in (OUT / "source_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(decisions) != len(selected) or {d["servId"] for d in decisions} != set(selected):
        raise ValueError("고정 표본과 판정 ID가 다르거나 중복됐습니다.")
    indexed = m.set_index("servId", drop=False)
    rows = []
    for d in decisions:
        row = indexed.loc[d["servId"]]
        source = source_only(row)
        digest = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if d["criteria_version"] != CRITERIA_VERSION or digest != d["source_sha256"]:
            raise ValueError(f"판정 기준 또는 원문이 변경됐습니다: {d['servId']}")
        if not d["evidence"] or any(not e["quote"] or e["quote"] not in source.get(e["field"], "") for e in d["evidence"]):
            raise ValueError(f"원문 인용 불일치: {d['servId']}")
        before = existing_labels(row)
        pending = set(d["unresolved_items"])
        if "economy" in pending:
            # ⑩ 범위가 미결정이면 영역 차이에서도 ⑩을 판정하지 않는다.
            old_areas = set(before["areas"]) - {"q21_10"}
            new_areas = set(d["areas"]) - {"q21_10"}
        else:
            old_areas, new_areas = set(before["areas"]), set(d["areas"])
        changes = []
        for key in ("youth", "areas", "economy", "education"):
            if key in pending or (key == "education" and "areas" in pending):
                continue
            if key == "areas":
                different = old_areas != new_areas
            elif key == "education" and "q21_07" not in old_areas | new_areas:
                different = False
            else:
                different = before[key] != d[key]
            if different:
                changes.append(key)
        rows.append({"servId": d["servId"], "servNm": row.servNm,
                     "review_groups": selected[d["servId"]]["review_groups"],
                     "changed_items": "|".join(changes), "unresolved_items": "|".join(sorted(pending)),
                     "old_labels": json.dumps(before, ensure_ascii=False, sort_keys=True),
                     "review_labels": json.dumps({k: d[k] for k in before}, ensure_ascii=False, sort_keys=True),
                     "note": d["note"], "evidence": json.dumps(d["evidence"], ensure_ascii=False),
                     "human_validated": False})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    prepare()
