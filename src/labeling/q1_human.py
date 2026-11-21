"""Q1 1차 사람 정답과 원판정 대조. 입력 CSV는 읽기 전용."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..collect._common import ROOT
from ..gap_index.q1_supply import AREAS, area_matrix
from .evaluate import ordinal_agreement, q1_multilabel_f1

REVIEW = ROOT / "logs/validation/q1_round1_review"
KEY = ["item_no", "servId"]
YES, NO = "예", "아니오"
AREA_LABELS = [f"h_{q}" for q in AREAS]
YOUTH, APPLY, OUTING = "h_youth_target", "h_apply_barrier", "h_use_outing"
ATTRIBUTES = [YOUTH, APPLY, OUTING]
FIELDS = AREA_LABELS + ATTRIBUTES
UNKNOWN = {**dict.fromkeys(AREA_LABELS, "불명"), YOUTH: "확인필요", APPLY: "N", OUTING: "N"}
LEVELS = {**dict.fromkeys(AREA_LABELS, [NO, YES, "불명"]),
          YOUTH: [NO, YES, "확인필요"], APPLY: ["0", "1", "2", "3", "N"], OUTING: ["0", "1", "2", "3", "N"]}
EVIDENCE = {**dict.fromkeys(AREA_LABELS, "h_q21_evidence"), YOUTH: "h_youth_evidence",
            APPLY: "h_apply_evidence", OUTING: "h_use_outing_evidence"}
REVIEWERS = ["R1", "R2", "R3", "R4"]
CSV = {"index": False, "encoding": "utf-8-sig", "lineterminator": "\n"}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_gold(folder: Path = REVIEW) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """배정 키와 실제 불일치 키가 원본·합의안과 완전히 일치할 때만 통합."""
    assignment_path = folder / "review_assignment.csv"
    resolution_path = folder / "disagreement_resolution.csv"
    paths = [folder / "human" / f"q1_review_{r}.csv" for r in REVIEWERS]
    assign = read_csv(assignment_path)
    require(assign[KEY].ne("").all().all() and assign["item_no"].is_unique and assign["servId"].is_unique,
            "배정표 item_no/servId 공란 또는 중복")
    expected = assign.melt(id_vars=KEY, value_vars=["reviewer_primary", "reviewer_second"],
                           var_name="role", value_name="reviewer").query("reviewer != ''")
    require(assign.reviewer_primary.isin(REVIEWERS).all() and expected.reviewer.isin(REVIEWERS).all()
            and not expected.duplicated(KEY + ["reviewer"]).any(), "검수 배정 오류")
    originals = pd.concat([read_csv(path).assign(reviewer=r, file=path.name)
                           for path, r in zip(paths, REVIEWERS)], ignore_index=True)
    join_key = KEY + ["reviewer"]
    actual = expected.merge(originals, on=join_key, how="outer", validate="one_to_one", indicator=True)
    require(actual["_merge"].eq("both").all(), "배정표와 R1~R4의 검수 키 불일치")
    for label in FIELDS:
        require(actual[label].isin(LEVELS[label]).all(), f"{label}: 공란 또는 허용 범위 밖 판정")

    primary = actual[actual.role.eq("reviewer_primary")].set_index(KEY)
    second = actual[actual.role.eq("reviewer_second")].set_index(KEY)
    resolution = read_csv(resolution_path).rename(columns={"불일치항목": "label"}).set_index(KEY + ["label"],
                                                                                         verify_integrity=True)
    audit = []
    used = set()
    for key, first in primary.iterrows():
        other = second.loc[key] if key in second.index else None
        for label in FIELDS:
            before = first[label]
            other_value = other[label] if other is not None else ""
            disagreement = other is not None and before != other_value
            field_key = (*key, label)
            evidence = EVIDENCE[label]
            row = dict(zip(KEY, key))
            row.update(label=label, servNm=first.servNm, reviewer_primary=first.reviewer,
                       primary_value=before, primary_file=first.file, primary_evidence=first[evidence],
                       primary_note=first.h_note, reviewer_second=other.reviewer if other is not None else "",
                       second_value=other_value, second_file=other.file if other is not None else "",
                       second_evidence=other[evidence] if other is not None else "",
                       second_note=other.h_note if other is not None else "",
                       source="single_review" if other is None else "reviewer_agreement",
                       status="", rationale="", official_source="", source_quote="", source_note="",
                       review_url=first.detail_url, resolution_file="")
            final = before
            if disagreement:
                require(field_key in resolution.index, f"미해결 불일치 {field_key}")
                decision = resolution.loc[field_key]
                original_values = {first.reviewer: before, other.reviewer: other_value}
                recorded = {decision["검수자1"]: decision["원판정1"], decision["검수자2"]: decision["원판정2"]}
                require(recorded == original_values, f"합의안 원판정 불일치 {field_key}")
                final = decision["합의안"]
                require(final in LEVELS[label] and decision["상태"] in {"사용자 결정", "공식자료 확인", "제공 기준 적용"}
                        and decision["판정근거"].strip(), f"합의안 상태·판정·근거 누락 {field_key}")
                row.update(source="resolution", status=decision["상태"], rationale=decision["판정근거"],
                           official_source=decision["공식출처"], review_url=decision["복지로원문"] or first.detail_url,
                           source_quote=decision["원문발췌"], source_note=decision["추가메모"],
                           resolution_file=resolution_path.name)
                used.add(field_key)
            row.update(final_value=final, changed_from_primary=final != before,
                       changed_from_second=other is not None and final != other_value,
                       final_unknown=final == UNKNOWN[label])
            audit.append(row)
    require(used == set(resolution.index), "원판정 불일치와 대응하지 않는 합의 행")
    trace = pd.DataFrame(audit)
    require(not (trace.changed_from_primary & trace.source.ne("resolution")).any(), "합의안 밖 판정 변경")
    gold = (trace.pivot(index=KEY, columns="label", values="final_value")[FIELDS]
            .rename_axis(columns=None).reset_index())
    gold = assign.merge(gold, on=KEY, validate="one_to_one")
    log = {
        "assignment_rows": len(assign), "review_rows": len(originals), "unique_services": gold.servId.nunique(),
        "double_reviewed": len(second), "disagreement_services": len({key[1] for key in used}),
        "disagreement_labels": len(used), "resolutions_applied": len(used), "unresolved": 0,
        "duplicate_servId": int(gold.servId.duplicated().sum()),
        "missing_labels": int(gold[FIELDS].eq("").sum().sum()),
        "unexpected_changes": int((trace.changed_from_primary & trace.source.ne("resolution")).sum()),
        "input_sha256": {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [assignment_path, resolution_path, *paths]},
    }
    return gold, trace, log


def predictions(m: pd.DataFrame) -> pd.DataFrame:
    """기존 v4_main 영역 후보와 명세의 elig_youth·신청·이용 값을 비교용으로 정규화."""
    values = area_matrix(m, "v4_main").replace({True: YES, False: NO}).add_prefix("h_")
    values[YOUTH] = m.elig_youth.fillna("N").replace("N", "확인필요")
    values[APPLY] = m.apply_barrier.fillna("N")
    values[OUTING] = m.use_outing.fillna("N")
    for field in ATTRIBUTES:
        require(values[field].isin(LEVELS[field]).all(), f"{field}: 자동 라벨 범위 오류")
    values["servId"] = m.servId
    return values.set_index("servId")


def pair_metrics(ref: pd.Series, pred: pd.Series, field: str) -> dict:
    counts = pd.crosstab(ref, pred).reindex(index=LEVELS[field], columns=LEVELS[field], fill_value=0)
    n = int(counts.to_numpy().sum())
    correct = int(np.trace(counts))
    chance = float(counts.sum(axis=0) @ counts.sum(axis=1)) / n**2 if n else np.nan
    both = counts.iloc[:-1, :-1]
    n_both = int(both.to_numpy().sum())
    known_ref = int(counts.iloc[:-1].to_numpy().sum())
    known_pred = int(counts.iloc[:, :-1].to_numpy().sum())
    metrics = {
        "n_total": n, "n_reference_unknown": n - known_ref, "n_prediction_unknown": n - known_pred,
        "reference_unknown_rate": (n - known_ref) / n if n else np.nan,
        "prediction_unknown_rate": (n - known_pred) / n if n else np.nan,
        "n_both_known": n_both, "n_excluded_known_only": n - n_both,
        "excluded_reason": f"either value is {UNKNOWN[field]}",
        "accuracy_including_unknown": correct / n if n else np.nan,
        "kappa_including_unknown": (correct / n - chance) / (1 - chance) if n and chance < 1 else np.nan,
        "accuracy_both_known": np.trace(both) / n_both if n_both else np.nan,
    }
    if field in (APPLY, OUTING):
        pairs = pd.DataFrame({"servId": range(len(ref)), "human": ref.to_numpy(), "rule": pred.to_numpy()})
        metrics["quadratic_kappa_both_known"] = ordinal_agreement(pairs, pairs, "rule", "human")["qwk"]
    return metrics


def evaluate(gold: pd.DataFrame, trace: pd.DataFrame, models: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """보정 전 자동 규칙만 평가. 미분류 자동 영역은 후보 미제시로 비교하며 사람 불명은 제외."""
    human = gold.set_index("servId")[FIELDS]
    dev = gold.set_index("servId").known_development_overlap
    split = {"all": human.index, "known_development": dev.index[dev], "no_known_overlap": dev.index[~dev]}
    area_rows, aggregate, attributes, confusion = [], [], [], []
    for version, m in models.items():
        pred = predictions(m).loc[human.index]
        rule_no_tags = m.set_index("servId").loc[human.index, "need_area_q21"].fillna("N").eq("N")
        for cohort, idx in split.items():
            h, p = human.loc[idx], pred.loc[idx]
            positive_tags = p[AREA_LABELS].eq(YES).apply(
                lambda row: "|".join(q for q in AREAS if row[f"h_{q}"]), axis=1)
            key = positive_tags.rename("need_area_q21").reset_index()
            scores = q1_multilabel_f1(h.reset_index(), key, AREAS)
            scores["n_total"] = len(h)
            scores["n_human_unknown"] = len(h) - scores.n_labeled
            scores["human_unknown_rate"] = scores.n_human_unknown / len(h)
            scores["n_auto_unclassified_services"] = int(rule_no_tags.loc[idx].sum())
            scores["auto_unclassified_service_rate"] = rule_no_tags.loc[idx].mean()
            scores["n_human_positive"] = scores.tp + scores.fn
            scores["excluded_reason"] = "human area label is 불명"
            # 사람 양성이 없으면 영역의 검출 성능을 식별할 수 없음. 혼동 건수는 보존.
            scores.loc[scores.n_human_positive.eq(0), ["recall", "f1"]] = np.nan
            scores["interpretation"] = np.select([scores.n_human_positive.eq(0), scores.n_human_positive.lt(5)],
                                                  ["no_human_positives", "fewer_than_five_positives"], default="descriptive")
            aggregate.append({"model": version, "cohort": cohort, "n_services": len(h),
                              "n_label_pairs": int(scores.n_labeled.sum()),
                              "n_excluded_human_unknown": int(scores.n_human_unknown.sum()),
                              "n_macro_areas": int(scores.f1.notna().sum()),
                              "micro_f1": scores.attrs["micro_f1"], "macro_f1": scores.f1.mean()})
            area_rows.append(scores.assign(model=version, cohort=cohort))
            for field in ATTRIBUTES:
                attributes.append({"model": version, "cohort": cohort, "label": field,
                                   **pair_metrics(h[field], p[field], field)})
                counts = pd.crosstab(h[field], p[field]).reindex(
                    index=LEVELS[field], columns=LEVELS[field], fill_value=0)
                confusion.extend({"model": version, "cohort": cohort, "label": field, "human": x, "automatic": y,
                                  "n_services": int(counts.loc[x, y])} for x in counts.index for y in counts.columns)

    doubles = trace[trace.reviewer_second.ne("")]
    reviewers = pd.DataFrame([{"label": field, **pair_metrics(g.primary_value, g.second_value, field)}
                              for field, g in doubles.groupby("label", sort=False)])
    changes = (trace.assign(n=1, double_reviewed=trace.reviewer_second.ne(""), disagreement=trace.source.eq("resolution"))
               .groupby("label", sort=False)[["n", "double_reviewed", "disagreement", "changed_from_primary",
                                             "changed_from_second", "final_unknown"]].sum().reset_index())
    transitions = (trace.groupby(["label", "primary_value", "final_value"], sort=False)
                   .size().rename("n_services").reset_index())
    return {"human_auto_area_metrics": pd.concat(area_rows, ignore_index=True),
            "human_auto_f1_summary": pd.DataFrame(aggregate), "human_auto_attribute_metrics": pd.DataFrame(attributes),
            "human_auto_confusion": pd.DataFrame(confusion), "human_interrater": reviewers,
            "human_label_changes": changes, "human_label_transitions": transitions}
