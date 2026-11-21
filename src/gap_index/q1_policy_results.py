"""Recalculate the Q1 supply tables and figure using the approved definitions."""
from __future__ import annotations

import pandas as pd

from ..labeling import q1_human as H
from . import q1_access as A, q1_compare as C, q1_demand as D, q1_final as F
from . import q1_followup, q1_policy as P, q1_supply as S

OUT = F.OUT


def calculate(demand, before):
    after, changes, pending = P.apply(before)
    supply = pd.concat([
        S.supply_by_scenario(frame, rules=["v4_main", "v4_all"]).assign(version=version)
        for version, frame in [("before", before), ("final", after)]
    ], ignore_index=True)
    rank = C.demand_supply_rank
    comparisons = []
    for version, group in supply.groupby("version", sort=False):
        comparable = group.assign(
            in_demand_comparison=group.area.isin(S.DEMAND_AREAS),
            is_reference=group.scenario.eq(C.REFERENCE_SCENARIO) & group.econ_rule.eq(C.REFERENCE_RULE),
            main_spec_status=version,
        )
        comparisons.append(rank(demand, comparable).assign(version=version))
    comparison = pd.concat(comparisons, ignore_index=True)
    strict = supply.query("scenario == 'S2_AB_strict' and econ_rule == 'v4_main'")
    metrics = ["n_services_in_scenario", "n_services", "share_of_scenario", "rank_in_demand_areas"]
    reference = strict.loc[strict.version.eq("before"), ["area", "area_label", *metrics]].merge(
        strict.loc[strict.version.eq("final"), ["area", *metrics]],
        on="area", suffixes=("_before", "_final"), validate="one_to_one",
    )
    need = comparison.query("version == 'final' and is_reference")[
        ["area", "rate_very_needed_4", "demand_rank_4", "rate_needed_3_4", "demand_rank_34"]]
    reference = reference.merge(need, on="area", how="left", validate="one_to_one")

    supplements = []
    education = S.area_matrix(after.drop(columns="q1_education_direct_only"))["q21_07"]
    for scenario, scopes, bases, _ in S.SCENARIOS:
        base, added = S.scenario_membership(after, scenario, scopes, bases)
        counts = after.loc[base & education, "q21_07_beneficiary"].value_counts()
        supplements.append({
            "scenario": scenario, "n_services": int((base | added).sum()), "n_open_space_added": int(added.sum()),
            "n_direct_care": int((base & after.q1_direct_care).sum()),
            "n_recovery_training_cost": int((base & after.q1_recovery_training_cost).sum()),
            "n_age_exception_only": int((base & after.q1_age_exception_only).sum()),
            **{f"education_{key}": int(counts.get(key, 0)) for key in ["self", "both", "dependent", "N"]},
        })

    # Reuse the original service-level access determinations for current area totals.
    home, _ = A.read_verification()
    physical = A.final_verification(A.home_verification(before, home),
                                    A.nonhome_verification(before, A.read_nonhome_verification()),
                                    A.read_physical_review()).set_index("servId")
    _, scopes, bases, _ = next(row for row in S.SCENARIOS if row[0] == C.REFERENCE_SCENARIO)
    base, added = S.scenario_membership(after, C.REFERENCE_SCENARIO, scopes, bases)
    membership = S.area_matrix(after)
    membership.loc[~base, S.AREAS[1:]] = False
    membership.loc[~(base | added), "q21_01"] = False
    membership["any_demand_area"] = membership[S.DEMAND_AREAS].any(axis=1)
    verdict = after.servId.map(physical.strict_status).fillna("not_assessed")
    access = []
    for area in membership:
        hit = membership[area]
        n = int(hit.sum())
        confirmed = int((hit & verdict.eq("home")).sum())
        access.append({
            "area": area, "n_services": n,
            "n_home": confirmed,
            "n_not_home": int((hit & verdict.eq("not_home")).sum()),
            "n_unclear": int((hit & verdict.eq("unclear")).sum()),
            "n_not_assessed": int((hit & verdict.eq("not_assessed")).sum()),
            "share_home": confirmed / n if n else None,
        })
    area_cols = ["servId", "youth_scope", "youth_scope_basis", "need_area_q21", "q21_10_tier", "q21_07_beneficiary",
                 "q1_direct_care", "q1_open_space_universal", "q1_recovery_training_cost", "q1_age_exception_only",
                 "q1_human_reviewed", *H.FIELDS]
    return {
        "final_policy_supply_layer": after[area_cols],
        "final_policy_service_changes": changes,
        "final_policy_unresolved": pending,
        "final_policy_supply": supply,
        "final_policy_demand_supply": comparison,
        "final_policy_reference": reference,
        "final_policy_supplement": pd.DataFrame(supplements),
        "final_policy_access": pd.DataFrame(access),
    }


def figure(reference, path):
    from matplotlib import font_manager, pyplot as plt

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"] if name in fonts),
                "DejaVu Sans")
    data = reference.dropna(subset=["demand_rank_4"]).sort_values("demand_rank_4")
    y = list(range(len(data)))
    with plt.rc_context({"font.family": font, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(11.5, 5.8), layout="constrained")
        left.barh(y, 100 * data.rate_very_needed_4, color="#2a6498")
        left.set_yticks(y, data.area_label)
        left.invert_yaxis()
        left.set(xlabel="매우 필요 응답 (%)", title="은둔 청년의 지원 필요도")
        right.barh([i - .18 for i in y], data.n_services_before, height=.32, color="#9ca8b3", label="변경 전")
        bars = right.barh([i + .18 for i in y], data.n_services_final, height=.32, color="#2a6498", label="최종 기준")
        right.bar_label(bars, padding=3, fontsize=9)
        right.margins(x=.13)
        right.set(xlabel="등록 서비스 (건)", title="엄격 시나리오의 영역별 공급")
        right.legend(loc="upper right")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def save(tables, out=OUT):
    for folder in ["tables", "figures"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / "tables" / f"{name}.csv", **H.CSV)
    figure(tables["final_policy_reference"], out / "figures/06_final_demand_supply.png")


def main():
    master = S.load_master()
    demand = D.demand_by_group(*D.load_demand())
    gold, trace, _ = H.build_gold()
    before = F.overlay(q1_followup.revise(master)[0], gold, trace)
    save(calculate(demand, before))


if __name__ == "__main__":
    main()
