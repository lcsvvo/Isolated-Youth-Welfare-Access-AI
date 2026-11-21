"""보조금24 원본·규칙 재검증과 무손실 v2 입력 생성.

python -m src.preprocess.gov24_preparation [--refresh]
기존 gov24_*.csv, SSIS 마스터, Q2 파일은 읽기만 한다.
AI 검토 기록은 별도 입력이며 사람 정답으로 취급하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import subprocess
import types
from pathlib import Path

import pandas as pd

from ..collect._common import ROOT
from ..collect.gov24_codes import JA_CODES
from . import gov24_eligibility as eligibility
from . import summary_html
from .gov24_youth import YOUTH_MIN, YOUTH_MAX, FOCUS_START_MIN, FOCUS_END_MAX

STAGE = "00_gov24_preparation"
BASE_COMMIT = "0d86a1d"
LEGACY_DETAIL = ["서비스목적", "구비서류", "문의처", "온라인신청사이트URL", "법령", "자치법규", "행정규칙"]
VISIT_CASES = ["138300000066", "314000000272", "O00105200003"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, low_memory=False)


def lf(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)


def valid_ids(frame: pd.DataFrame, name: str) -> None:
    ids = frame["서비스ID"]
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name}: 서비스ID 결측 또는 중복")


def merge_full(lst: pd.DataFrame, detail: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """세 endpoint의 원문을 보존한다. 상세와 목록의 동명 필드는 합치지 않는다."""
    for name, frame in [("목록", lst), ("상세", detail), ("지원조건", conditions)]:
        valid_ids(frame, name)
        if set(frame["서비스ID"]) != set(lst["서비스ID"]):
            raise ValueError(f"{name}: 목록과 서비스ID 집합 불일치")
    det = detail.rename(columns={c: f"detail_{c}" for c in detail if c != "서비스ID"})
    cond = conditions.rename(columns={c: f"{c}_{JA_CODES[c]}" if c in JA_CODES else
                                     "conditions_서비스명" if c == "서비스명" else c for c in conditions})
    out = lst.merge(det, on="서비스ID", validate="one_to_one", sort=False)
    out = out.merge(cond, on="서비스ID", validate="one_to_one", sort=False)
    a0 = pd.to_numeric(out["JA0110_대상연령(시작)"], errors="coerce")
    a1 = pd.to_numeric(out["JA0111_대상연령(종료)"], errors="coerce")
    personal = out["사용자구분"].str.contains("개인")
    out["youth_age_unknown"] = personal & (a0.isna() | a1.isna())
    out["youth_overlap"] = personal & a0.le(YOUTH_MAX) & a1.ge(YOUTH_MIN)
    out["youth_focused"] = out.youth_overlap & a0.ge(FOCUS_START_MIN) & a1.le(FOCUS_END_MAX)
    out["has_online_url"] = out["detail_온라인신청사이트URL"].str.strip().ne("")
    return out


def eligibility_input(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    for c in LEGACY_DETAIL:
        frame[c] = frame[f"detail_{c}"]
    # 기존 추출기의 결측 처리를 유지한다. 원문 저장용 merged 자체는 변경하지 않는다.
    return frame.replace("", float("nan"))


def serialized(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=False)


def differences(old: pd.DataFrame, new: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_ids(old, label + ":old")
    valid_ids(new, label + ":new")
    if set(old["서비스ID"]) != set(new["서비스ID"]):
        raise ValueError(f"{label}: 비교 키 불일치")
    a, b = old.set_index("서비스ID"), serialized(new).set_index("서비스ID")
    b = b.reindex(a.index)
    counts, cells = [], []
    for c in a.columns.intersection(b.columns):
        mask = a[c].ne(b[c])
        counts.append({"comparison": label, "column": c, "changed_rows": int(mask.sum())})
        for sid in a.index[mask]:
            cells.append({"comparison": label, "서비스ID": sid, "column": c,
                          "old": a.loc[sid,c], "new": b.loc[sid,c]})
    return pd.DataFrame(counts), pd.DataFrame(cells, columns=["comparison", "서비스ID", "column", "old", "new"])


def baseline_module():
    source = subprocess.run(["git", "show", f"{BASE_COMMIT}:src/preprocess/gov24_eligibility.py"],
                            cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    module = types.ModuleType("src.preprocess.gov24_eligibility_baseline")
    module.__package__ = "src.preprocess"
    exec(compile(source, f"git:{BASE_COMMIT}:gov24_eligibility.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def legacy_rebuild(module, frame, method):
    text = frame["지원대상"].fillna("") + "\n" + frame["선정기준"].fillna("")
    out = pd.concat([frame[["서비스ID", "서비스명", "소관기관명", "소관기관유형", "youth_overlap", "youth_focused"]],
                     module.age(frame,text), module.income(frame,text), module.application(frame,method)], axis=1)
    out["elig_status"], out["elig_household"] = module.listed(frame,module.STATUS), module.listed(frame,module.HOUSEHOLD)
    out["residence_mentioned"] = text.str.contains(module.RE_RESIDENCE)
    out["needs_review"] = out.elig_income_conflict | out.elig_age_conflict
    out["personal"] = frame["사용자구분"].fillna("").str.contains("개인")
    out["info_known_count"] = sum((out[c] != module.NONE).astype(int) for c in
                                ["elig_age_src", "elig_income_src", "apply_channel", "docs_required"])
    return out


def run(*, refresh: bool = False) -> dict:
    processed = ROOT / "data/processed"
    tables = ROOT / "outputs" / STAGE / "tables"
    products = processed / "gov24_preparation_v2"
    report = ROOT / "outputs" / STAGE / "summary" / f"{STAGE}_summary_final.html"
    manifest_path = ROOT / "logs/validation/gov24_preparation_2026-11-05.json"
    if (manifest_path.exists() or report.exists() or products.exists()) and not refresh:
        raise FileExistsError("기존 생성물이 있습니다. 이 단계가 생성한 파일만 --refresh로 갱신할 수 있습니다.")
    raw = ROOT / "data/raw/gov24"
    raw_paths = list(raw.glob("*"))
    protected = [p for p in raw_paths if p.is_file()] + [processed/n for n in
                   ["gov24_merged.csv", "gov24_youth.csv", "gov24_eligibility.csv", "services_master.csv", "services_master.meta.json"]]
    protected += [p for p in processed.glob("q2*") if p.is_file()]
    protected += [p for p in (ROOT / "logs/validation").glob("q2*/**/*") if p.is_file()]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected if p.exists()}
    frames = {name: read(raw/f"gov24_{name}.csv") for name in ["serviceList", "serviceDetail", "supportConditions"]}
    lst, det, cond = (frames[n] for n in ["serviceList", "serviceDetail", "supportConditions"])
    merged = merge_full(lst, det, cond)
    collection = json.loads((raw/"_collection.json").read_text(encoding="utf-8"))
    for name, frame in frames.items():
        if len(frame) != collection["endpoints"][name]["rows"]:
            raise ValueError(f"{name}: 수집 기록과 행 수 불일치")
    logged = {p["name"]:p for p in collection["files"]}
    hash_rows, quality = [], []
    for name, frame in frames.items():
        p = raw/f"gov24_{name}.csv"
        log = logged[p.name]
        hash_rows.append({"file": p.name, "rows":len(frame), "columns":len(frame.columns), "bytes":p.stat().st_size,
                          "sha256":sha(p), "collection_sha256":log["sha256"], "collection_hash_match":sha(p)==log["sha256"],
                          "collection_bytes":log["bytes"], "CR_bytes":p.read_bytes().count(b"\r")})
        for c in frame:
            quality.append({"endpoint":name, "column":c, "rows":len(frame),
                            "missing_rows":int(frame[c].str.strip().eq("").sum())})
    old_merged = read(processed/"gov24_merged.csv")
    old_elig = read(processed/"gov24_eligibility.csv")
    old_merged = old_merged.set_index("서비스ID").loc[lst["서비스ID"]].reset_index()
    reconstructed = old_merged[list(lst.columns)].to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
    reconstructed_match = hashlib.sha256(reconstructed).hexdigest() == logged["gov24_serviceList.csv"]["sha256"]
    list_normalized_diff = sum(int(lf(lst[c]).ne(lf(old_merged[c])).sum()) for c in lst.columns)
    detail_checks = []
    original = det.set_index("서비스ID")
    full = merged.set_index("서비스ID")
    historical = old_merged.set_index("서비스ID")
    for c in det.columns:
        if c == "서비스ID":
            continue
        current_diff = int(original[c].ne(full.loc[original.index,f"detail_{c}"]).sum())
        comparable = c in LEGACY_DETAIL
        hist_diff = int(lf(original[c]).ne(lf(historical.loc[original.index,c])).sum()) if comparable else None
        detail_checks.append({"field":c, "current_raw_to_v2_changed_cells":current_diff,
                              "historical_reference":"legacy merged" if comparable else "원수집 스냅샷 미보관",
                              "historical_normalized_changed_cells":hist_diff,
                              "missing_rows":int(original[c].str.strip().eq("").sum())})
    method = old_merged["서비스ID"].map(original["신청방법"])
    old_input = old_merged.replace("", float("nan"))
    base, base_hash = baseline_module()
    base_rebuilt = legacy_rebuild(base, old_input, method)
    fixed_on_legacy = eligibility.build_eligibility(old_input,method)
    new_elig = eligibility.build_eligibility(eligibility_input(merged),merged["detail_신청방법"])
    diff_counts, diff_cells = [], []
    for a,b,label in [(old_elig,base_rebuilt,"stored_vs_baseline_code"),
                      (serialized(base_rebuilt),fixed_on_legacy,"baseline_vs_fixed_same_inputs"),
                      (old_elig,new_elig,"stored_vs_v2_current_raw")]:
        c,d = differences(a,b,label)
        diff_counts.append(c); diff_cells.append(d)
    visit_rows=[]
    for sid in VISIT_CASES:
        i = old_merged.index[old_merged["서비스ID"].eq(sid)][0]
        visit_rows.append({"서비스ID":sid,"서비스명":old_merged.loc[i,"서비스명"],"method_raw":method.iloc[i],
                           "baseline_visit_text":bool(base_rebuilt.loc[i,"apply_visit_text"]),
                           "fixed_visit_text":bool(fixed_on_legacy.loc[i,"apply_visit_text"]),
                           "stored_visit_text":old_elig.set_index("서비스ID").loc[sid,"apply_visit_text"],
                           "final_channel_changed":base_rebuilt.loc[i,"apply_channel"]!=fixed_on_legacy.loc[i,"apply_channel"]})
    ai_path = ROOT/"logs/validation/gov24_ai_review_2026-11-05.json"
    boundary_path = ROOT/"logs/validation/gov24_boundary_ai_review_2026-11-05.json"
    ai = pd.DataFrame(json.loads(ai_path.read_text(encoding="utf-8"))["judgments"])
    expected_unknown = set(old_elig.loc[old_elig.personal.eq("True") & old_elig.apply_channel.eq(eligibility.NONE),"서비스ID"])
    valid_ids(ai,"AI review")
    if set(ai["서비스ID"]) != expected_unknown:
        raise ValueError("AI 검토 ID가 기존 개인 미확인 집단과 불일치")
    for row in ai.to_dict("records"):
        text = original.loc[row["서비스ID"],"신청방법"]
        if hashlib.sha256(text.encode()).hexdigest()!=row["detail_method_sha256"] or text!=row["method_raw"]:
            raise ValueError("AI 검토 이후 원문 변경: " + row["서비스ID"])
    ai_index = ai.set_index("서비스ID")
    new_elig["apply_channel_ai_candidate"] = new_elig["서비스ID"].map(ai_index.ai_channel_candidate).fillna("")
    new_elig["apply_channel_ai_reason"] = new_elig["서비스ID"].map(ai_index.ai_reason).fillna("")
    new_elig["apply_channel_ai_evidence"] = new_elig["서비스ID"].map(ai_index.method_raw).fillna("")
    new_elig["rule_version"] = eligibility.RULE_VERSION
    new_elig["human_verified"] = False
    boundary = pd.DataFrame(json.loads(boundary_path.read_text(encoding="utf-8"))["judgments"])
    prior = json.loads((ROOT/"logs/validation/gov24_eligibility_review_2026-10-31.json").read_text(encoding="utf-8"))
    prior_ids=set()
    def walk(obj):
        if isinstance(obj,dict):
            if "서비스ID" in obj: prior_ids.add(obj["서비스ID"])
            for value in obj.values(): walk(value)
        elif isinstance(obj,list):
            for value in obj: walk(value)
    walk(prior)
    assert not set(boundary["서비스ID"]) & prior_ids
    for row in boundary.to_dict("records"):
        sid=row["서비스ID"]
        text=historical.loc[sid,"지원대상"]+"\n"+historical.loc[sid,"선정기준"]
        for c in ["evidence","additional_evidence"]:
            assert not row[c] or row[c] in text, (sid,c)
    boundary["target_raw"] = boundary["서비스ID"].map(original["지원대상"])
    boundary["criteria_raw"] = boundary["서비스ID"].map(original["선정기준"])
    unknown_after = int((new_elig.personal & new_elig.apply_channel.eq(eligibility.NONE)).sum())
    candidates = int(ai.ai_channel_candidate.ne(eligibility.NONE).sum())
    counts = pd.DataFrame([
        {"metric":"all_services","n":len(merged)},
        {"metric":"personal_services","n":int(new_elig.personal.sum())},
        {"metric":"youth_overlap_provisional","n":int(merged.youth_overlap.sum())},
        {"metric":"legacy_personal_channel_unknown","n":len(ai)},
        {"metric":"v2_rule_personal_channel_unknown","n":unknown_after},
        {"metric":"ai_reviewed_legacy_unknown","n":len(ai)},
        {"metric":"ai_channel_candidates","n":candidates},
        {"metric":"ai_channel_still_unspecified","n":len(ai)-candidates},
        {"metric":"ai_new_boundary_review","n":len(boundary)},
        {"metric":"independent_human_judgments","n":0},
    ])
    # Q2는 읽기 전용 메타데이터로 확인하며 기존 128열 마스터를 재생성하지 않는다.
    ref=ROOT/"logs/validation/gov24_q2_reference_2026-11-05.json"
    q2={}
    if ref.exists():
        snapshots=json.loads(ref.read_text(encoding="utf-8"))["snapshots"]
        qmeta=snapshots["data/processed/q2_match.meta.json"]["metadata"]
        q2={"source_branch":"feature/q2-cross-platform-match", "snapshot_meta_sha256":sha(ref),
            "match_created_at":qmeta["created_at"], "counts":qmeta["rows"],
            "raw_input_hashes_match":{p:sha(ROOT/p)==h for p,h in qmeta["inputs_sha256"].items() if p.startswith("data/raw/gov24/")},
            "ssis_columns":snapshots["data/processed/services_master.meta.json"]["metadata"]["cols"],
            "candidate_key_membership":"미검증: 연결 CSV 대용량 파일은 connector에서 내용 미반환. 메타데이터로 입력 해시·건수만 확인."}
    for p in [tables,products]: p.mkdir(parents=True,exist_ok=True)
    def save(frame,name): frame.to_csv(tables/f"{name}.csv",index=False,encoding="utf-8-sig",lineterminator="\n")
    save(pd.DataFrame(hash_rows),"raw_integrity")
    save(pd.DataFrame(quality),"field_missingness")
    save(pd.DataFrame(detail_checks),"detail_field_validation")
    save(pd.concat(diff_counts,ignore_index=True),"code_csv_difference_counts")
    save(pd.concat(diff_cells,ignore_index=True),"code_csv_changed_cells")
    save(pd.DataFrame(visit_rows),"visit_regex_regression")
    save(ai,"unknown_channel_ai_review")
    save(boundary,"boundary_ai_review")
    save(counts,"preparation_counts")
    # 사람 판정 열은 별도 파일에 두고 재실행 시 기존 기입을 지우지 않는다.
    blind=ai[["서비스ID","서비스명","method_raw"]].copy()
    for c in ["reviewer_1_channel","reviewer_1_evidence","reviewer_2_channel","reviewer_2_evidence","adjudicated_channel"]: blind[c]=""
    human_path=tables/"channel_human_review.csv"
    if not human_path.exists(): blind.to_csv(human_path,index=False,encoding="utf-8-sig")
    merged.to_csv(products/"gov24_merged_v2.csv",index=False,encoding="utf-8-sig",lineterminator="\n")
    new_elig.to_csv(products/"gov24_eligibility_v2.csv",index=False,encoding="utf-8-sig",lineterminator="\n")
    # 저장 후 상세 20개 항목과 ID를 현재 원본과 다시 대조한다.
    disk=read(products/"gov24_merged_v2.csv").set_index("서비스ID")
    for c in det.columns:
        if c!="서비스ID": assert original[c].equals(disk.loc[original.index,f"detail_{c}"].rename(c))
    after = {p:sha(ROOT/p) for p in before}
    assert before==after, "보호 대상 파일이 변경됐습니다."
    manifest={"stage":STAGE,"rule_version":eligibility.RULE_VERSION,"date":"2026-11-05",
              "source_collected_at":collection["collected_at"],"python":platform.python_version(),"pandas":pd.__version__,
              "baseline_commit":BASE_COMMIT,"baseline_code_sha256":base_hash,
              "code_sha256":{str(p.relative_to(ROOT).as_posix()):sha(p) for p in [Path(__file__),ROOT/"src/preprocess/gov24_eligibility.py"]},
              "protected_sha256":before,"protected_unchanged":before==after,
              "list_reconstruction_matches_collection_sha256":reconstructed_match,
              "list_normalized_changed_cells":list_normalized_diff,
              "legacy_merged_CR_bytes":(processed/"gov24_merged.csv").read_bytes().count(b"\r"),
              "previous_reconstruction_record":"logs/validation/gov24_hash_check_2026-11-02.json; 당시 재구성 성공 기록이며 현재 LF 병합본으로 그 바이트 해시를 다시 재현하지 못함",
              "historical_detail_fields_comparable":len(LEGACY_DETAIL),"historical_detail_fields_unverified":len(det.columns)-1-len(LEGACY_DETAIL),
              "ai_review_input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in [ai_path,boundary_path]},
              "counts":dict(zip(counts.metric,counts.n.astype(int))),"q2":q2,
              "unresolved":["현재 기존 병합본도 셀 안 CRLF가 LF로 정규화되어 이전 목록 재구성 해시 일치 재현 불가. 목록 22열 정규화 값 차이는 0",
                            "수집 당시 상세 13개 필드의 원본 스냅샷 미보관: 완전 원본 일치 미검증",
                            "AI 판정은 사람 정답·독립 2인 검증이 아님. 정밀도·재현율 산출 보류",
                            "연령 다중 대상군·소득 구간과 예외의 단일 값 자동 자격 판정 금지"],
              "outputs_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in list(products.glob("*.csv"))+list(tables.glob("*.csv"))},
              "overall_status":"부분 완료: 기계적 검증·AI 검토 완료, 역사 원본·사람 독립 검증 미완료"}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    def table(name,title,changed_only=False):
        p=tables/f"{name}.csv"
        frame=pd.read_csv(p,keep_default_na=False)
        if changed_only: frame=frame[frame.changed_rows.gt(0)]
        return ("table",frame,title,p.relative_to(ROOT).as_posix())
    summary_html.build(report,title="보조금24 전처리 재검증",lead=f"전체 {len(merged):,}건 보존 · 미확인 {len(ai)}건 AI 검토 · 추가 경로 후보 {candidates}건",
        summary="수집 시점 2026-10-31. 현재 상세정보 20개 비키 필드를 빠짐없이 보존한 v2를 별도 저장했다. 기존 원본·파생 데이터·SSIS·Q2 파일은 변경하지 않았다. AI 후보는 잠정 판정이며 사람 검수 정확도와 구별한다.",
        meta=["2026-11-05 실행",f"규칙 {eligibility.RULE_VERSION}","서비스 1행 · 비가중 건수","부분 완료 · 사람 독립 검증 대기"],
        sections=[
            {"id":"results","title":"핵심 결과와 범위","blocks":[table("preparation_counts","표 1. 처리 범위와 검토 건수"),
              ("p",f"관찰: 기존 미확인 {len(ai)}건 중 {candidates}건에 신청경로 후보를 제시했고 {len(ai)-candidates}건은 미확인을 유지했다. 원문 문의처만으로 전화 신청을 추정하지 않았다. AI 후보는 apply_channel_ai_candidate에 따로 보존하며 규칙 값을 덮어쓰지 않는다.")]},
            {"id":"integrity","title":"원자료와 상세 필드 보존","blocks":[table("raw_integrity","표 2. 수집 기록 해시와 현재 파일"),
              table("detail_field_validation","표 3. 상세 전체 20개 비키 필드 검증 범위"),
              ("box","warning",f"현재 병합본을 사용한 수집 당시 목록 재구성 해시 일치: {reconstructed_match}. 11월 2일 성공 기록은 보존했지만 현재 남은 병합본도 내부 CRLF가 LF로 정규화돼 원 바이트 해시를 재현하지 못한다. 목록 22열의 줄바꿈 정규화 값 차이는 {list_normalized_diff}개다. 상세는 기존 병합본의 7개 항목만 정규화 후 대조 가능하며, 나머지 13개 항목의 수집 당시 원본 완전 일치는 미검증이다. 현재 원본→v2 전체 상세 필드 저장 일치와 구분한다.")]},
            {"id":"rules","title":"코드와 기존 CSV · 방문 3건","blocks":[table("visit_regex_regression","표 4. 웹사이트 다음 줄 실제 방문 안내 회귀 검증"),
              table("code_csv_difference_counts","표 5. 차이가 있는 열: 기존 파일·기준 코드·수정 코드",changed_only=True),
              ("p","웹사이트 뒤의 공백 매칭이 줄바꿈을 넘어 다음 방문 단어까지 제거하던 오류를 수정했다. 별도 신청이라는 문구만으로 신청 불필요로 판정하던 규칙도 명시적 부정 표현이 있을 때로 제한했다. 최종 채널 변경과 근거·중간 플래그 변경을 분리해 셀 단위 표에 저장했다.")]},
            {"id":"ai","title":"AI 원문 검토 · 한계","blocks":[table("boundary_ai_review","표 6. 기존 검토와 겹치지 않는 경계 16건 AI 재검토"),
              ("p","신청경로 미확인 전수 AI 검토는 unknown_channel_ai_review.csv에 원문·판정·사유를 기록했다. 눈가림 사람용 channel_human_review.csv는 AI 답안을 포함하지 않는다. 연령 표본은 자녀 연령·복수 사업·구매/발급 단계를, 소득 표본은 구간 경계·일부 선발인원 예외·사업별 조건을 구분했다."),
              ("box","warning","사람 정답이 없으므로 정밀도·재현율·일치도를 산출하지 않는다. 위험 기반 신규 표본 16건은 모집단 오류율 추정용이 아니다. AI가 2인 독립 사람 검수를 대체한 것으로 표시하지 않는다.")]},
            {"id":"next","title":"Q2 연결과 남은 검증","blocks":[
              ("p","Q2 메타데이터에서 연결 후보 2,215건, SSIS 128열 입력과 보조금24 원자료 해시 일치를 확인했다. 후보 CSV의 ID별 포함 관계는 커넥터의 대용량 내용 미반환으로 미검증이다. 수집 시점과 Q2 생성 시점은 검증 JSON에 각각 기록했다. 후보는 확정 동일사업이 아니다."),
              ("p","남은 작업: 상세 13개 항목의 수집 당시 원본 확보, 사람 2인 독립 판정 및 합의, 검수 정답 기반 성능 평가. 새 API 응답으로 과거 원본을 대체하지 않는다. 기존 Q2 매칭·골드셋과 128열 입력을 재생성하지 않았다."),
              ("box","warning","보고서 브라우저 렌더링·오프라인 화면·좁은 화면·인쇄 미리보기는 미검증이다. 컴퓨터 사용 도구가 현재 브라우저 URL을 확인하지 못해 화면 검증을 중단했다. HTML 구조·자원·앵커의 정적 검증과 구별한다.")]},
        ],footer=f"근거: notebooks/{STAGE}.ipynb · outputs/{STAGE}/tables/*.csv · logs/validation/gov24_preparation_2026-11-05.json. 그림 해당 없음: 필드·버전·사례 검증은 표로 보고. 미검증 항목을 완료로 처리하지 않음.")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh",action="store_true",help="이 단계가 생성한 v2·표·요약만 갱신. 사람 판정 시트는 보존")
    args=parser.parse_args()
    result=run(refresh=args.refresh)
    print(json.dumps({"status":result["overall_status"],"counts":result["counts"]},ensure_ascii=False,indent=2))
