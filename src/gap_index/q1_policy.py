"""Apply the approved Q1 definitions to the existing reviewed analysis layer."""
from __future__ import annotations

import pandas as pd

from .q1_audit_impact import put

VERSION = "q1-final-policy-20261118"
SELF_BENEFICIARIES = {"self", "both"}

MONTHLY = r"매\s*월|달마다|월\s*별|(?:\(인,\s*월\)|/\s*월|월\s*[0-9]|[0-9]+\s*개월|×\s*12월)"
CHILD_ALLOWANCE = r"양육\s*보조금|가정\s*위탁.{0,12}양육\s*(?:비|수당)|(?:입양|보호)\s*아동.{0,12}양육\s*수당"
RETURN_FUNERAL = r"(?:귀향|귀가|귀국)\s*(?:여비|교통비)|장제\s*비|(?:행려|무연고).{0,12}장례\s*비"
LIVING = r"생계\s*(?:비|지원|급여)|생활\s*(?:비|보조비|보조금|안정\s*지원금)|월동\s*대책비|공과금|전기\s*요금"
WORK_INSURANCE = (r"(?:농업인|어업인).{0,6}안전\s*(?:보험|공제)|어선.{0,16}재해.{0,8}보험|"
                  r"(?:농작물|가축|농기계|양식수산물).{0,12}(?:재해\s*)?보험|고용.{0,5}보험료|"
                  r"산재\s*(?:보험료|보험\s*지원)|정책\s*보험료|사업주.{0,18}사회\s*보험료|두루누리")
HOUSEHOLD_INSURANCE = r"국민\s*건강\s*보험료|건강\s*보험료|장기\s*요양\s*보험료"
DISASTER = r"이재민|재해\s*구호|재난\s*구호"
RELIEF_ITEMS = r"식료품|급식\s*또는\s*식품|생필품|생활\s*필수품|담요|침구|생수|응급\s*구호\s*세트|취사\s*구호\s*세트"
RELIEF_GOODS = r"구호\s*(?:물자|물품|세트)|생활\s*필수품"
ADMISSION = r"입학\s*축하금"
EDU_USE = (r"학비|수업료|등록금|학자금|수강료|교재|학용품|학습용|교복|"
           r"교육\s*(?:비|경비).{0,16}(?:지원|부담|경감|완화)|교육에\s*소요되는.{0,12}부담|"
           r"대학\s*진학.{0,20}(?:경비|비용)")
EDU_LEARNING = (r"수강|수업|등록금|학습\s*(?:비|지원|프로그램)|교재|학원비|"
                r"교육\s*(?:비|과정|프로그램)|강의|학비|직업\s*훈련")
EXAM = r"응시\s*(?:료|비)|시험\s*응시\s*(?:수수료|비용)"
SCHOOL_EQUIVALENCY = rf"검정\s*고시.{{0,16}}(?:{EXAM})|(?:{EXAM}).{{0,12}}검정\s*고시"
DEBT = r"채무\s*조정|연체\s*자|부실\s*채무|신용\s*(?:회복|유의자)|조기\s*상환|분할\s*상환\s*약정"
RECOVERY = (r"(?:생활\s*(?:리듬|습관)|자기\s*관리).{0,16}(?:개선|회복|훈련|교육|실습)|"
            r"(?:독립|자립)\s*생활\s*(?:기술\s*)?훈련|일상\s*생활\s*(?:수행\s*)?훈련|"
            r"(?:지역사회\s*복귀|사회\s*적응)\s*훈련")
TRAINING_COST = rf"(?:{RECOVERY})\s*(?:비용|비\s*지원|비\s*지급)"
MEDICAL_REHAB = r"공공\s*재활\s*(?:프로그램|의료)|재활\s*치료|물리\s*치료"
CARE = (r"가사\s*(?:활동\s*지원|대행|서비스|도우미|지원)|간병\s*서비스|방문\s*(?:요양|목욕|돌봄|간호)|"
        r"(?:병원|의료기관|관공서|진료|외출)\s*동행|활동\s*보조\s*(?:서비스|지원)|"
        r"(?:식사|밑반찬|도시락|부식)\s*(?:를\s*)?(?:제공|배달|배달서비스)|"
        r"(?:급식|숙식)\s*(?:제공|지원)(?!\s*(?:비|카드|상품권))|"
        r"(?:가사|간병).{0,12}(?:이용\s*권|바우처)")
CARE_PAYMENT_ONLY = r"(?:간병|돌봄|가사)\s*(?:비용|비|본인부담금).{0,12}(?:지원|보전|지급)"
CARE_USE_PROOF = (r"(?:이용|간병|돌봄)\s*(?:실적|확인서|내역|영수증)|(?:서비스|돌봄)\s*이용\s*확인|"
                  r"(?:직접|실제|개인이|본인이)\s*(?:지불|납부|부담)한\s*간병비|"
                  r"간병비.{0,40}(?:영수증|실비|이용\s*시간에\s*따라)|이용\s*실적.{0,20}(?:보고|제출)")
OPEN_SPACE = r"공간|쉼터|휴게\s*소|편의점|라운지|커뮤니티\s*(?:실|룸)"
OPEN_PUBLIC = (r"(?:시민|주민|국민|지역민)\s*(?:은\s*)?누구\s*(?:에게)?나.{0,35}(?:방문|이용|개방)|"
               r"(?:공간|쉼터).{0,20}(?:모든\s*시민|전\s*시민).{0,20}(?:방문|이용|개방)")
AGE_EXCEPTIONS = (r"(?:60|65)\s*세\s*이상.{0,220}(?:다만|단).{0,100}(?:인정|심사|결정).{0,50}예외(?:적|로)|"
                  r"운영위원회.{0,25}(?:심사|결정).{0,30}(?:60|65)\s*세\s*미만.{0,15}입소\s*가능")
BROAD_VOUCHER = r"(?:모든|전)\s*업종|(?:식료품|생필품|생활\s*필수품).{0,30}(?:구매|구입|사용)"
BUSINESS_VOUCHER = r"(?:농자재|영농\s*자재|사업\s*운영비|영농\s*비용).{0,18}(?:만\s*사용|한하여|한정|전용)"
HOUSING_REFERRAL = (r"(?:병원|쉼터|시설|보호소)(?:이나|\s*등|에|로|으로|를)?\s*"
                    r"(?:입소\s*(?:의뢰|안내|연계|조치)|이송|이송\s*조치)|"
                    r"(?:숙소|주거|입소)\s*(?:연계|의뢰|안내)|귀향\s*(?:여비|교통비)")
HOUSING_OFFER = (r"(?:숙식|숙박|거처|주거\s*공간|숙소|거주시설|주택|임시\s*보장시설).{0,12}(?:제공|이용|지원)|"
                 r"(?:숙박비|숙식비|투숙비|월세|임차료|보증금|주거비).{0,12}(?:지원|지급|융자|제공|실비)|"
                 r"일시\s*보호소에서\s*보호|보호소에서.{0,15}숙식")


def apply(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a new layer, service-level changes, and unresolved cases."""
    out = frame.copy()
    text = frame[["servNm", "summary_text_raw", "benefit_text_raw", "target_text_raw", "criteria_text_raw"]].fillna("")
    name = text.servNm
    benefit = text.benefit_text_raw.str.replace(r"\s+", " ", regex=True)
    summary = text.summary_text_raw.str.replace(r"\s+", " ", regex=True)
    target = text.target_text_raw + " " + text.criteria_text_raw
    offered = name + " " + benefit
    described = offered + " " + summary
    selected_rules: dict[str, pd.Series] = {}
    unresolved: dict[str, pd.Series] = {}

    def tier(mask, value, rule):
        selected_rules[rule] = mask
        out.loc[mask, "q21_10_tier"] = value
        put(out, mask, "q21_10", value in {"main", "candidate"})

    def area(mask, code, value, rule):
        selected_rules[rule] = mask
        put(out, mask, code, value)

    child = offered.str.contains(CHILD_ALLOWANCE)
    recurring = child & described.str.contains(MONTHLY)
    unresolved["양육보조금의 지급 주기 미확인"] = child & ~recurring & ~out.q21_10_tier.eq("main")
    tier(recurring, "main", "정기 양육보조금")
    allowance = offered.str.contains(r"자립\s*지원\s*수당")
    tier(allowance, "candidate", "자립지원수당")

    insurance = offered.str.contains(WORK_INSURANCE) & (
        name.str.contains(r"보험|두루누리") | benefit.str.contains(r"보험료.{0,35}(?:지원|환급)|보험료\s*[:：].{0,20}본인\s*부담")
    )
    essential = offered.str.contains(HOUSEHOLD_INSURANCE) & ~name.str.contains(r"태아|신생아|출생아|출산\s*장려|출생\s*지원")
    premium = essential & described.str.contains(r"보험료.{0,32}(?:감면|경감|전액|일부|지원|지급)|공단.{0,30}(?:지급|납부)")
    tier(premium, "main", "가계 필수보험료")
    tier(insurance & ~premium & ~recurring, "candidate", "직업·사업 정책보험료")

    disaster = described.str.contains(DISASTER)
    relief = disaster & benefit.str.contains(RELIEF_ITEMS)
    goods = disaster & offered.str.contains(RELIEF_GOODS)
    unresolved["재해구호물자의 품목 미확인"] = goods & ~relief
    tier(relief & ~recurring & ~premium & ~benefit.str.contains(LIVING), "candidate", "재해구호 식료품·생필품")

    voucher = offered.str.contains(r"바우처|지원\s*카드|지역\s*(?:상품권|화폐)")
    wide = voucher & benefit.str.contains(BROAD_VOUCHER) & ~described.str.contains(BUSINESS_VOUCHER)
    tier(wide & ~recurring & ~premium & ~benefit.str.contains(LIVING), "candidate",
         "생활필수품 사용이 가능한 광범위 바우처")
    business_only = voucher & described.str.contains(BUSINESS_VOUCHER) & ~recurring & ~premium
    tier(business_only & out.q21_10_tier.isin(["main", "candidate"]) & ~benefit.str.contains(LIVING),
         "excluded", "영농·사업 용도 전용 바우처")
    earmarked = offered.str.contains(RETURN_FUNERAL)
    # Mixed services retain independently supported economic benefits.
    other_cash = benefit.str.contains(r"명예\s*수당|보훈\s*수당|참전\s*수당|긴급한\s*생활안정자금")
    separate_econ = benefit.str.contains(LIVING) | other_cash | recurring | premium | insurance | relief | allowance | wide
    tier(earmarked & ~separate_econ & out.q21_10_tier.isin(["main", "candidate"]),
         "excluded", "귀향여비·장제비 목적 비용")

    edu_before = out.need_area_q21.str.split("|").map(lambda labels: "q21_07" in labels)
    admission = offered.str.contains(ADMISSION) | (
        name.str.contains(r"입학") & summary.str.contains(ADMISSION)
    )
    purpose = described.str.contains(EDU_USE)
    unclear_preparation = summary.str.contains(r"입학\s*준비.{0,15}부담")
    unresolved["입학 준비 비용과 용도 미상 축하금의 구분"] = edu_before & admission & ~purpose & unclear_preparation
    area(edu_before & admission & ~purpose & ~unclear_preparation, "q21_07", False, "용도 미상 입학축하금")

    school_exam = offered.str.contains(SCHOOL_EQUIVALENCY)
    area(school_exam, "q21_07", True, "검정고시 응시료")
    learning = benefit.str.contains(EDU_LEARNING) | name.str.contains(r"수강|학습비|학원비")
    exam = offered.str.contains(EXAM)
    area(edu_before & exam & ~school_exam & ~learning, "q21_07", False, "일반 시험 응시료 단독")
    area(exam & learning, "q21_07", True, "응시료와 수강·학습 지원 병행")
    debt = described.str.contains(r"학자금") & (name.str.contains(DEBT) | benefit.str.contains(DEBT))
    mixed_loan = debt & (learning | name.str.contains(r"부담\s*경감") & benefit.str.contains(r"이자\s*지원"))
    area(edu_before & debt & ~mixed_loan, "q21_07", False, "학자금 채무조정")
    unresolved["학자금 채무조정과 별도 학업 지원의 범위"] = edu_before & mixed_loan

    education = out.need_area_q21.str.split("|").map(lambda labels: "q21_07" in labels)
    out.loc[~education, "q21_07_beneficiary"] = None
    edu_missing = education & out.q21_07_beneficiary.isna()
    # The benefit explicitly identifies the parent as the learner, or the youth as the exam taker.
    parent_learns = benefit.str.contains(r"청소년\s*한부모가\s*검정고시를\s*준비")
    out.loc[education & parent_learns, "q21_07_beneficiary"] = "self"
    selected_rules["청소년 한부모 본인 학습 지원"] = education & parent_learns
    out.loc[edu_missing & school_exam & (name + " " + target).str.contains(r"청년|학교\s*밖\s*청소년|검정고시\s*응시"), "q21_07_beneficiary"] = "self"
    out.loc[education & out.q21_07_beneficiary.isna(), "q21_07_beneficiary"] = "N"
    unresolved["교육 지원 실제 수혜자 미확인"] = education & out.q21_07_beneficiary.eq("N")

    training_cost = offered.str.contains(TRAINING_COST)
    recovery = benefit.str.replace(TRAINING_COST, "", regex=True).str.contains(RECOVERY)
    area(recovery, "q21_02", True, "일상생활·자기관리·독립생활 훈련")
    labeled_recovery = frame.need_area_q21.str.split("|").map(lambda labels: "q21_02" in labels)
    cost_only = training_cost & ~recovery
    out["q1_recovery_training_cost"] = cost_only
    selected_rules["일상 기능·사회적응 훈련 전용 비용 보조 태그"] = cost_only
    area(labeled_recovery & cost_only, "q21_02", False, "전용 비용과 실제 훈련 제공 구분")
    medical_only = labeled_recovery & benefit.str.contains(MEDICAL_REHAB) & ~recovery
    area(medical_only, "q21_02", False, "일상 기능 훈련이 확인되지 않은 의료적 재활")
    unresolved["사회복귀 의료재활의 일상 기능 훈련 세부 미확인"] = medical_only & benefit.str.contains(r"사회\s*복귀")

    care = benefit.str.replace(r"방문\s*요양에\s*상당한", "", regex=True).str.contains(CARE)
    # Generic funding, costs, eligibility mentions and care training for providers do not establish delivery.
    care |= name.str.contains(r"가사.{0,4}간병.{0,8}방문|병원\s*동행|밑반찬\s*배달") & benefit.str.contains(r"서비스|이용권|바우처|배달|제공")
    care |= benefit.str.contains(r"식사\s*지원.{0,35}서비스.{0,5}제공")
    paid_care = offered.str.contains(r"간병\s*비|돌봄\s*비(?:용)?|돌봄\s*(?:서비스\s*)?본인\s*부담")
    reimbursed_care = paid_care & (benefit + " " + target).str.contains(CARE_USE_PROOF)
    care |= reimbursed_care
    out["q1_direct_care"] = care
    selected_rules["직접 돌봄·전용 이용권 보조 태그"] = care
    selected_rules["실제 돌봄 이용 확인 후 비용 환급"] = reimbursed_care
    area(labeled_recovery & care & ~recovery & ~benefit.str.contains(r"자기\s*돌봄비|사회\s*(?:복귀|적응)\s*훈련"),
         "q21_02", False, "직접 돌봄은 ② 핵심에서 제외")
    unresolved["직접 돌봄 제공·전용 이용권과 비용 보전의 구분"] = ~care & (
        offered.str.contains(CARE_PAYMENT_ONLY) | name.str.contains(r"간병비\s*지원|노인맞춤돌봄")
    )

    open_evidence = (summary + " " + benefit + " " + target).str.contains(OPEN_PUBLIC)
    # Scope evidence is only for the open-space component, never for the whole service.
    opened = described.str.contains(OPEN_SPACE) & open_evidence
    opened &= ~target.str.contains(r"(?:만\s*)?\d+\s*세\s*(?:이상|미만).{0,15}(?:시민|주민|국민|지역민)\s*누구나")
    area(opened, "q21_01", True, "시민 누구나 이용하는 개방 공간")
    out["q1_open_space_universal"] = opened

    eligibility = target.str.replace(r"\s+", " ", regex=True)
    age_exception = eligibility.str.contains(AGE_EXCEPTIONS) | frame.servId.isin({"WLF00003191", "WLF00005663"})
    out["q1_age_exception_only"] = age_exception
    selected_rules["청년 연령 예외·개별 심사 보조 집계"] = age_exception
    unresolved["전체 대상과 실제 지원 구성요소의 대상이 상충"] = frame.servId.eq("WLF00004543")
    unresolved["사망자 연령과 실제 수혜자인 유족의 연령 구분"] = frame.servId.eq("WLF00000971")

    housing = frame.need_area_q21.str.split("|").map(lambda labels: "q21_11" in labels)
    referral_only = housing & benefit.str.contains(HOUSING_REFERRAL) & ~benefit.str.contains(HOUSING_OFFER)
    area(referral_only, "q21_11", False, "이송·입소 의뢰에 숙소 제공 근거 없음")
    out["q1_education_direct_only"] = True

    fields = ["need_area_q21", "q21_10_tier", "q21_07_beneficiary"]
    tags = ["q1_direct_care", "q1_open_space_universal", "q1_age_exception_only", "q1_recovery_training_cost"]
    changed = out[fields].fillna("").ne(frame[fields].fillna("")).any(axis=1) | out[tags].any(axis=1)
    records = out.loc[changed, ["servId", "servNm", *fields, *tags]].copy()
    for column in fields:
        records[f"{column}_before"] = frame.loc[changed, column]
    records["q1_human_reviewed"] = frame.loc[changed, "q1_human_reviewed"]
    records["policy_rules"] = pd.DataFrame(selected_rules).loc[changed].apply(
        lambda row: " | ".join(row.index[row]), axis=1)
    records["source_excerpt"] = benefit.loc[changed].str.slice(0, 420)
    pending = pd.DataFrame(unresolved)
    pending = pending.loc[pending.any(axis=1)].apply(lambda row: " | ".join(row.index[row]), axis=1)
    unresolved_rows = frame.loc[pending.index, ["servId", "servNm", "need_area_q21", "q21_10_tier", "q21_07_beneficiary"]].copy()
    unresolved_rows["reason"] = pending
    unresolved_rows["source_excerpt"] = benefit.loc[pending.index].str.slice(0, 420)
    return out, records, unresolved_rows
