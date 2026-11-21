"""1단계 추출 후처리: LLM 결과(normalize 이후)를 원문 표현 규칙으로 보정한다.

kanana_v1 개발 세트 오류 분석(2026-11-12) 결과를 반영한다.
- 제약: LLM이 부담·불가를 자주 혼동함(정확도 0.771). 원문 표현 규칙으로 다시 판정한다.
- 활동 범위: 원문에 평소 외출 패턴 표현이 있으면 규칙 값을 쓰고, 없으면 LLM 값을 둔다.
- 영역: 상태 서술만으로 붙기 쉬운 영역(REQUEST_GATE_DOMAINS)은 문장에 요청·바람 표현이 없으면 '확인 필요'로 낮춘다(삭제하지 않음).
- 원문에서 검진, 수면 상담, 생활습관 개선처럼 직접 요청한 내용이 누락되면 영역을 복원한다.

규칙은 개발 세트(말뭉치 300건)를 보고 만들었으므로 최종 성능은 평가 카드로 확인한다.
"""
from __future__ import annotations

import re

# ── 영역: 요청·바람 표현 ─────────────────────────────────────────────
REQUEST = re.compile(r"(싶|필요|있을까|있나|있는지|알려|도와|받고|원해|원합|바라|좋겠|부탁|해주|해 주|주세요|"
                     r"줄 수|줄수|가능한|방법|찾고|알아보|궁금|어떻게|뭘 해야|뭐 해야|해볼까|해 볼까|할까|"
                     # 오타·구어 변형: 잇나여/있나여, 시퍼, 알려주세여, 도움받, 어케/어떡, 없을까, 좋을 것 같
                     r"[있잇](?:나|을까|을지|는지|냐|으려나|을려나)|없(?:을까|나|는지)|수\s*[있잇]|"
                     r"시퍼|십어|알려\s*(?:줘|주세|줄|달)|도움\s*(?:받|을|이 되|되)|받을\s*수|"
                     r"어케|어떡|어쩌지|어찌|좋을\s*(?:것|거|듯|텐데)|"
                     r"(?:나여|나요|까여|까요|을까|는지|ㄴ지)\s*[?？ㅠㅜ])")
# 취업 실패·고립·외출 범위·연락 방식 서술만으로 잘못 붙던 영역 (v1 오답: 미취업 17, 관계참여 26, 일상·여가 13)
REQUEST_GATE_DOMAINS = {"미취업(구직)", "관계참여", "일상·여가", "노동(일하는 중)"}
JOB_HELP = re.compile(r"(?:일자리(?:를)?\s*(?:찾는|구하는)\s*도움|(?:취업|구직)\s*(?:상담|도움|지원))"
                      r".{0,24}(?:알려\s*(?:줘|주)|받(?:고\s*싶|을\s*(?:수|곳))|필요(?:해|합))")
JOB_HELP_DECLINED = re.compile(r"필요\s*없|원하지\s*않|받(?:고\s*싶지\s*않|기\s*싫)|말고|안\s*받")

_OBJECT = r"(?:을|를|도)?\s*"
_RECEIVE = r"받(?:아\s*보)?"
_RECOVERY_ACTIONS = {
    "건강": re.compile(
        rf"(?<![가-힣])(?:(?:건강\s*검진|검진|진료|치료|심리\s*상담|정신\s*건강\s*상담){_OBJECT}"
        rf"(?:{_RECEIVE}|해\s*보|하)|"
        rf"(?:수면(?:\s*문제)?|불면(?:증)?|낮에\s*자는)\s*(?:문제로|때문에|관련해|걸|것을)?\s*"
        rf"상담{_OBJECT}{_RECEIVE})"
    ),
    "일상·여가": re.compile(
        rf"(?<![가-힣])(?:일상(?:생활)?{_OBJECT}(?:다시\s*)?(?:챙기|되찾|회복하)|"
        rf"생활\s*(?:습관|리듬|패턴){_OBJECT}(?:다시\s*)?(?:잡|되찾|고치|바꾸|개선하)|"
        rf"하루(?:에)?\s*(?:할\s*일|일과|루틴|계획){_OBJECT}(?:만들(?:어\s*보)?|세우|정하)|"
        rf"(?:자기\s*관리|(?:독립|일상)\s*생활\s*훈련){_OBJECT}(?:{_RECEIVE}|배우|하))"
    ),
}
_RECOVERY_ACTION = "(?:" + "|".join(pattern.pattern for pattern in _RECOVERY_ACTIONS.values()) + ")"
# A shared desire also applies to coordinated actions such as "일상을 챙기고 검진도 받고 싶다".
_RECOVERY_REQUEST = re.compile(
    rf"(?P<actions>{_RECOVERY_ACTION}(?:고\s+{_RECOVERY_ACTION})*)"
    r"고\s*싶(?!다고|대|더|어\s*(?:하|해))(?:어요|어여|어서|어|습니다|다|고|은데|은|으면)?"
)

# ── 제약: 전화·방문 × 부담·불가 ────────────────────────────────────────
OUTING_CHANNEL = r"(?:(?:집\s*)?밖(?:에|으로|엔)\s*(?:못\s*|안\s*)?나가|외출)"
ACCESS_CHANNEL_WORDS = {
    "전화": r"(?:전화|통화|벨\s*소리|모르는\s*번호)",
    "방문": (r"(?:(?<!비)대면|방문|주민\s?센터|행정복지센터|동사무소|구청|센터\s?(?:가|방문)|"
             r"직접\s?(?:가|방문|찾아)|찾아가|" + OUTING_CHANNEL + r")"),
}
_VISIT = (r"(찾아오|가서|가는 ?건|가는 ?것|가는 ?거|가야|가 볼|가볼|갈 수|갈수|갈 엄두|"
          r"기관|센터|주민센터|동사무소|모임|밖으로 나가|밖에 나가는|나가는 ?건|외출|이동|어디 가|어디가|거기까지|"
          r"현관 밖|밖에서 진행|어디로|오라고|오라는|가려니|가는게|가는 게|자리에|가는 데|나가지 못|못 ?나가|멀리)")
_NO = re.compile(r"불가(?:능)?|못\s*(?:해|하|가|받|나가|걸|갈)|나가지 못|수(?:가)?\s*없|"
                 r"안\s*(?:돼|되)|끊어버릴|무리")
ACCESS_RELUCTANCE = r"(?:피하|피할|꺼려|꺼리|내키지\s*않)"
EXTRA_ACCESS_BURDEN = (ACCESS_RELUCTANCE + r"|떨[려리]|숨(?:이)?\s*막|엄두(?:가|도)?\s*(?:안\s*나|나지\s*않)|"
                       r"버거|거북|두렵|두려|불편|곤란")
ELIDED_ACCESS_BURDEN = re.compile(
    r"^(?:(?:되도록(?:이면)?|가능하면|가급적|웬만하면|그래도|좀|조금|아직|너무)\s+)*"
    r"(?:무섭|무서|" + ACCESS_RELUCTANCE + r")"
)
_SOFT = re.compile(r"부담|무섭|무서|긴장|망설|겁|쫄|힘들|힘드|어렵|어려|싫|편해|편합|별로|좀 그래|쑥스|막막|"
                   r"마음이 무겁|용기가 필요|어색|심장|말이 안 ?나|대답 못|" + EXTRA_ACCESS_BURDEN)
_CLAUSE_BREAK_BASE = (r"(?<=[,;])\s*|(?<=데)\s+|(?<=지만)\s+|(?<=니까)\s+|"
                      r"(?<=[고며])\s+(?!(?:싶|있|나서))")
# 검색 쿼리(query/build.py)는 '~서' 뒤에서도 끊는다 (#75)
_ACCESS_CLAUSE_BREAK = re.compile(_CLAUSE_BREAK_BASE + r"|(?<=서)(?<!에서)\s+")
# 제약 판정은 '~서'에서 끊지 않는다. "직접 찾아가서 상담받는 건 부담"처럼 채널과 정도가 갈라지기 때문 (kanana_v3 ⑱)
_CONSTRAINT_CLAUSE_BREAK = re.compile(_CLAUSE_BREAK_BASE)
_NEGATED_CUE_END = re.compile(r"\s*[가-힣]{0,5}(?:지(?:는|도)?|진)\s*않|"
                              r"\s*[가-힣]{0,5}\s*(?:건|게|것(?:은|이|도)?)\s*아(?:니|닌|닙)")
# "가야 되면 가볼게"처럼 할 수는 있다는 말이 있으면 불가가 아니라 부담
_CAN = (r"(갈 수는|갈수는|가 볼 수는|가볼 수는|가볼수는|가볼게|가 볼게|가 보려|가볼 의향|가 볼 생각|꼭 가야|"
        r"필요하면 가|필요하다면|가야 ?되면|가면 갈 수)")
# "전화 말고 문자로"처럼 대체 수단을 원하면 전화 부담
_PREF = (r"(말고|말구|대신|보다|아닌|만 아니면).{0,12}(문자|톡|메시지|메세지|채팅|글|앱)?|"
         r"(문자|톡|메시지|메세지|채팅|글)(로|으로)만|(문자|톡|메시지|메세지|채팅)(로|으로)? ?(해|받|알려|부탁|답|주|면 좋)")

# ── 활동 범위: 평소 외출 패턴 ─────────────────────────────────────────
_HYPO = (r"(싶|면 좋|가야 ?되면|가야 ?하면|가야 한다면|필요하면|필요하다면|해볼|해 볼|가볼|가 볼|할까|생각은|알려|"
         r"주세요|있을까|있나요|있는지|라던데|오라는|오라고|얘기 들었|보라는)")
_ROOM = (r"(방에서 안 ?나|방에서 (게임|컴퓨터|핸드폰|유튜브|밥|혼자)|방, 화장실|방에서|방에만|방 ?밖|방밖|방문 ?닫|"
         r"방에 (있|틀어|처박|머물)|방 안|방안|방에서만)")
_ROOM_OUT = r"(방에서?는 ?나와|방에선 ?나와|방에서는 나오|방 ?밖으로?는 ?나)"   # 방에서는 나온다 = 집 안
_HOME = (r"(방에서?는 ?나와|방에선 ?나와|방에서는 나오|방 ?밖으로?는 ?나|집에만|집에서만|거의 안 ?나가|밖엔 거의 안|밖에 거의 안|집에서 거의 안|집 ?밖|현관 ?밖|집 안에서만|"
         r"집안에서만|집에서만 지|밖에 (안|못) ?나간 ?지|밖에 안나간|밖에 나간 게 언제|외출을? (거의 )?(안|못|하지 않))")
_NEAR = r"(편의점|마트|집 ?앞|동네|근처|산책|담배 ?사러|장 ?보러|컵라면 ?사러|가게 정도|슈퍼)"
_NEAR_ONLY = r"(정도|만|외에는|말고는|빼고|말곤|까지만|이 다|게 다|전부)"
_OUT = (r"((다니|다녀요|다녀$|다닙|나가요|나가고 있|나가$|나가는데|나가기 시작|등록).{0,3}$|"
        r"(학원|수업|화실|클래스|크루|모임|동호회|합창단|체육관|헬스장?|탁구장|수영|요가|배우러|그림|운동|등산|"
        r"클라이밍|전시|공원|재활|치료|진료|검진|병원)[^.]{0,15}(다니|다녀|다닙|나가|가요|가$|가고|갈 때|갈때|만 ?외출|때만)|"
        r"출근|출퇴근|퇴근|직장[^.]{0,8}다|회사[^.]{0,6}(다니|가|나가)|외출이 늘|주말마다|일주일에|가끔 나가|자주 나가)")

GOLD_ORDER = [("방문", "불가"), ("전화", "불가"), ("방문", "부담"), ("전화", "부담")]


def sentences(text: str, *, split_commas: bool = True) -> list[str]:
    sep = r"(?<=[.?!~;])\s*|(?<=[ㅠㅜ])\s+"
    parts = re.split(sep + (r"|,\s*" if split_commas else ""), text or "")
    return [p.strip() for p in parts if p and p.strip()]


def has_request(text: str) -> bool:
    return bool(REQUEST.search(text or ""))


def access_clauses(text: str) -> list[str]:
    return [c.strip() for c in _ACCESS_CLAUSE_BREAK.split(text) if c.strip()]


def constraint_clauses(text: str) -> list[str]:
    """제약 판정용 절 나누기. access_clauses와 같지만 '~서' 뒤에서는 끊지 않는다."""
    return [c.strip() for c in _CONSTRAINT_CLAUSE_BREAK.split(text) if c.strip()]


def negates_cue(tail: str) -> bool:
    """제약 표현 바로 뒤의 '싫지는 않다', '불가능한 건 아니다' 같은 부정."""
    return bool(_NEGATED_CUE_END.match(tail))


def mask_negated_cues(text: str, cues: re.Pattern) -> str:
    """이중 부정의 안쪽 표현과 바깥 부정어를 함께 가려 다시 세지 않는다."""
    end = 0
    parts = []
    for m in cues.finditer(text):
        negation = _NEGATED_CUE_END.match(text, m.end())
        if negation and m.start() >= end:
            parts.append(text[end:m.start()])
            end = negation.end()
    return " ".join(parts + [text[end:]])


def add_recovery_needs(needs: list[dict], text: str) -> list[dict]:
    """Recover omitted health and daily needs only from directly requested actions."""
    found = {need["domain"] for need in needs}
    added = []
    for sentence in sentences(text):
        for request in _RECOVERY_REQUEST.finditer(sentence):
            if negates_cue(sentence[request.end():]):
                continue
            actions = request["actions"]
            evidence = []
            for domain, pattern in _RECOVERY_ACTIONS.items():
                matches = list(pattern.finditer(actions)) if domain not in found else []
                if matches:
                    start = request.start("actions") + matches[0].start()
                    end = request.start("actions") + matches[-1].end() + 1
                    if matches[-1].end() == len(actions):
                        end = request.end()
                    evidence.append((start, end, domain))
            for start, end, domain in sorted(evidence):
                added.append({"domain": domain, "level": "sub" if needs or added else "main",
                              "evidence": sentence[start:end], "status": "명시"})
                found.add(domain)
    return [*needs, *added] if added else needs


def rule_constraints(text: str) -> list[dict]:
    """절 안에서 제약을 판정하고, 명시적인 회피는 바로 앞 절의 채널을 이어받는다."""
    found: dict[str, dict] = {}
    whole_can = bool(re.search(_CAN, text or ""))
    for s in sentences(text, split_commas=False):
        previous = set()
        for clause in constraint_clauses(s):
            channels = {ch for ch, pat in ACCESS_CHANNEL_WORDS.items() if re.search(pat, clause)}
            inherited = not channels and bool(previous and ELIDED_ACCESS_BURDEN.match(clause))
            if inherited:
                channels = previous
            elif not channels and re.search(_VISIT, clause):
                channels = {"방문"}
            previous = channels
            if not channels:
                continue
            hard = list(_NO.finditer(clause))
            soft = list(_SOFT.finditer(clause))
            no = any(not negates_cue(clause[m.end():]) for m in hard)
            burden = any(not negates_cue(clause[m.end():]) for m in soft)
            negated = any(negates_cue(clause[m.end():]) for m in hard + soft)
            can = not negated and bool(re.search(_CAN, clause))
            for ch in channels:
                pref = ch == "전화" and not negated and re.search(_PREF, clause)
                if no:
                    lv = "부담" if can or (ch == "방문" and whole_can) else "불가"
                elif burden or pref or (ch == "방문" and can):
                    lv = "부담"
                else:
                    continue
                if ch not in found or (found[ch]["level"] == "부담" and lv == "불가"):
                    found[ch] = {"channel": ch, "level": lv, "evidence": s if inherited else clause}
    return [found[ch] for ch in ("전화", "방문") if ch in found]


def clauses(text: str) -> list[str]:
    """문장을 '~는데', '~지만' 뒤에서 한 번 더 나눈다 (평소 패턴과 바람이 한 문장에 섞인 경우)."""
    out = []
    for s in sentences(text):
        out += [c.strip() for c in re.split(r"(?<=는데)\s+|(?<=지만)\s+", s) if c.strip()]
    return out


_STRONG_BREAK = re.compile(r"(?<=[,;])\s*|(?<=는데)\s+|(?<=인데)\s+|(?<=지만)\s+|(?<=니까)\s+|(?<=면서)\s+")
_ANY_ACTIVITY = re.compile("|".join((_ROOM, _HOME, _NEAR, _OUT)))


def rule_activity(text: str) -> tuple[str, str | None]:
    """평소 외출 패턴 표현으로 활동 범위를 판정한다. 바람·가정 구절은 보지 않는다."""
    cues: dict[str, str | None] = {"방 안": None, "집 안": None, "집 근처": None, "필요하면 외출": None}
    for s in clauses(text):
        if re.search(_HYPO, s):
            continue
        for k, pat in (("방 안", _ROOM), ("집 안", _HOME), ("집 근처", _NEAR), ("필요하면 외출", _OUT)):
            if k == "방 안" and re.search(_ROOM_OUT, s):
                continue
            if cues[k] is None and re.search(pat, s):
                cues[k] = s
    pats = {"방 안": _ROOM, "집 안": _HOME, "집 근처": _NEAR, "필요하면 외출": _OUT}

    def narrow(k, ev):
        """근거는 패턴이 든 절로 좁힌다. 같은 문장의 욕구·상황 서술(월세 밀림 등)이 활동 근거에 섞이지 않게.
        '~고/~며'처럼 약한 경계와 '독서'의 '서'에서는 끊지 않고, 이어진 절이 함께 활동을 서술하면 합친다."""
        parts = [c.strip() for c in _STRONG_BREAK.split(ev) if c.strip()]
        if k == "집 안":
            hit = next((c for c in parts if re.search(OUTING_CHANNEL, c) and re.search(_HOME, c)), None)
            if hit:
                return hit
        i = next((j for j, c in enumerate(parts) if re.search(pats[k], c)), None)
        if i is None:
            return ev
        lo = hi = i
        while lo > 0 and _ANY_ACTIVITY.search(parts[lo - 1]):
            lo -= 1
        while hi + 1 < len(parts) and _ANY_ACTIVITY.search(parts[hi + 1]):
            hi += 1
        return " ".join(parts[lo:hi + 1])

    if cues["집 근처"] and re.search(_NEAR_ONLY, cues["집 근처"]):
        return "집 근처", narrow("집 근처", cues["집 근처"])
    for k in ("필요하면 외출", "집 근처", "방 안", "집 안"):
        if cues[k]:
            return k, narrow(k, cues[k])
    return "판단 불가", None


# ── 나이: LLM이 놓친 "22살", "스물일곱인데"를 원문에서 보정 (kanana_v3 개발 카드 9장 누락) ─────────
_KO_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40}
_KO_ONES = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4, "네": 4, "다섯": 5,
            "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_AGE_NUM = re.compile(r"(?<![\d.])(\d{2})\s*(?:살|세)(?!\s*(?:때|부터|까지|쯤|전|이후|무렵))")
_AGE_KO = re.compile(r"(?<![가-힣])(열|스물|스무|서른|마흔)\s?(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉)?"
                     r"(?=\s*살|인데|이고|이구|이에요|이예요|입니다|이야|임|이라|[,.\s]|$)"
                     r"(?!\s*살?\s*(?:때|부터|까지|쯤|전|이후|무렵))")
AGE_RULE_RANGE = (15, 49)


def rule_age(text: str) -> int | None:
    """원문에 나온 본인 나이. 숫자(22살·22세)와 한글(스물일곱, 서른 살)만, '~살 때' 같은 과거 시점은 제외."""
    for m in _AGE_NUM.finditer(text or ""):
        v = int(m.group(1))
        if AGE_RULE_RANGE[0] <= v <= AGE_RULE_RANGE[1]:
            return v
    for m in _AGE_KO.finditer(text or ""):
        tens, ones = m.group(1), m.group(2)
        if tens == "스무" and ones:
            continue
        rest = text[m.end():]
        if not ones and not re.match(r"\s*살|인데|이고|이구|이에요|이예요|입니다|이야|임|이라", rest):
            continue   # "서른 넘어서"처럼 나이 단정이 아닌 말
        v = _KO_TENS[tens] + (_KO_ONES[ones] if ones else 0)
        if AGE_RULE_RANGE[0] <= v <= AGE_RULE_RANGE[1]:
            return v
    return None


# ── 힌트: 상태 서술 단서 → 영역 (확인 필요 후보) ───────────────────────
# 추출된 영역이 하나도 없을 때만 쓴다. 상태 서술은 '명시' 요구가 아니므로 status는 항상 '확인 필요'이고,
# 검색에 쓰이지 않으며(confirm usable=명시만) 되묻기 단계에서 후보로만 보인다. extract_rules.md '영역이 아닌 것'의 상태 표현을 단서로 삼음.
HINT_CUES = {
    "관계참여": r"사람\s*(?:만나|대하|마주|사귀)\S*|대인\s*관계|외로[웠워운]\S*|외롭|쓸쓸\S*|고립|"
              r"(?:사람|친구|동료)\S*\s*(?:이|도|가)?\s*(?:아무도\s*)?없\S*|혼자\s*(?:지내|버티|살)\S*|"
              r"주변\s*사람\S*\s*(?:다\s*)?끊\S*|사람(?:이|들이)\s*무서\S*",
    "일상·여가": r"낮밤\S*|밤낮\S*|생활\s*(?:리듬|패턴)\S*|하루(?:가|를)?\s*(?:어떻게|그냥)\S*|무기력\S*|"
              r"의욕(?:이|도)?\s*(?:없|안)\S*|아무것도\s*(?:하기\s*싫|안\s*해)\S*|할\s*(?:게|일이|거|것)\s*(?:없|딱히)\S*|"
              r"심심\S*|재미(?:가|있는\s*게)?\s*없\S*|(?:낮엔|새벽에)\s*자\S*",
    "건강": r"잠(?:을|이)?\s*(?:못|안)\s*\S*|불면\S*|우울\S*|불안\S*|공황\S*|통증\S*|"
          r"몸(?:이|도)\s*(?:자꾸\s*|많이\s*)?(?:안\s*좋|아프|아파|무거|안\s*따라)\S*|"
          r"마음(?:이|도)\s*(?:안\s*좋|힘들|지쳐)\S*|번아웃\S*|스트레스\S*|나쁜\s*생각\S*",
    "미취업(구직)": r"취업(?:도|이|을)?\s*(?:안|못)\s*\S*|일(?:을|자리를)?\s*(?:못|안)\s*(?:구|찾)\S*|구직\s*(?:포기|중단)\S*|"
                  r"(?:공채|면접|서류|시험)\S*\s*(?:계속\s*|모두\s*)?(?:떨어|탈락)\S*|탈락\S*|(?:백수|무직|취준)\S*|"
                  r"지원\s*(?:자체를\s*)?멈\S*",
    "교육훈련": r"공부(?:를|도)?\s*(?:못|안|손에)\S*",
    "미래설계": r"앞(?:날|으로)\S*\s*(?:막막|모르|걱정)\S*|(?:뭘|뭐)\s*해야\s*(?:할|될|되는|하는)지\S*|"
              r"진로\S*\s*(?:모르|고민|막막)\S*|미래\S*\s*(?:막막|걱정|불안)\S*|하고\s*싶은\s*(?:게|일이)\s*없\S*",
    "경제": r"돈(?:이|도)?\s*(?:계속\s*|다\s*)?(?:없|부족|떨어)\S*|생활비\S*|공과금\S*|빚\S*|대출\S*",
    "주거": r"월세\S*|집세\S*|관리비\S*|쫓겨\S*|보증금\S*|(?:살|지낼)\s*(?:곳|집)(?:이|도)?\s*없\S*",
}
HINT_RE = {d: re.compile(p) for d, p in HINT_CUES.items()}
HINT_MAX = 2
# 원하지 않는다고 한 문장, 다른 사람의 바람('~고 싶어해요')은 힌트로 쓰지 않는다
_HINT_SKIP = re.compile(r"싶지\s*않|원하지\s*않|필요\s*없|받기\s*싫|싶어\s*하|싶어해")


def state_hints(text: str) -> list[dict]:
    """상태 서술 단서로 만든 영역 힌트(확인 필요, 최대 2개). 근거는 원문 구절 그대로."""
    found, seen = [], set()
    for i, sent in enumerate(sentences(text or "")):
        if _HINT_SKIP.search(sent):
            continue
        for d, pat in HINT_RE.items():
            m = pat.search(sent)
            if m and d not in seen:
                seen.add(d)
                found.append((i, m.start(), d, m.group(0).strip().rstrip(".,!?~;ㅠㅜㅎ…")))
    found.sort()
    out = [{"domain": d, "level": "sub", "evidence": ev, "status": "확인 필요"}
           for *_, d, ev in found[:HINT_MAX] if ev]
    if out:
        out[0]["level"] = "main"   # 상태 서술 전용 낮춤(위)과 같게 첫 항목을 main
    return out


def postprocess(rec: dict, text: str) -> dict:
    """normalize()를 거친 extract-v1 기록을 규칙으로 보정한다. 원본은 바꾸지 않는다."""
    rec = {**rec}
    if not has_request(text):   # 삭제하지 않고 '확인 필요'로 낮춰 사용자 확인 단계에서 고르게 한다
        rec["needs"] = [{**n, "status": "확인 필요"} if n["domain"] in REQUEST_GATE_DOMAINS else n
                        for n in rec.get("needs", [])]
        if rec["needs"] and all(n["level"] == "sub" for n in rec["needs"]):
            rec["needs"][0] = {**rec["needs"][0], "level": "main"}

    needs = rec.get("needs", [])
    if any(n["status"] == "명시" for n in needs) and not any(n["domain"] == "미취업(구직)" for n in needs):
        for sentence in sentences(text):
            if JOB_HELP.search(sentence) and not JOB_HELP_DECLINED.search(sentence):
                rec["needs"] = [*needs, {"domain": "미취업(구직)", "level": "sub",
                                         "evidence": sentence, "status": "확인 필요"}]
                break

    if rec.get("extract_status") != "failed":   # 직접 요청한 검진·수면 상담 등 누락 영역 복원(명시)
        rec["needs"] = add_recovery_needs(rec.get("needs", []), text)
    if not rec.get("needs"):   # 그래도 영역이 하나도 없을 때만: 상태 서술 힌트(검색에는 쓰이지 않음)
        rec["needs"] = state_hints(text)

    ar, ev = rule_activity(text)
    if ar != "판단 불가":
        rec["activity_range"], rec["activity_evidence"] = ar, ev

    rec["constraints"] = rule_constraints(text)
    if rec.get("age") is None:
        rec["age"] = rule_age(text)
    rec["next_action"] = "추천" if any(n.get("status") == "명시" for n in rec.get("needs", [])) else "되묻기"
    return rec
