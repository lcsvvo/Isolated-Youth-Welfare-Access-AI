# -*- coding: utf-8 -*-
"""일상 대화(인사·고마움·잡담) 응답. 이 문장만 외부 AI(Google Gemini)로 보낸다.

원칙
- 위기 확인은 항상 먼저다. 이 모듈은 위기를 판정하지 않는다: 호출하는 쪽(app.start)이 pipeline.prepare의
  위기 확인을 통과하고 필요한 도움이 하나도 없을 때만 reply()를 부른다.
- 보내는 것은 지금 한 문장 + 앞선 '일상 대화' 최근 3턴뿐이다. 위기·힘듦·도움 신호가 있는 문장, 정보 수집 중 답, 세션·프로필은
  보내지 않는다(호출하는 쪽이 신호 없는 문장만 넘기고, 기록도 Gemini로 보낸 턴만 쌓는다).
- 활동 범위 단계(0~3, 3이 가장 고립)에 따라 지시문과 고정 문구의 말투가 달라진다. 단계를 모르면 3(가장 조심스럽게).
- 키 없음·패키지 없음·API 오류·시간 초과·빈/차단 응답은 모두 고정 기본 답으로 대체한다(예외를 밖으로 내보내지 않음).
- 키 값은 어디에도 기록·출력하지 않는다. 오류는 종류 이름만 남긴다.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-3.5-flash-lite"
TIMEOUT_S = 10
MAX_INPUT = 200
MAX_REPLY = 300
MAX_HISTORY = 3          # Gemini에 보내는 일상 대화 기록(사용자·답 한 쌍 = 1턴)
OFFER_COOLDOWN = 3        # [그냥 얘기할래요] 뒤 같은 신호로 다시 묻기까지 쉬는 턴 수
FALLBACK_REPLY = "말씀해 주셔서 고마워요. 필요한 도움이 있으면 편하게 이야기해 주세요."

# ── 활동 범위 단계 (0~3, 3이 가장 고립). 판단 불가·미입력은 3(가장 조심스럽게) ──
ACTIVITY_STAGE = {"필요하면 외출": 0, "집 근처": 1, "집 안": 2, "방 안": 3}
UNKNOWN_STAGE = 3


def care_stage(activity_range=None) -> int:
    return ACTIVITY_STAGE.get(activity_range, UNKNOWN_STAGE)


# 고정 문구(Gemini를 부르지 않는 말). 3단계는 짧게, 외출·만남·연락·전화 권유 없이, 답하지 않아도 된다고 매번 말한다.
OFFER_QUESTION = "그 부분, 도움이 될 만한 걸 같이 찾아볼까요?"
OFFER_BUTTONS = ("찾아볼래요", "그냥 얘기할래요")
OFFER_NOTE = {0: "편할 때 골라 주세요.", 1: "천천히 골라도 괜찮아요.", 2: "괜찮아요. 천천히 골라도 돼요.",
              3: "답하지 않아도 괜찮아요."}
# [그냥 얘기할래요] 같은 고정 공감 답: 단계별 3개를 돌아가며 쓴다. 질문이 아니므로 "대답하지 않아도 괜찮아요"는 붙이지 않는다
# (그 안내는 질문하는 말풍선 OFFER_NOTE에만). 3단계는 짧게, 외출·만남·연락·전화 권유 없이.
TALK_REPLIES = {
    0: ["좋아요. 편하게 얘기해요.", "네, 좋아요. 하고 싶은 말이 있으면 해 주세요.", "알겠어요. 가볍게 이야기해요."],
    1: ["네, 그냥 얘기해요. 천천히 편하게 말해 주세요.", "좋아요. 편한 만큼만 이야기해 주세요.", "네, 천천히 얘기해요. 저는 여기 있어요."],
    2: ["네, 괜찮아요. 천천히 얘기해도 돼요.", "괜찮아요. 말하고 싶을 때만 해 주세요.", "네, 괜찮아요. 서두르지 않아도 돼요."],
    3: ["네. 여기 있을게요.", "네. 천천히 해도 괜찮아요.", "네. 편한 말만 해도 돼요."],
}
TALK_REPLY = {stage: texts[0] for stage, texts in TALK_REPLIES.items()}   # 첫 번째 문구(하위 호환)


def talk_reply_text(stage, n=0) -> str:
    """n번째 고정 공감 답(돌아가며). 단계를 모르면 3."""
    texts = TALK_REPLIES[stage if stage in TALK_REPLIES else UNKNOWN_STAGE]
    return texts[n % len(texts)]


FALLBACK_BY_STAGE = {0: FALLBACK_REPLY, 1: FALLBACK_REPLY, 2: FALLBACK_REPLY,
                     3: "말씀해 주셔서 고마워요. 편할 때 또 얘기해 주세요."}
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "chitchat_system.md"
_FALLBACK_PROMPT = ("복지 안내 도구의 대화 도우미예요. 일상 인사나 잡담에 해요체로 1~2문장으로 짧게 답하고, "
                    "의료·법률 조언과 복지 서비스 안내는 하지 말고, 도움이 필요하면 이야기해 달라고 안내해요.")

# ── 일상 대화 판별 (규칙 기반. 애매하면 False = 기존 복지 흐름으로) ──
_STRIP = re.compile(r"[\s.,!?~ㅋㅎㅠㅜ^;:()\-_'\"…]+")
_CHITCHAT = re.compile(
    r"^(안녕|안녕하세요|안녕하세여|하이|헬로|반가워|반갑습니다|반가워요|처음뵙겠습니다|좋은아침|좋은아침이에요|굿모닝|"
    r"고마워|고마워요|고맙습니다|감사해요|감사합니다|감사|땡큐|고마워용|"
    r"잘가|잘가요|안녕히계세요|안녕히가세요|다음에봐|다음에봬요|"
    r"너누구야|넌누구야|누구세요|이름이뭐야|이름이뭐예요|뭐해|뭐하고있어|뭐하세요|"
    r"심심해|심심해요|오늘날씨어때|날씨좋다|날씨좋네요|오늘뭐했어|"
    r"그래|그렇구나|그렇군요|알겠어요|알겠어|응|네|넵|아하|오케이|ok)"
    r"(요|용|여|야|에요|예요|네|네요|죠)?$", re.I)
# 이 말이 들어 있으면 일상 대화가 아니라고 본다 (복지·도움·힘듦·위험 표현은 기존 흐름이 처리)
_NOT_CHITCHAT = re.compile(
    r"돈|월세|생활비|집세|주거|일자리|취업|구직|알바|직장|일하|상담|지원|신청|복지|서비스|혜택|대출|수당|병원|치료|"
    r"학교|공부|교육|훈련|자격증|빚|채무|공과금|도움|필요|"
    r"힘들|우울|외로|죽|무서|불안|아프|아파|괴로|지쳐|지치|무기력|의욕|막막|모르겠|살기|사라|끝내|자해|폭력|학대|위험")


def has_signal(text) -> bool:
    """힘듦·도움·복지 표현이 있으면 True. 규칙만 쓴다(문장이 외부로 나가지 않음). 있으면 Gemini로 보내지 않는다."""
    return bool(_NOT_CHITCHAT.search(str(text or "")))


_CONCRETE = re.compile(r"월세|전세|집세|보증금|생활비|공과금|밥값|통신비|일자리|취업|구직|알바|이직|실직|해고|학비|등록금|병원비|치료비|빚|대출|채무|주거|이사")
_EXPLICIT = re.compile(
    r"서비스.{0,6}(찾|내놔|알려|추천|보여|신청|받)|(찾아|추천해?|보여|신청해?)(줘|줄래|주세요|달라|봐|볼래|보고|드려)"
    r"|찾고싶|신청(하고|방법|할래|할수)|내놔")
_ASK_KNOW = re.compile(r"알려(줘|줄래|주세요|달라|드려)")           # '알려 줘'는 아래 복지 단어와 함께일 때만 분명한 요청
_WELFARE_WORD = re.compile(r"서비스|지원|혜택|복지|제도|신청")
_YES = re.compile(r"(찾아볼래|찾아볼게|찾아봐|찾아봐줘|찾아줘|찾아주세요|찾아보자|네|넵|넹|예|응|어|그래|좋아|좋습니다|부탁해|부탁드려|해줘|해주세요|오케이|ok)(요|용|유)?")
_NO = re.compile(r"(그냥얘기(할래|해|할게)|그냥이야기(할래|해|할게)|얘기할래|아니|아뇨|아니에|싫어|싫습니다|괜찮아|괜찮습니다|나중에|다음에|됐어|됐습니다|패스)(요|용|유)?")


def has_concrete_need(text) -> bool:
    """월세·생활비·일자리처럼 구체적인 도움 주제가 있으면 True (막연한 힘듦과 구분)."""
    return bool(_CONCRETE.search(str(text or "")))


def is_explicit_request(text) -> bool:
    """'서비스 찾아', '추천해 줘', '신청' 같은 분명한 요청이면 True. 제안 없이 바로 기존 흐름으로 간다.
    '알려 줘'는 서비스·지원·혜택·복지·제도·신청 같은 단어와 함께 있을 때만 (예: '오늘 날씨 알려 줘'는 일상 대화)."""
    t = _STRIP.sub("", str(text or ""))
    if not t:
        return False
    return bool(_EXPLICIT.search(t)) or bool(_ASK_KNOW.search(t) and _WELFARE_WORD.search(t))


def offer_answer(text):
    """제안 질문에 글자로 답했을 때: 'yes'([찾아볼래요]와 같음) | 'talk'([그냥 얘기할래요]와 같음) | None."""
    t = _STRIP.sub("", str(text or "")).lower()
    if _YES.fullmatch(t):
        return "yes"
    if _NO.fullmatch(t):
        return "talk"
    return None


_PRIVATE = re.compile(r"(?<!\d)0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)|\d{6}[-\s]?[1-4]\d{6}|\S+@\S+\.\S+|\d{8,}")


def can_send(text) -> bool:
    """Gemini로 보내도 되는 문장인가: 비어 있지 않고 200자 이하이며 전화·주민번호·이메일·긴 숫자 모양이 없음."""
    t = str(text or "").strip()
    return bool(t) and len(t) <= MAX_INPUT and not _PRIVATE.search(t)


def looks_like_chitchat(text) -> bool:
    """짧은 인사·고마움·잡담이면 True. 복지·도움·힘듦 표현이 있거나 길면 False."""
    t = str(text or "").strip()
    if not t or len(t) > 30 or _NOT_CHITCHAT.search(t):
        return False
    return bool(_CHITCHAT.match(_STRIP.sub("", t)))


# ── 키·모델 읽기 ──
def _secret(name, secrets=None):
    """st.secrets(없거나 비어 있어도 안전) → 환경변수 순서. 값은 문자열이거나 None."""
    value = None
    if secrets is not None:
        try:
            value = secrets[name]
        except Exception:      # 파일 없음·키 없음
            value = None
    value = value or os.environ.get(name)
    value = str(value).strip() if value else ""
    return value or None


def resolve_api_key(secrets=None):
    return _secret("GEMINI_API_KEY", secrets)


def resolve_model(secrets=None):
    return _secret("GEMINI_MODEL", secrets) or DEFAULT_MODEL


_STAGE_HEAD = re.compile(r"^## 단계 ([0-3])\s*$", re.M)


def system_prompt(stage=None) -> str:
    """공통 규칙 + 해당 단계 블록. stage가 None·범위 밖이면 3(가장 조심스럽게)."""
    try:
        text = PROMPT_FILE.read_text(encoding="utf-8")
    except OSError:
        text = ""
    parts = _STAGE_HEAD.split(text)               # [공통, "0", 블록0, "1", 블록1, ...]
    base = (parts[0].strip() if parts else "") or _FALLBACK_PROMPT
    blocks = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    st = stage if stage in blocks else UNKNOWN_STAGE
    return base + (f"\n\n[말투 단계 {st}]\n{blocks[st]}" if st in blocks else "")


def make_client(api_key, timeout_s=TIMEOUT_S):
    """google-genai 클라이언트. 패키지가 없으면 ImportError. 시간 제한은 밀리초 단위(HttpOptions)."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


@dataclass
class ChatReply:
    text: str
    source: str            # "gemini" | "fallback"
    error: str = ""        # fallback일 때 이유(종류 이름만. 키·원문 없음)


def _fallback(reason, stage=None):
    return ChatReply(FALLBACK_BY_STAGE[care_stage_value(stage)], "fallback", reason)


def care_stage_value(stage):
    return stage if stage in FALLBACK_BY_STAGE else UNKNOWN_STAGE


def _clean_reply(text):
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s[:MAX_REPLY].rstrip() if s else ""


def _history_contents(history, types):
    out = []
    for pair in list(history or [])[-MAX_HISTORY:]:
        user, bot = (str(pair[0] or "").strip()[:MAX_INPUT], _clean_reply(pair[1]))
        if user and bot:
            out += [types.Content(role="user", parts=[types.Part(text=user)]),
                    types.Content(role="model", parts=[types.Part(text=bot)])]
    return out


def reply(text, secrets=None, stage=None, history=()) -> ChatReply:
    """일상 대화 한 문장에 답한다. history는 앞선 일상 대화 (사용자, 답) 쌍(최근 3턴까지만 사용).
    어떤 실패도 예외로 내보내지 않고 단계별 기본 답으로 바꾼다."""
    msg = str(text or "").strip()
    if not msg or not can_send(msg):        # 200자 초과·개인정보 모양은 잘라 보내지 않고 아예 보내지 않음
        return _fallback("empty_input" if not msg else "not_sendable", stage)
    key = resolve_api_key(secrets)
    if not key:
        return _fallback("no_key", stage)
    try:
        from google.genai import types
        client = make_client(key)
        config = types.GenerateContentConfig(system_instruction=system_prompt(stage), temperature=0.7,
                                             max_output_tokens=400)
        past = _history_contents(history, types)
        contents = past + [types.Content(role="user", parts=[types.Part(text=msg)])] if past else msg
        response = client.models.generate_content(model=resolve_model(secrets), contents=contents, config=config)
        answer = _clean_reply(getattr(response, "text", None))
    except Exception as e:      # 패키지 없음·시간 초과·API 오류·차단 등 전부
        return _fallback(type(e).__name__, stage)
    return ChatReply(answer, "gemini") if answer else _fallback("empty_reply", stage)
