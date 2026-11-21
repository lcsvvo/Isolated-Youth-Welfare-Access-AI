# -*- coding: utf-8 -*-
"""「반 걸음」 Streamlit 챗봇 화면 v5 (실제 Pipeline 연결)

실행:  ai/ 폴더에서   python -m streamlit run app.py

화면 흐름
  첫 화면(home) → 채팅(chat, 한 번에 한 질문) → 정리표(confirm) → 백엔드 추가 질문(chat) → 결과(results) → 상세(detail)
  그 밖에: 서비스 둘러보기(explore) · 정보 부족 탐색(unsure) · 담은 서비스(visit) · 이용 안내(about) · 지금 필요한 도움(safety) · 개발자(dev)
  위기(crisis_stop)일 때는 safety 화면만 열리고, 새 이야기 시작 말고는 막힌다.

백엔드 연결
- 화면은 Pipeline(ai/pipeline.py)의 prepare / resume / search 반환 모양만 읽는다. 백엔드는 고치지 않는다.
- 채팅 질문(필요한 도움·지역·외출·나이·전화)은 확인 단계(awaiting_confirm)의 확인 조작
  (add_need·set_region·set_activity·set_age·set_constraint)으로 보낸다. 자유 입력은 먼저 memo로 보내 위기 확인을 거친다
  (memo는 검색에 쓰이지 않음).
- 정리표의 '이 조건으로 도움 찾기' = 바뀐 값만 확인 조작으로 보낸 뒤 finish_confirm.
  그 뒤 백엔드가 묻는 질문(awaiting_answer)은 채팅에서 받고, ready가 되면 바로 검색한다.
- '조건 바꾸기'는 finish_confirm 직전 세션 사본(snapshot)으로 돌아가 정리표를 다시 연다(ready 세션은 resume 불가).
- MockPipeline도 같은 모양을 돌려주므로 화면 쪽에 가짜/실제 분기가 없다.
- 말투 수위(tone_level 1~3)는 화면 고정 문구 고르는 데만 쓴다. 순위·필터·자격 판단에는 쓰지 않고 화면에도 보이지 않는다.
- 사용자 입력·CSV 원문·백엔드 문구는 unsafe_allow_html에 넣지 않는다(마크다운 이스케이프 후 표시).
  HTML은 고정 문구(스타일·제목·안내 카드)에만 쓴다.
"""
import copy
import html
import json
import logging
import math
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

import chitchat
from runtime_state import pipeline_inputs
from confirm import questions as Q
from confirm.session import confirm_view, new_session
from small_activities import should_offer, select_activities, pair_activities
from ui_text import friendly_summary
from extract.schema import empty_record
from rules.normalize import MERGED_SIDO, NATIONWIDE, norm_sido, parse_bound, use_sgg_map
from safety.guidance import crisis_guidance

st.set_page_config(page_title="반 걸음", page_icon="🌗", layout="wide", initial_sidebar_state="expanded")

AI_DIR = Path(__file__).resolve().parent


def setup_logging():
    """AI 설명이 원문 안내로 바뀐 이유(explain 로거)를 서버 로그(Colab은 streamlit.log)에 남긴다. 스크립트가 다시 돌아도 한 번만 설정."""
    lg = logging.getLogger("explain")
    if not lg.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(handler)
    lg.setLevel(logging.INFO)


setup_logging()

# ─────────────────────────────────────────────
# 1. 화면용 설정값
# ─────────────────────────────────────────────
SIDO = sorted(["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기",
               "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"])
# 버튼은 광주·전남을 따로 둔다. 검색은 둘 다 전남광주통합특별시 서비스를 똑같이 찾는다(rules.normalize.MERGED_SIDO).
REGION_LABELS = {"전남광주통합특별시": "전남·광주"}      # 예전 세션에 통합 표기가 남아 있을 때만 쓰는 표시 이름
REGION_CHOICES = sorted(SIDO)
SKIP_QUESTION = "이 질문은 건너뛸게요."
# apply_channel(0~2/N): 서비스 마스터의 신청 경로 값. apply_barrier와 같은 척도(docs/services_master_spec.md D절, 표본 5,288건 중
# 0~2는 apply_barrier와 모두 일치): 0 온라인 완결·명시적 신청 불필요 / 1 전화·우편·이메일·팩스·대리 / 2 방문. 그 밖의 값은 뜻을 몰라 숨김
APPLY_CHANNEL = {"0": "온라인으로 신청할 수 있어요 (따로 신청이 필요 없는 경우 포함)",
                 "1": "전화·우편·이메일·팩스·대리로 신청해요",
                 "2": "직접 방문해서 신청해요"}
ACTIVITY = {  # activity_range 값 → 화면 문구
    "방 안": "방에서 거의 안 나와요",
    "집 안": "방에서는 나오지만 집 밖은 잘 안 나가요",
    "집 근처": "편의점처럼 집 근처까지만 나가요",
    "필요하면 외출": "필요하면 밖에 나가요",
}
PHONE = {"없음": "전화는 괜찮아요", "부담": "부담스럽지만 가능해요", "불가": "지금은 전화하기 어려워요"}
VISIT = {"없음": "직접 가도 괜찮아요", "부담": "부담스럽지만 갈 수 있어요", "불가": "지금은 가기 어려워요"}
CHANNEL_LABELS = {"전화": PHONE, "방문": VISIT}
CONTACT = {"상관없어요": None, "사람을 안 만나도 되는 것만": 0, "일대일까지 괜찮아요": 1}   # contact_burden 상한
REPEAT = {"상관없어요": None, "한 번이면 끝나는 것": 0, "몇 번까지는 괜찮아요": 1}          # repeat_burden 상한
INFO_COLS = ["summary_text", "detail_url", "region_sido", "source",
             "contact_burden", "repeat_burden", "use_outing", "apply_interview"]
BURDEN_COLS = ["contact_burden", "repeat_burden", "use_outing"]
DETAIL_COLS = ["servNm", "source", "region_sido", "region_sgg", "summary_text", "target_text", "criteria_text",
               "benefit_text", "apply_text", "apply_channel", "apply_interview", "contact_json", "detail_url"]
EXAMPLES = [("⌂", "월세와 생활비가\n걱정돼요"), ("↗", "일하고 싶은데\n시작이 어려워요"),
            ("◌", "밖에 나가는 게 부담스러워요"), ("☷", "누군가와 이야기하고 싶어요")]
PIPE_CHOICES = ["시연용 가짜", "실제 흐름(가짜 모델)", "실제 모델"]
SCENARIOS = ["일반", "위기", "욕구 없음", "결과 없음", "조건으로 전부 제외", "일부 오류", "오류"]
MAX_CHAT_Q = 3          # 채팅에서 먼저 묻는 질문 수 (참고 화면 chatCount>=3과 같음). 나머지는 정리표에서 선택 보완
TOP_CARDS = 5
UNKNOWN = "아직 알려주지 않으셨어요"

CHAT_SLOTS = {  # 채팅 질문 (참고 화면 문구). 백엔드 확인 조작으로 보낸다
    "need": "지금 가장 먼저 덜고 싶은 어려움은 어떤 쪽일까요?",
    "region_sido": "살고 계신 시·도를 알려주실 수 있을까요? 가까운 도움을 찾아보는 데 쓸게요.",
    "activity_range": "요즘 집 밖에 나가는 건 어느 정도 괜찮으세요?",
    "age": "나이를 알려주실 수 있을까요? 지원 연령을 확인할 때만 사용할게요.",
    "phone": "도움을 받는 과정에서 전화로 이야기하는 건 어떠세요?",
}
CHAT_ORDER = ["region_sido", "activity_range", "age", "phone"]
NEED_WORDS = [("경제", r"(생활비|경제|돈)(이요|요)?"), ("미취업(구직)", r"(취업|일자리|구직)(이요|요)?"),
              ("건강", r"(마음|건강|상담)(이요|요)?"), ("주거", r"(주거|살 곳|집)(이요|요)?")]

GREETING = ("안녕하세요. 반 걸음이에요.  \n요즘 가장 마음에 걸리는 일이 있나요?  \n"
            "정리되지 않은 말이어도 괜찮아요.")

# 말투 수위별 문구 (초안, 팀에서 다듬을 것). 1 아주 부드럽고 짧게 / 2 작은 다음 행동 하나 / 3 구체적으로 바로
TONE_COPY = {
    1: {
        "to_result": "찾은 서비스를 보여 드릴게요. 지금 바로 무언가 하지 않아도 괜찮아요.",
        "half_step": "지금 바로 안 해도 괜찮아요. 안내 페이지를 한 번 눈으로 훑어보기만 해도 반 걸음이에요.",
        "empty": "지금은 맞는 서비스를 찾지 못했어요. 괜찮아요. 생각나는 게 있을 때 다른 말로 다시 적어 주세요.",
        "all_excluded": "찾은 서비스가 있었지만, 고르신 조건과 맞지 않았어요. 괜찮아요. 다른 말로 다시 적어 주셔도 돼요.",
    },
    2: {
        "to_result": "맞을 만한 서비스를 찾았어요. 하나만 골라서 살펴보면 돼요.",
        "half_step": "안내 페이지에서 신청 방법까지만 읽어 보세요. 신청은 다음에 해도 돼요.",
        "empty": "지금 조건으로는 맞는 서비스를 찾지 못했어요. 요즘 가장 신경 쓰이는 걸 한 가지만 다른 말로 적어 주시겠어요?",
        "all_excluded": "찾은 서비스가 있었지만, 고르신 조건에 모두 걸렸어요. 조건을 하나만 덜 적어서 다시 적어 볼까요?",
    },
    3: {
        "to_result": "조건에 맞는 서비스를 찾았어요. 신청 방법까지 같이 보여 드릴게요.",
        "half_step": "오늘 문의할 곳에 연락하거나, 가 볼 날짜를 하나 정해 보세요. 준비물이 있으면 미리 챙겨 두세요.",
        "empty": "지금 조건으로는 맞는 서비스를 찾지 못했어요. 원하는 도움을 조금 더 구체적으로 적어 주시면 다시 찾아볼게요.",
        "all_excluded": "찾은 서비스가 있었지만, 고르신 조건에 모두 걸렸어요. 나이나 지역 같은 조건을 바꿔서 다시 적어 주시면 다시 찾아볼게요.",
    },
}
MSG = {  # 수위와 상관없는 안내 문구 (초안)
    "explore": "어떤 도움이 필요한지 아직 잘 모르겠어요. 괜찮아요. 요즘 가장 신경 쓰이는 걸 한 가지만 더 적어 주시겠어요?",
    "error": "잠시 문제가 생겨서 찾지 못했어요. 적어 주신 내용은 그대로 있어요.",
    "detail_none": "자세히 고른 조건에 맞는 서비스가 없어요. 조건을 줄여 보세요.",
    "summary": "이야기해 주신 내용을 모아 봤어요. 이미 알려주신 건 채워 두었고, 아직 모르는 항목만 표시했어요. "
               "더 이야기하지 않고 지금 조건으로 찾아봐도 괜찮아요.",
    "not_parsed": "말씀해 주신 답은 그대로 남겨 두었어요. 이 표현은 정확히 알아듣지 못해서 조건으로 정하지 않았어요. "
                  "아래에서 가까운 답을 고르거나 건너뛰어도 괜찮아요.",
    "safety": "지금은 서비스 추천을 잠시 멈추고, 안전을 위한 연락 방법을 먼저 보여 드릴게요.",
    "partial": "일부 검색 경로가 응답하지 않아 가능한 결과부터 보여 드려요.",
    "footer": "실제 지원 대상과 신청 가능 여부는 기관의 최신 안내를 확인해 주세요.",
}
EXPLORE_DOMAINS = [("경제", "생활비"), ("주거", "주거"), ("건강", "건강"),
                   ("미취업(구직)", "취업"), ("교육훈련", "교육 & 훈련"), ("관계참여", "사회참여"),
                   ("일상·여가", "일상 & 여가"), ("노동(일하는 중)", "근로 & 권익"), ("미래설계", "진로")]
EXPLORE_MAX = 20        # 서비스 둘러보기 표시 수
UNSURE_MAX = 6          # 정보 부족 탐색 표시 수 (분야당 1개)
UNSURE_BTN = "무엇이 필요한지 모르겠어요."
# 일상 대화(인사·고마움·잡담)만 외부 AI(Google Gemini)로 보낸다. 위기 확인을 통과하고 필요한 도움이 없을 때만.
CHITCHAT_NOTICE = ("일상 대화는 외부 AI(Google Gemini)로 처리돼요. "
                   "복지 서비스 찾기와 위기 안내는 이 서비스 안에서만 처리해요.")
CHITCHAT_AI_NOTE = "AI가 만든 답이에요."
CHITCHAT_ABOUT = ("<p>인사·고마움 같은 일상 대화 문장(앞선 일상 대화 최근 3턴 포함)만 외부 AI(Google Gemini)로 전송돼요. 힘듦·도움이 보이는 말, 복지 상담 내용, 나이·지역 등 조건은 "
                  "보내지 않고, 위기 확인은 그보다 먼저 이 서비스 안에서 해요. 설정된 키가 없거나 연결이 안 되면 정해진 "
                  "기본 답으로 대신해요.</p>")
PAGE_NAMES = {"explore": "서비스 둘러보기", "unsure": "서비스 먼저 살펴보기", "home": "대화로 찾아보기", "chat": "반 걸음과 대화", "confirm": "함께 정리한 내용",
              "results": "나에게 맞는 도움", "detail": "서비스 자세히", "visit": "담은 서비스",
              "activity_offer": "지금 편한 방향 고르기", "activities": "👣 작은 활동 살펴보기", "rest": "잠시 쉬어가기",
              "about": "이용 안내·개인정보", "safety": "지금 필요한 도움", "dev": "개발자"}
CONV_PAGES = ("home", "chat", "confirm", "results", "safety", "activity_offer", "activities", "rest")
MODE_BADGE = {"시연용 가짜": "◌ 시연용 가짜 · 화면 확인용",
              "실제 흐름(가짜 모델)": "● Pipeline 연결 · 테스트 모델",
              "실제 모델": "● 실제 Pipeline 연결"}

# ── 디자인 (참고 style.css의 색·글꼴·말풍선·카드). 고정 문자열만 들어간다 ──
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=Noto+Serif+KR:wght@400;500;600&display=swap');
:root{--bg:#f8f8f3;--paper:#fffefa;--ink:#263e35;--muted:#6e7b72;--green:#365f4e;--soft:#eaf0e5;--line:#dfe5d9;--orange:#bb8052;--shadow:0 12px 45px #263e3506}
/* 글꼴: 본문·버튼·입력칸·카드 제목은 고딕. 명조는 화면 큰 제목(h1·상세 제목 h2·hero)과 안내 카드 제목(.aside-h)만 */
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--ink)!important}
.stApp,.stApp p,.stApp li,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
.stApp span:not([data-testid="stIconMaterial"]):not([class*="material"]),[data-baseweb="popover"] li,[data-baseweb="popover"] li *{font-family:'Noto Sans KR','Malgun Gothic',sans-serif}
/* Streamlit 아이콘은 Material Symbols 글꼴 그대로(덮어쓰면 아이콘 이름이 글자로 보임) */
[data-testid="stIconMaterial"],[data-testid="stSidebarCollapseButton"] span,[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stExpander"] summary span[data-testid="stIconMaterial"],.material-symbols-rounded{font-family:'Material Symbols Rounded'!important;font-weight:400;letter-spacing:normal;text-transform:none;font-feature-settings:'liga'}
.stApp h3,.stApp h4,.stApp h5,.stApp h6{font-weight:600;color:var(--ink);letter-spacing:-.4px}
.stApp h1,.stApp h2{font-family:'Noto Serif KR','Batang',serif;font-weight:500;color:var(--ink);letter-spacing:-.8px}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"],.block-container{max-width:1080px;padding-top:2.2rem}
.stApp p{overflow-wrap:anywhere}
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong{color:var(--ink)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,.stApp small{color:var(--muted)!important}
/* 입력칸·선택칸·숫자칸·라벨 */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,.stApp label{color:var(--ink)!important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],[data-testid="stNumberInputContainer"],
[data-testid="stTextInputRootElement"]{background:var(--paper)!important;border-color:var(--line)!important;color:var(--ink)!important}
[data-baseweb="select"] *,[data-baseweb="input"] input,[data-testid="stNumberInputField"],.stApp input,.stApp textarea{color:var(--ink)!important;background:var(--paper)!important;-webkit-text-fill-color:var(--ink)}
.stApp input::placeholder,.stApp textarea::placeholder,[data-baseweb="select"] [aria-disabled="true"]{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)}
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputClearButton"]{background:var(--paper)!important;color:var(--ink)!important}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li,[data-baseweb="menu"],[role="listbox"],[role="option"]{background:var(--paper)!important;color:var(--ink)!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:var(--soft)!important}
/* 필요한 도움 칩 (st.pills) */
[data-testid="stBaseButton-pills"]{background:var(--paper)!important;border:1px solid var(--line)!important;color:var(--ink)!important;border-radius:30px}
[data-testid="stBaseButton-pillsActive"]{background:#e4eddc!important;border:1px solid #7d9a6c!important;color:#325235!important;border-radius:30px}
[data-testid="stBaseButton-pills"] *,[data-testid="stBaseButton-pillsActive"] *{color:inherit!important}
[data-testid="stBaseButton-pills"]:hover{border-color:#a5b598!important;background:var(--soft)!important}
[data-testid="stBaseButton-pills"]:focus,[data-testid="stBaseButton-pillsActive"]:focus{box-shadow:none!important}
/* 사이드바 = 좌측 레일 */
[data-testid="stSidebar"]{background:#eff1e9;border-right:1px solid var(--line)}
.brand{font-size:26px;letter-spacing:-1.7px;font-weight:600;display:flex;align-items:center;gap:10px;color:var(--ink)}
.brand-mark{display:inline-block;font-size:34px;transform:rotate(-32deg);line-height:1}
.brand-dot{color:var(--orange);margin-left:-7px}
.brand-sub{font-size:11px;color:var(--muted);margin:4px 0 22px 43px}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"],[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{justify-content:flex-start!important;text-align:left;min-height:42px;height:42px;padding:0 13px!important;border-radius:10px;width:100%;border:0!important;box-shadow:none!important;font-weight:400}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]>div,[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]>div,
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p,[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p{justify-content:flex-start!important;text-align:left!important;width:100%;margin:0}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]{background:transparent!important}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"],[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] *{color:#65756a!important}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover{background:#e6ebdf!important}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:#eaf0e5!important;font-weight:600}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *{color:var(--ink)!important}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{justify-content:flex-start!important;border-radius:12px;min-height:42px;height:42px;padding:0 13px!important;margin-bottom:10px}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]>div{justify-content:flex-start!important;text-align:left!important;width:100%;margin:0}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.3rem}
/* 사이드바 스크롤 막대: 얇게(스크롤은 그대로) */
[data-testid="stSidebar"] *{scrollbar-width:thin;scrollbar-color:#d3dccb transparent}
[data-testid="stSidebar"] *::-webkit-scrollbar{width:4px;height:4px}
[data-testid="stSidebar"] *::-webkit-scrollbar-thumb{background:#d3dccb;border-radius:4px}
[data-testid="stSidebar"] *::-webkit-scrollbar-track{background:transparent}
.rail-note{padding:30px 6px 10px}
.rail-note .leaf{font-size:28px;color:#879e78}
.rail-note p{font-size:15px;font-weight:500;line-height:1.9;letter-spacing:-.5px;margin:6px 0}
.rail-note small{font-size:11px;color:var(--muted)}
/* 상단 위치 표시 */
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e8ebe2;padding:0 0 14px;margin-bottom:26px}
.breadcrumb{font-size:12px;color:var(--muted)}.breadcrumb span{margin:0 12px;color:#c3cbbb}.breadcrumb b{font-weight:400;color:var(--ink)}
.mode{font-size:10px;background:#eef1e8;border:1px solid var(--line);border-radius:30px;padding:4px 11px;color:#69795e}
/* 버튼 */
[data-testid="stBaseButton-primary"],[data-testid="stBaseLinkButton-primary"]{background:var(--green)!important;border-color:var(--green)!important;border-radius:10px}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primary"]:focus,[data-testid="stBaseLinkButton-primary"]:hover{background:#2c4f40!important;border-color:#2c4f40!important}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primary"] *,[data-testid="stBaseLinkButton-primary"],[data-testid="stBaseLinkButton-primary"] *{color:#fff!important}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseLinkButton-secondary"]{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:10px;color:var(--ink)!important}
[data-testid="stBaseButton-secondary"] *,[data-testid="stBaseLinkButton-secondary"] *{color:var(--ink)!important}
[data-testid="stBaseButton-secondary"]:hover,[data-testid="stBaseLinkButton-secondary"]:hover{border-color:#a5b598!important;background:var(--soft)!important}
[data-testid="stBaseButton-tertiary"],[data-testid="stBaseButton-tertiary"] *{color:var(--muted)!important}
.stApp button:focus-visible,.stApp a:focus-visible{outline:3px solid #a2b18d;outline-offset:3px}
/* 말풍선: 봇은 왼쪽 아이콘 + '반 걸음' 이름 + 말풍선, 사용자는 오른쪽 정렬 */
[data-testid="stChatMessage"]{display:flex!important;flex-direction:row!important;justify-content:flex-start!important;align-items:flex-start!important;gap:12px!important;background:transparent!important;padding:0!important;width:100%}
/* 블록 사이 기본 간격(gap 16px)을 감안: 메시지 위 26px / 아래 16px, 연속 봇 메시지는 8px */
[data-testid="stChatMessage"]{margin:0!important}
[data-testid="stElementContainer"]:has([data-testid="stChatMessage"]){margin-top:10px}
[data-testid="stElementContainer"]:has([data-testid="stChatMessage"])+[data-testid="stElementContainer"]:has([data-testid="stChatMessage"]){margin-top:-8px}
[data-testid="stChatMessage"]>div:first-child{flex:0 0 32px;width:32px!important;height:32px!important;min-width:32px;border-radius:50%!important;background:#e0e8d6!important;display:flex;align-items:center;justify-content:center;font-size:19px;color:var(--ink)}
[data-testid="stChatMessageContent"]{display:block!important;flex:0 1 auto!important;width:auto!important;max-width:85%;margin:0!important;padding:0!important;background:transparent!important;border:0!important;color:var(--ink)!important}
[data-testid="stChatMessageContent"]::before{content:"반 걸음";display:block;font-size:11px;font-weight:600;color:var(--ink);margin:2px 0 6px}
[data-testid="stChatMessageContent"]>div{background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:0 18px 18px 18px;padding:14px 20px;box-shadow:var(--shadow);font-size:14px;line-height:1.9;color:var(--ink)!important}
[data-testid="stChatMessageContent"] *{color:var(--ink)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])>div:first-child{display:none!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]::before{content:"나";text-align:right;font-weight:400;font-size:10px;color:var(--muted)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]>div{background:#e7eddf!important;border-color:#dce4d2!important;border-radius:18px 0 18px 18px;box-shadow:none}
/* 채팅 입력칸 */
[data-testid="stChatInput"],[data-testid="stChatInput"]>div{border:1px solid #b9c7b1!important;border-radius:18px;background:var(--paper)!important}
[data-testid="stChatInputTextArea"]{background:var(--paper)!important;color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
[data-testid="stChatInputSubmitButton"]{background:var(--green)!important;color:#fff!important;border-radius:50%}
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:var(--bg)!important}
/* 카드·패널 (key로 지정한 container) */
div[class*="st-key-card-"],div[class*="st-key-panel-"]{background:var(--paper)!important;border:1px solid var(--line);border-radius:18px;padding:22px 24px;box-shadow:var(--shadow);color:var(--ink)}
div[class*="st-key-card-"]:hover{box-shadow:0 12px 28px #314f4010}
div[class*="st-key-urgent"]{background:#fbf0e7!important;border:1px solid #e6ccba;border-radius:18px;padding:22px 24px;color:var(--ink)}
div[class*="st-key-aside"]{background:#f1f3ea!important;border:1px solid var(--line);border-radius:18px;padding:20px;color:var(--ink)}
[data-testid="stExpander"] details,[data-testid="stExpander"] summary{background:var(--paper)!important;border-color:var(--line)!important;color:var(--ink)!important}
/* 첫 화면 안내 카드: 1100px 이하에서는 본문 아래로 내림 */
@media(max-width:1100px){
[data-testid="stHorizontalBlock"]:has(.st-key-aside-home){flex-wrap:wrap!important}
[data-testid="stHorizontalBlock"]:has(.st-key-aside-home)>[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important;min-width:100%!important}
.st-key-aside-home .side-art{display:none}}
/* 고정 HTML 조각 */
.eyebrow{font-size:10px;letter-spacing:2.8px;color:#78896e;font-weight:500}
.hero-title{font-family:'Noto Serif KR','Batang',serif;font-size:42px;line-height:1.6;letter-spacing:-2.4px;font-weight:500;margin:18px 0 16px;color:var(--ink)}
.hero-title em{font-style:normal;color:#66805d}
.intro{color:var(--muted);font-size:13px;line-height:1.95;margin-bottom:10px}
.page-head{margin-bottom:22px}.page-head h1{font-size:30px;line-height:1.6;margin:8px 0;padding:0}
.page-head p{font-size:12.5px;color:var(--muted);margin:0}
.subtle{font-size:11.5px;color:var(--muted)}
.side-art{height:130px;position:relative;margin:10px 0}
.stone{position:absolute;border:1px solid #c1cdb6;background:#dce6d0;border-radius:50%;width:66px;height:35px;transform:rotate(-22deg);left:10px;bottom:10px}
.stone:nth-child(2){width:52px;height:29px;left:65px;bottom:52px;background:#e5ebd9}
.stone:nth-child(3){width:39px;height:23px;left:111px;bottom:92px;background:#edf0e3}
.aside-h{font-family:'Noto Serif KR',serif;font-size:17px;line-height:1.9;margin:0 0 10px}
.side-list div{display:flex;gap:10px;font-size:11.5px;color:var(--muted);margin:14px 0}
.side-list b{font-size:10px;font-weight:400;background:#e9eddf;border-radius:50%;min-width:20px;height:20px;text-align:center;color:#708263}
.side-list strong{display:block;color:#455c4b;font-weight:500}
.footer{text-align:center;color:#84907f;font-size:10.5px;padding:40px 0 10px}
@media(max-width:800px){.hero-title{font-size:32px}.page-head h1{font-size:24px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* Reference HTML proportions. Use widget/key selectors supported across Streamlit versions. */
.stApp{font-size:14px;line-height:1.75}
.stApp p,.stApp li{font-size:13px;line-height:1.95}
.stApp [data-testid="stCaptionContainer"] p{font-size:11px;line-height:1.9}
.stApp button p,.stApp a[kind] p{font-size:12px;line-height:1.75}
.stApp h2{font-family:'Noto Sans KR','Malgun Gothic',sans-serif;font-size:19px;font-weight:600}
.stApp [data-testid="stButton"] button,.stApp [data-testid="stLinkButton"] a{border:1px solid var(--line);border-radius:10px;background:var(--paper);color:var(--ink);padding:11px 18px;min-height:44px;box-shadow:none}
.stApp [data-testid="stButton"] button[kind="primary"]{background:var(--green)!important;border-color:var(--green)!important}
.stApp [data-testid="stButton"] button[kind="primary"] p{color:white!important}
.stApp [data-testid="stButton"] button[kind="tertiary"]{border:0;background:transparent}
[data-testid="stMainBlockContainer"]{max-width:1080px;padding:0 45px 40px}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]{gap:16px}
.topbar{height:86px;margin:0 -45px 32px;padding:0 45px;border-bottom:1px solid #e8ebe2}
[data-testid="stSidebar"]{width:246px!important;min-width:246px!important;max-width:246px!important}
[data-testid="stSidebarHeader"]{display:none}
[data-testid="stSidebarUserContent"]{padding:39px 25px!important}
[data-testid="stSidebarUserContent"]>[data-testid="stVerticalBlock"]{min-height:calc(100dvh - 78px);gap:0!important}
[data-testid="stSidebar"] .brand{font-size:26px;line-height:1.4;gap:10px}
[data-testid="stSidebar"] .brand-mark{font-size:36px}
[data-testid="stSidebar"] .brand-sub{font-size:11px;margin:7px 0 35px 43px;letter-spacing:-.2px}
[data-testid="stSidebar"] .st-key-nav_new{margin-bottom:30px}
[data-testid="stSidebar"] [data-testid="stButton"] button{justify-content:flex-start!important;text-align:left;width:100%;border:0!important;padding:12px 13px!important;min-height:48px!important;height:auto!important;background:transparent!important;color:#65756a!important}
[data-testid="stSidebar"] [data-testid="stButton"] button p{font-size:13px!important;color:#65756a!important;text-align:left;line-height:1.75;margin:0}
[data-testid="stSidebar"] .st-key-nav_new button{background:var(--green)!important;border-radius:12px!important;min-height:50px!important}
[data-testid="stSidebar"] .st-key-nav_new button p{color:#fff!important}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]{background:#e1e8d9!important;font-weight:600}
[data-testid="stSidebar"] .st-key-rail-nav>[data-testid="stVerticalBlock"]{gap:8px}
[data-testid="stSidebar"] .st-key-nav_conv button p::before{content:'◌';margin-right:11px;color:#819379}
[data-testid="stSidebar"] .st-key-nav_explore button p::before{content:'▦';margin-right:11px;color:#819379}
[data-testid="stSidebar"] .st-key-nav_visit button p::before{content:'◷';margin-right:11px;color:#819379}
[data-testid="stSidebar"] .st-key-nav_visit button{position:relative}
[data-testid="stSidebar"] .st-key-nav_visit button::after{content:var(--visit-count,'0');position:absolute;right:13px;top:50%;transform:translateY(-50%);background:#d7e0ce;border-radius:4px;padding:0 6px;font-size:10px;line-height:20px;color:#65756a}
[data-testid="stSidebar"] .st-key-rail-note{margin-top:auto!important}
.rail-note{padding:38px 9px 30px}
.rail-note .leaf{font-size:30px;color:#879e78}
.stApp .rail-note p{font-family:'Noto Serif KR','Batang',serif;font-size:17px;font-weight:400;line-height:1.9;letter-spacing:-.5px;margin:17px 0}
.rail-note small{font-size:11px;color:var(--muted)}
.st-key-rail-bottom{border-top:1px solid #dce2d5;padding-top:20px}
[data-testid="stSidebar"] .st-key-rail-bottom [data-testid="stVerticalBlock"]{gap:9px}
[data-testid="stSidebar"] .st-key-rail-bottom [data-testid="stButton"] button{padding:0!important;min-height:20px!important;border-radius:0!important;background:transparent!important}
[data-testid="stSidebar"] .st-key-rail-bottom [data-testid="stButton"] button p{font-size:11px!important;font-weight:400!important;color:#647468!important}
.stApp .rail-copyright{font-size:10px;color:#8a958a;margin:14px 0 0}
.st-key-home-layout{padding-top:0}
.st-key-home-layout>[data-testid="stHorizontalBlock"]{gap:50px!important;flex-wrap:nowrap!important}
.st-key-home-layout>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:0!important}
.stApp .hero-title{font-size:43px;line-height:1.65;letter-spacing:-2.6px;margin:21px 0 19px}
.stApp .intro{font-size:13px;line-height:1.95;color:var(--muted);margin-bottom:18px}
.st-key-aside-home{padding:22px!important}
.st-key-aside-home .side-art{display:block!important;height:155px;margin:15px 0}
.side-art::after{content:'✳';position:absolute;right:12px;top:0;font-size:29px;color:#acbb99}
.stApp .aside-h{font-family:'Noto Serif KR','Batang',serif;font-size:17px;line-height:1.9;margin:0 0 12px}
.stApp .aside-copy{font-size:11px;color:var(--muted);line-height:1.9;margin:0}
.side-list{display:grid;gap:18px;margin-top:27px}
.side-list div{display:flex;gap:10px;font-size:11px;color:var(--muted);margin:0}
.side-list strong{font-weight:500}
.page-head{margin-bottom:14px}.page-head h1{font-size:30px;line-height:1.7;letter-spacing:-1.5px}
.stApp .page-head p{font-size:12px;color:var(--muted)}
.stApp .reference-panel{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:25px;margin:0 0 18px}
.stApp .reference-prose h2{font-family:'Noto Sans KR','Malgun Gothic',sans-serif;font-size:16px;line-height:1.8;font-weight:600;color:var(--ink);margin:30px 0 12px;letter-spacing:-.4px}
.stApp .reference-prose h2:first-child{margin-top:0}
.stApp .reference-prose p{font-size:13px;line-height:2;color:#647163;margin:0 0 14px}
.st-key-panel-visit-empty{text-align:center;padding:50px 20px!important}
.stApp .empty-symbol{font-size:40px;color:#92a47e;line-height:1.5}
.stApp .empty-title{font-family:'Noto Serif KR','Batang',serif;font-size:24px;font-weight:500;line-height:1.8;margin:12px 0}
.stApp .empty-copy{font-size:12px;color:var(--muted);margin:0 0 18px}
.st-key-panel-visit-empty [data-testid="stButton"]{display:flex;justify-content:center}
.support-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:9px 0}
.support-card{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:26px;display:flex;flex-direction:column}
.support-card.urgent{background:#fbf0e7;border-color:#e6ccba}
.stApp .support-card h2{font-family:'Noto Sans KR','Malgun Gothic',sans-serif;font-size:19px;font-weight:600;line-height:1.65;margin:17px 0;letter-spacing:-.6px}
.stApp .support-card p{font-size:12px;line-height:1.9;color:var(--muted);margin:0 0 12px;flex:1}
.support-card .tag{align-self:flex-start;background:#edf1e5;color:#647653;border-radius:5px;padding:2px 7px;font-size:10px}
.support-card .support-link{display:flex;align-items:center;justify-content:center;text-decoration:none;border:1px solid var(--line);border-radius:10px;min-height:48px;padding:11px 12px;margin-top:12px;font-size:12px;color:var(--green);background:var(--paper);text-align:center}
.support-card .support-link.primary{background:var(--green);border-color:var(--green);color:white}
.stApp .connection-notice{background:#edf1e5;border-radius:12px;padding:13px 17px;font-size:11px;line-height:1.9;color:#627154;margin:8px 0}
.footer{font-size:10px;color:#84907f;padding:40px 0 10px}
@media(min-width:1450px){[data-testid="stSidebar"]{width:270px!important;min-width:270px!important;max-width:270px!important}[data-testid="stSidebarUserContent"]{padding:44px 30px!important}.hero-title{font-size:48px!important}}
@media(min-width:801px) and (max-width:1100px){[data-testid="stSidebar"]{width:210px!important;min-width:210px!important;max-width:210px!important}[data-testid="stSidebarUserContent"]{padding:30px 18px!important}[data-testid="stMainBlockContainer"]{padding:0 30px 35px}.topbar{margin-left:-30px;margin-right:-30px;padding:0 30px}.st-key-home-layout>[data-testid="stHorizontalBlock"]{gap:28px!important}.stApp .hero-title{font-size:35px;letter-spacing:-2px}.st-key-aside-home{padding:18px!important}}
@media(max-width:800px){.support-grid{grid-template-columns:1fr}.st-key-home-layout>[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}.st-key-home-aside{display:none}.hero-title{font-size:37px!important}[data-testid="stMainBlockContainer"]{padding:0 24px 32px}.topbar{margin-left:-24px;margin-right:-24px;padding:0 24px;height:59px}.stApp .page-head h1{font-size:26px}}


/* Remove Streamlit's reserved scrollbar gutters inside the custom rail. */
[data-testid="stSidebarContent"]{padding:0!important;scrollbar-gutter:auto!important}
[data-testid="stSidebarUserContent"]{width:100%!important;box-sizing:border-box!important}
[data-testid="stSidebarUserContent"]>div>[data-testid="stVerticalBlock"]{min-height:calc(100dvh - 78px);gap:0!important}
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(>.st-key-rail-note){margin-top:auto!important}
[data-testid="stSidebar"] .st-key-rail-nav{gap:8px!important}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]{gap:50px!important;flex-wrap:nowrap!important}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:0!important}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:first-child{flex:1 1 0!important}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{flex:0 0 220px!important}
.stApp .page-head h1,.stApp .page-head h1 *,.stApp .empty-title,.stApp .empty-title *{font-family:'Noto Serif KR','Batang',serif!important;font-weight:500}
.stApp .page-head h1{font-size:30px!important;line-height:1.7!important}
.mode{text-decoration:none}
@media(min-width:801px) and (max-width:1100px){.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]{gap:28px!important}.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{flex:0 0 190px!important}[data-testid="stSidebarUserContent"]>div>[data-testid="stVerticalBlock"]{min-height:calc(100dvh - 60px)}}
@media(max-width:800px){.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{display:none!important}.stApp .page-head h1{font-size:26px!important}}


@media(max-width:800px){[data-testid="stSidebarHeader"]{display:flex!important}[data-testid="stHeader"]{display:block!important;background:transparent!important}}
.stApp .mode,.stApp a.mode{color:#69795e!important;text-decoration:none!important}
.stApp .mode:hover{border-color:#a5b598}


.stApp .empty-symbol,.stApp .empty-title,.stApp .empty-copy{text-align:center!important}
.stApp .empty-title,.stApp .reference-prose h2,.stApp .support-card h2{padding:0!important}
.st-key-visit_browse{align-self:center!important}
.st-key-home_input [data-testid="stChatInput"]{min-height:130px;padding:12px}
.st-key-home_input textarea{min-height:82px!important;font-size:13px!important}
[data-testid="stChatMessage"]>div:first-child{font-size:0!important}
[data-testid="stChatMessage"]>div:first-child>*{display:none!important}
[data-testid="stChatMessage"]>div:first-child::after{content:'◒';font-size:23px;line-height:1;transform:rotate(-25deg)}


[data-testid="stChatInput"]>div{border:0!important;background:transparent!important;box-shadow:none!important}
@media(min-width:801px) and (max-height:800px){.rail-note{padding:20px 9px 15px}.stApp .rail-note p{font-size:15px;line-height:1.8;margin:12px 0}[data-testid="stSidebar"] .brand-sub{margin-bottom:28px}}


.small-activity-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.small-activity-card{background:#f0f4e8;border:1px solid #dce4d3;border-radius:17px;padding:23px 19px}
.stApp .small-activity-card .activity-number{display:block;font-family:'Noto Serif KR',serif;font-size:25px;color:#a2b38d;margin-bottom:16px}
.stApp .small-activity-card .activity-meta{font-size:10px;color:var(--muted)}
.stApp .small-activity-card h3{font-size:clamp(19px,2vw,23px);line-height:1.5;font-weight:700;letter-spacing:-.4px;padding:0;margin:10px 0 0;word-break:keep-all;overflow-wrap:anywhere}
.stApp .small-activity-card p{font-size:11px;line-height:1.9;color:var(--muted)}
@media(max-width:800px){.small-activity-grid{grid-template-columns:1fr}}

/* Reading sizes apply last so earlier reference styles cannot shrink body copy. */
.stApp{--muted:#566451}
.stApp p,.stApp li,[data-testid="stDialog"] p,[data-testid="stDialog"] li{font-size:16px;line-height:1.8}
.stApp [data-testid="stCaptionContainer"] p,[data-testid="stDialog"] [data-testid="stCaptionContainer"] p{font-size:14px;line-height:1.7;color:#566451!important}
.stApp button p,.stApp a[kind] p{font-size:15px;line-height:1.5}
.stApp .intro,.stApp .reference-prose p{font-size:16px;line-height:1.85}
.stApp .page-head p,.stApp .aside-copy,.stApp .support-card p,.stApp .small-activity-card p{font-size:15px;line-height:1.8}
.stApp .small-activity-card .activity-meta,.stApp .connection-notice{font-size:14px}
.st-key-home_input textarea{font-size:16px!important}

/* ── 화면 배치: 넓은 본문 · 아래 고정 입력창 · 읽기 쉬운 말풍선 · 접히는 사이드바 ── */
[data-testid="stMainBlockContainer"]{max-width:1280px!important}
[data-testid="stBottomBlockContainer"]{max-width:1280px!important;margin:0 auto;padding:6px 45px 10px!important;box-sizing:border-box}
[data-testid="stBottomBlockContainer"] [data-testid="stElementContainer"]{margin:0!important;position:relative!important}
[data-testid="stBottomBlockContainer"]>[data-testid="stVerticalBlock"]{gap:6px!important}
@media(min-width:801px) and (max-width:1100px){[data-testid="stBottomBlockContainer"]{padding:6px 30px 10px!important}}
@media(max-width:800px){[data-testid="stBottomBlockContainer"]{padding:6px 24px 10px!important}}
/* 아래 고정 영역: 예시 버튼은 한 줄 알약, 안내 문구는 작게 */
[data-testid="stBottom"] [data-testid="stButton"] button{min-height:34px;padding:4px 12px;border-radius:999px;height:auto}
[data-testid="stBottom"] [data-testid="stButton"] button p{font-size:13.5px!important;line-height:1.4;white-space:normal}
[data-testid="stBottom"] [data-testid="stHorizontalBlock"]{gap:8px!important}
.st-key-bottom-hint [data-testid="stCaptionContainer"] p{font-size:12px!important;line-height:1.4;margin:0;text-align:center}
.st-key-home_input [data-testid="stChatInput"]{min-height:0!important;padding:0!important}
.st-key-home_input textarea{min-height:44px!important}
/* 대화 말풍선: 글자색·배경 대비를 높이고 폭·크기를 키움 */
[data-testid="stChatMessageContent"]{max-width:92%!important}
[data-testid="stChatMessageContent"]::before{font-size:12.5px;color:#16241d}
[data-testid="stChatMessageContent"]>div{background:#ffffff!important;border:1.5px solid #a9bc9d!important;font-size:16px!important;line-height:1.85!important;padding:16px 22px!important;box-shadow:0 2px 12px #263e3512!important}
[data-testid="stChatMessageContent"] p,[data-testid="stChatMessageContent"] li,[data-testid="stChatMessageContent"] span{color:#16241d!important;font-size:16px!important;line-height:1.85}
[data-testid="stChatMessageContent"] strong{color:#0d1a14!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]>div{background:#dfe9d4!important;border:1.5px solid #a3b894!important;box-shadow:none!important}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]::before{font-size:12px;color:#566451}
/* 봇 아바타: 이모지·기호 대신 '반' 글자 원 */
[data-testid="stChatMessage"]>div:first-child{background:#e0e8d6!important;border:1px solid #c9d6bb;flex-basis:38px;width:38px!important;height:38px!important;min-width:38px}
[data-testid="stChatMessage"]>div:first-child::after{content:'';display:block;width:24px;height:22px;transform:none;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-6 -16 164 150'%3E%3Cg stroke='%238ea27c' stroke-width='5'%3E%3Cellipse cx='43' cy='102.5' rx='33' ry='17.5' fill='%23b4c6a0' transform='rotate(-22 43 102.5)'/%3E%3Cellipse cx='91' cy='63.5' rx='26' ry='14.5' fill='%23c6d4b6' transform='rotate(-22 91 63.5)'/%3E%3Cellipse cx='130.5' cy='26.5' rx='19.5' ry='11.5' fill='%23d8e2cb' transform='rotate(-22 130.5 26.5)'/%3E%3C/g%3E%3Cg stroke='%236f8a5a' stroke-width='6' stroke-linecap='round'%3E%3Cline x1='146' y1='-12' x2='146' y2='6'/%3E%3Cline x1='137' y1='-9' x2='155' y2='3'/%3E%3Cline x1='155' y1='-9' x2='137' y2='3'/%3E%3C/g%3E%3C/svg%3E") center/contain no-repeat}
/* 첫 화면: 채팅 옆 안내 카드 */
.steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:6px 0 4px}
.steps .step{background:var(--paper);border:1px solid #c9d6bb;border-radius:16px;padding:18px 20px}
.steps .step b{display:inline-block;font-size:12px;font-weight:600;background:#e4ecd9;border-radius:50%;min-width:24px;height:24px;line-height:24px;text-align:center;color:#4d6640;margin-bottom:10px}
.steps .step strong{display:block;font-size:15px;font-weight:600;color:#1f3329;margin-bottom:4px}
.steps .step span{display:block;font-size:13.5px;line-height:1.6;color:#566451}
.st-key-home-guide .steps{grid-template-columns:1fr;gap:8px}
.st-key-home-guide .step{display:grid;grid-template-columns:auto 1fr;column-gap:10px;padding:10px 14px}
.st-key-home-guide .step b{grid-row:span 2;margin:0}
.st-key-home-layout [data-testid="stChatMessageContent"]{flex:1 1 auto!important}
.st-key-home-layout [data-testid="stChatMessageContent"]>div{padding:clamp(16px,3vw,30px)!important}
.st-key-home-layout [data-testid="stChatMessageContent"] p{font-size:clamp(17px,2vw,21px)!important}
.st-key-home_input textarea,.st-key-chat_input textarea{min-height:76px!important}
.stApp .support-card .support-link{font-family:'Noto Sans KR','Malgun Gothic',sans-serif;font-size:17px;font-weight:700}
.stApp .support-card.urgent{background:#fff0ef;border-color:#e3aaa5}
.stApp .support-card.urgent .tag{background:#f8d8d5;color:#842d29}
.stApp .support-card.urgent .support-link.primary{background:#992f2d;border-color:#992f2d}
[data-testid="stNumberInput"] [data-testid="InputInstructions"]{display:none}
@media(max-width:800px){.steps{grid-template-columns:1fr}}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]{gap:40px!important}
.st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]:last-child{flex:0 0 340px!important}
.st-key-home_explore button{border:1.5px solid #a9bc9d!important;background:var(--paper)!important;border-radius:12px!important;padding:10px 22px!important}
.st-key-home_explore button:hover{background:var(--soft)!important;border-color:#7d9a6c!important}
/* 사이드바 접기·펼치기: 헤더는 두되 우측 도구만 숨기고, 펼침 버튼을 왼쪽 위에 띄움 */
[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],[data-testid="stStatusWidget"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{position:fixed!important;top:0;left:0;right:auto;width:auto;height:0!important;min-height:0!important;padding:0!important;background:transparent!important;pointer-events:none;z-index:1000}
[data-testid="stToolbar"]{background:transparent!important;pointer-events:none}
[data-testid="stExpandSidebarButton"]{position:fixed!important;top:14px;left:14px;pointer-events:auto;background:var(--paper)!important;border:1px solid var(--line)!important;border-radius:10px;z-index:1001}
[data-testid="stSidebarHeader"]{display:flex!important;position:absolute;top:8px;right:8px;width:auto;height:32px;min-height:0;padding:0;margin:0;z-index:3}
[data-testid="stSidebarHeader"] [data-testid="stLogoSpacer"],[data-testid="stSidebarHeader"] [data-testid="stSidebarLogo"]{display:none}
/* 사이드바 아래 메뉴(이용 안내·도움)는 바닥에 붙임 */
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(>.st-key-rail-bottom){margin-top:auto!important}
[data-testid="stSidebar"] .st-key-rail-bottom{margin-top:auto}
.stApp .st-key-input-notices [data-testid="stCaptionContainer"] p{font-size:11px!important;line-height:1.6;color:#6b7667!important;margin:0}
.st-key-input-notices{gap:2px!important;margin-top:4px}
.st-key-browse-controls{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px!important;align-items:stretch}
.st-key-browse-controls>div{min-width:0!important;width:100%!important}
.st-key-browse-controls>.st-key-ex_q_in{grid-column:span 2}
.stApp .st-key-browse-controls button{width:100%;height:auto;min-height:58px;padding:8px}
.st-key-browse-controls button [data-testid="stMarkdownContainer"],
.st-key-browse-controls button p{min-width:0;white-space:normal!important;overflow:visible!important;text-overflow:clip!important;word-break:keep-all;overflow-wrap:anywhere}
.st-key-ex_q_in [data-testid="stTextInputRootElement"]{min-height:58px}
.st-key-ex_q_in [data-testid="stTextInputRootElement"]::after{
  content:"";width:20px;height:20px;flex:0 0 20px;align-self:center;margin:0 14px 0 4px;
  background:var(--green);
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='10.5' cy='10.5' r='6.5'/%3E%3Cpath d='m16 16 4.5 4.5'/%3E%3C/svg%3E") center / contain no-repeat;
  pointer-events:none;
}
[data-testid="stMainBlockContainer"]{container:main-content / inline-size}
.stApp .hero-title{font-size:clamp(28px,5cqi,48px);line-height:1.5;word-break:keep-all;overflow-wrap:anywhere}
@container main-content (max-width:760px){
  .st-key-browse-controls{grid-template-columns:repeat(2,minmax(0,1fr))}
  .st-key-home-layout>[data-testid="stHorizontalBlock"],
  .st-key-home-layout>[data-testid="stLayoutWrapper"]>[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:20px!important}
  .st-key-home-layout [data-testid="stHorizontalBlock"]:has(.st-key-aside-home)>[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important;min-width:0!important}
}
.topbar{height:auto;min-height:60px;flex-wrap:wrap;gap:8px;padding-block:12px}
[data-testid="stSidebar"] .st-key-nav_visit button{padding-right:44px!important}
[data-testid="stDialog"] [data-testid="stHorizontalBlock"]{gap:20px}
@media(max-width:800px){[data-testid="stDialog"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap}[data-testid="stDialog"] [data-testid="stColumn"]{min-width:100%!important}}
</style>
"""
HERO_HTML = """
<div class="hero-title">서두르지 않아도,<br><em>반 걸음</em>은 시작이니까.</div>
"""
ASIDE_HTML = """
<div class="side-art" aria-hidden="true"><i class="stone"></i><i class="stone"></i><i class="stone"></i></div>
<div class="aside-h">오늘은,<br>알아보는 것만으로<br>충분해요.</div>
<p class="aside-copy">지금 바로 신청하지 않아도,<br>모든 질문에 답하지 않아도<br>괜찮아요.</p>

"""
STEPS_HTML = """<div class="steps">
<div class="step"><b>1</b><strong>편하게 이야기하기</strong><span>지금 필요한 도움을 찾아요.</span></div>
<div class="step"><b>2</b><strong>함께 확인하기</strong><span>이해한 내용은 바꿀 수 있어요.</span></div>
<div class="step"><b>3</b><strong>나에게 맞는 도움 보기</strong><span>신청 방법까지 살펴봐요.</span></div>
</div>
"""
HOME_HINT = "이름·연락처 같은 개인정보는 적지 않아도 돼요. 일상 대화는 외부 AI(Google Gemini)로 처리돼요."
BRAND_HTML = """
<div class="brand"><span class="brand-mark" aria-hidden="true">◒</span>반 걸음<span class="brand-dot">·</span></div>
<div class="brand-sub">나의 속도로 찾는 도움</div>
"""


# ─────────────────────────────────────────────
# 2. 시연용 가짜 파이프라인 (실제 Pipeline과 같은 모양)
# ─────────────────────────────────────────────
MOCK_SERVICES = pd.DataFrame([  # 실제 services_master.csv와 같은 열 이름, 값은 문자열("N"은 정보 없음)
    {"servId": f"MOCK{i:03d}", "servNm": f"[서비스 이름 {i + 1}]", "summary_text": "[서비스 한 줄 요약]",
     "detail_url": "https://www.bokjiro.go.kr", "contact_burden": c, "repeat_burden": r,
     "use_outing": o, "apply_interview": iv, "source": "central" if i % 2 == 0 else "local",
     "region_sido": "N" if i % 2 == 0 else "서울", "region_sgg": "N", "age_min": "N", "age_max": "N",
     "app_domain": dom,
     "target_text": "[지원 대상 원문]", "criteria_text": "[선정 기준 원문]", "benefit_text": "[지원 내용 원문]",
     "apply_text": "[신청 방법 원문]", "apply_channel": str(i % 3), "contact_json": "[문의처 원문]"}
    for i, (c, r, o, iv, dom) in enumerate([("0", "0", "0", "없음", "경제|주거"), ("1", "1", "1", "없음", "미취업(구직)"),
                                           ("2", "2", "3", "있음", "건강"), ("1", "0", "N", "N", "일상·여가"),
                                           ("N", "2", "2", "없음", "교육훈련|관계참여")])
]).set_index("servId")
MOCK_SGG = {"서울": ["강남구", "관악구", "노원구", "마포구", "은평구"],
            "부산": ["부산진구", "해운대구", "사하구"],
            "경기": ["수원시", "성남시", "고양시", "용인시"],
            "광주": ["북구"], "전남": ["목포시"]}


class MockPipeline:
    """시연용. 세션·질문은 실제 코드(confirm.session, confirm.questions)를 쓰고, 추출·검색만 가짜 값."""

    def __init__(self, scenario="일반"):
        self.scenario = scenario
        self.cfg = yaml.safe_load(open(AI_DIR / "configs" / "config.yaml", encoding="utf-8"))
        self.domains = yaml.safe_load(open(AI_DIR / "configs" / "domains.yaml", encoding="utf-8"))["domains"]
        self.terms = yaml.safe_load(open(AI_DIR / "configs" / "query_terms.yaml", encoding="utf-8"))
        self.services = MOCK_SERVICES
        self.sgg_map = {k: frozenset(v) for k, v in MOCK_SGG.items()}

    def _crisis(self, text):
        if self.scenario == "위기":
            return {"crisis": True, "by": "keyword", "evidence": "[위기 단서 예시]"}
        return {"crisis": False}

    def _extract(self, text, cfg_extract=None):
        rec = empty_record()
        if self.scenario != "욕구 없음":
            rec.update(
                needs=[{"domain": "미취업(구직)", "level": "main", "evidence": "일은 하고 싶은데", "status": "명시"},
                       {"domain": "관계참여", "level": "sub", "evidence": "사람 만나는 게 무서워요", "status": "확인 필요"}],
                activity_range="집 안", activity_evidence="[외출 근거 예시]",
                constraints=[{"channel": "방문", "level": "부담", "evidence": "[제약 근거 예시]"}],
                next_action="추천")
        return rec

    def prepare(self, text, profile=None):
        crisis = self._crisis(text)
        if crisis["crisis"]:
            return new_session(text, crisis, None, profile)
        s = new_session(text, crisis, self._extract(text), profile)
        if self.scenario == "욕구 없음":
            s["status"] = "explore_only"   # 시연용: 바로 탐색 안내를 보여 줌
        return s

    def resume(self, session, action):
        with use_sgg_map(self.sgg_map):
            return Q.step(session, action, cfg=self.cfg, domains=self.domains,
                          crisis_fn=self._crisis, extract_fn=self._extract)

    def search(self, session, *, explain_mode="model"):
        if session.get("status") != "ready":
            out = {"status": session.get("status"), "searched": False, "crisis": session.get("crisis"),
                   "session": session, "cards": [], "retrieval_status": "not_searched", "retrieval_errors": []}
            if out["status"] == "crisis_stop":
                out["guidance"] = crisis_guidance()
            return out
        ids = list(self.services.index)
        ranks = {"bm25": ids[::-1], "dense": ids, "fused": ids, "status": "ok", "errors": [],
                 "rrf": [{"servId": sid, "rrf_rank": i + 1, "rrf_score": round(0.03 - i * 0.002, 4),
                          "ranks": {"bm25": len(ids) - i, "dense": i + 1}} for i, sid in enumerate(ids)]}
        base = {"status": "ready", "searched": True, "crisis": session["crisis"], "session": session,
                "queries": {"bm25": "[BM25 쿼리 예시]", "dense": "[Dense 쿼리 예시]"},
                "rules_view": copy.deepcopy(session["confirmed_profile"]), "ranks": ranks,
                "cards": [], "excluded": [], "retrieval_status": "ok", "retrieval_errors": []}
        if self.scenario == "오류":
            err = [{"module": "dense", "type": "Timeout", "message": "예시 오류"}]
            base["ranks"] = {**ranks, "bm25": [], "dense": [], "fused": [], "rrf": [], "status": "error", "errors": err}
            base.update(retrieval_status="error", retrieval_errors=err)
            return base
        if self.scenario == "결과 없음":
            base["ranks"] = {**ranks, "bm25": [], "dense": [], "fused": [], "rrf": [], "status": "empty"}
            base["retrieval_status"] = "empty"
            return base
        if self.scenario == "조건으로 전부 제외":
            base["excluded"] = [{"servId": sid, "rank": rank,
                                 "checks": {"지역": "제외"},
                                 "check_reasons": {"지역": "시연용 지역 조건 불일치"}}
                                for rank, sid in enumerate(ids)]
            return base
        if self.scenario == "일부 오류":   # 한쪽 검색만 실패, 남은 검색으로 카드는 있음
            err = [{"module": "dense", "type": "Timeout", "message": "예시 오류"}]
            base["ranks"] = {**ranks, "dense": [], "status": "partial", "errors": err}
            base.update(retrieval_status="partial", retrieval_errors=err)
        age = session["confirmed_profile"].get("age")
        for sid in ids:
            base["cards"].append({
                "servId": sid, "servNm": self.services.loc[sid, "servNm"],
                "reason": "[추천 이유 1~2문장 · explain 결과]", "how_to_apply": "[신청 방법]",
                "check_needed": ["[신청 전에 확인할 것 1]", "[신청 전에 확인할 것 2]"],
                "evidence": "[근거 문장]", "fallback": False,
                "checks": {"연령": "확인 필요" if age is None else "통과", "지역": "통과", "외출": "통과"},
                "check_reasons": {},
            })
        if explain_mode == "source_first":
            for c in base["cards"]:
                c.update(reason=self.services.loc[c["servId"], "summary_text"],
                         explanation_pending=True, explanation_kind="source")
        return base

    def explain_service(self, session, result, serv_id):
        if session.get("status") != "ready":
            raise ValueError("검색을 마친 세션에서만 설명을 요청할 수 있습니다")
        card = next(c for c in result["cards"] if c["servId"] == serv_id)
        return {**card, "reason": "[AI 설명 시연] 입력하신 상황과 관련된 서비스예요.",
                "explanation_pending": False, "explanation_kind": "model"}


# ─────────────────────────────────────────────
# 3. 파이프라인 고르기 · 상태 보관
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(mock: bool, config_path=None, master_path=None, version=None, require_review=False):
    """실제 Pipeline은 무거워서 한 번만 만든다. 실패하면 예외가 그대로 올라가고 캐시되지 않는다."""
    from pipeline import Pipeline
    return Pipeline(mock=mock, config_path=config_path, services_path=master_path)


def demo_mode() -> bool:
    """배포용 시연 모드. st.secrets 또는 환경변수 DEMO_MODE가 true/1/yes/on이면 켜짐. secrets 파일이 없어도 안전하다."""
    value = None
    try:
        value = st.secrets["DEMO_MODE"]
    except Exception:                                    # 파일 없음·키 없음
        value = None
    if value is None:
        value = os.environ.get("DEMO_MODE")
    return str(value).strip().lower() in ("true", "1", "yes", "on")


DEMO = demo_mode()
ss = st.session_state
# 시연 모드: 첫 화면부터 '실제 흐름(가짜 모델)'. 아니면 기존 그대로
ss.setdefault("pipe_choice", PIPE_CHOICES[1] if DEMO else os.environ.get("BANGEOL_PIPELINE_MODE", PIPE_CHOICES[0]))
ss.setdefault("pipe_error", None)
ss.setdefault("scenario", "일반")
ss.setdefault("dev", False)

CONV_KEYS = ("session", "out", "log", "outcome", "detail", "last_error", "asked", "chat_count", "slot",
             "snapshot", "no_burden", "guidance", "selected", "detail_from", "last_text",
             "activity_return", "activity_declined", "activity_cards", "activity_scope", "activity_energy", "activity_contact",
             "chat_hist", "care_stage", "offer_cooldown", "pending_offer", "skipped", "talk_n")


def init_state():
    ss.setdefault("page", "home")            # 지금 화면
    ss.setdefault("conv_page", "home")       # '대화로 찾아보기'로 돌아갈 화면
    ss.setdefault("session", None)
    ss.setdefault("out", None)
    ss.setdefault("log", [])                 # 말풍선 기록. 화면을 옮겨도 남는다
    ss.setdefault("outcome", None)           # None | results | empty | all_excluded | error
    ss.setdefault("detail", {"contact": None, "repeat": None, "no_interview": False})
    ss.setdefault("last_error", None)        # 대화 중 파이프라인 예외 (개발자 보기용)
    ss.setdefault("asked", [])               # 채팅에서 이미 물은 항목
    ss.setdefault("chat_count", 0)
    ss.setdefault("slot", None)              # 지금 채팅 질문 (확인 단계)
    ss.setdefault("snapshot", None)          # finish_confirm 직전 세션 (조건 바꾸기용)
    ss.setdefault("no_burden", [])           # '괜찮아요'(없음)라고 답한 채널. 세션에는 없음이 기록되지 않아 표시용으로만 둠
    ss.setdefault("guidance", None)          # 위기 안내 (search가 돌려준 정적 안내)
    ss.setdefault("selected", None)          # 상세로 연 servId
    ss.setdefault("detail_from", "results")
    ss.setdefault("last_text", "")
    ss.setdefault("activity_return", "start")
    ss.setdefault("activity_declined", False)
    ss.setdefault("activity_cards", None)
    ss.setdefault("chat_hist", [])           # Gemini로 보낸 일상 대화 [사용자, 답] 쌍 (최근 3턴만 다시 보냄)
    ss.setdefault("care_stage", None)        # 활동 범위 단계 0~3 (3이 가장 고립). None = 모름 → 3처럼 조심스럽게
    ss.setdefault("offer_cooldown", 0)       # [그냥 얘기할래요] 뒤 같은 신호로 다시 묻기까지 남은 턴
    ss.setdefault("talk_n", 0)               # 고정 공감 문구를 돌아가며 쓰는 번호
    ss.setdefault("pending_offer", None)     # 도움 제안(찾아볼래요/그냥 얘기할래요) 대기 중인 문장과 prepare 결과
    ss.setdefault("skipped", [])             # 이미 보여 줬는데 건너뛰거나 빈칸으로 둔 항목 (백엔드가 다시 물어도 안 보여 줌)
    ss.setdefault("visit", [])               # 담은 서비스의 servId (이 창에서만)
    ss.setdefault("cv", 0)                   # 정리표 위젯 key 버전


def new_story():
    """새 대화. 담은 서비스 목록은 남긴다(참고 화면과 같음)."""
    ss.story_revision = ss.get("story_revision", 0) + 1
    for k in CONV_KEYS:
        ss.pop(k, None)
    ss.page = ss.conv_page = "home"
    init_state()


def reset_all():
    new_story()
    ss.visit = []


def get_pipeline():
    if ss.pipe_choice == "시연용 가짜":
        return MockPipeline(ss.scenario)
    try:
        with st.spinner("준비하고 있어요. 처음 한 번은 시간이 조금 걸려요..."):
            config_path, master_path, version = pipeline_inputs(os.environ.get("BANGEOL_CONFIG"), os.environ.get("BANGEOL_MASTER"))
            if ss.get("pipeline_version") is not None and ss.pipeline_version != version:
                new_story()
                st.info("서비스 자료나 AI 설정이 갱신되어 새 대화로 시작해요.")
            ss.pipeline_version = version
            return load_pipeline(ss.pipe_choice == "실제 흐름(가짜 모델)", config_path, master_path, version,
                                 os.environ.get("BANGEOL_REQUIRE_REVIEW_DATA") == "1")
    except Exception as e:
        ss.pipe_error = f"{type(e).__name__}: {e}"
        st.error("선택한 AI 실행 환경을 준비하지 못했어요. GPU 연결과 설치 상태를 확인해 주세요.")
        with st.expander("실행 오류 확인"):
            st.code(ss.pipe_error)
        if st.button("같은 모드로 다시 연결", key="retry_pipeline"):
            st.rerun()
        st.stop()



init_state()


def go(page):
    if page != ss.page:
        ss.screen_revision = ss.get("screen_revision", 0) + 1
    ss.page = page
    if page in CONV_PAGES:
        ss.conv_page = page


# ─────────────────────────────────────────────
# 4. 화면 쪽 계산: 말투 수위(숨김) · 카드 정보 · 세부 필터 · 표시용 이스케이프
# ─────────────────────────────────────────────
def profile():
    """사용자 수정까지 반영된 현재 값 (confirmed_profile)."""
    return (ss.session or {}).get("confirmed_profile") or {}


def status():
    return (ss.session or {}).get("status")


def tone_level():
    """말투 수위 1~3. activity_range만 쓴다. 판단 불가·미입력은 1.
    문구 고르는 데만 쓴다(순위·필터·자격 판단에 쓰지 않음, 화면에 표시하지 않음)."""
    return {"방 안": 1, "집 안": 1, "집 근처": 2, "필요하면 외출": 3}.get(profile().get("activity_range"), 1)


def copy_for(key):
    return TONE_COPY[tone_level()][key]


def labels():
    """대화·정리표·추천과 둘러보기에서 같은 분야 이름을 사용한다."""
    return dict(EXPLORE_DOMAINS)


def region_label(area):
    return REGION_LABELS.get(area, area)


_MD = re.compile(r"([\\`*_\[\]{}()#+\-.!|<>~$:])")


def esc(v):
    """사용자 입력·CSV 원문·백엔드 문구를 마크다운으로 해석하지 않고 그대로 보이게 한다(HTML도 쓰지 않음)."""
    return _MD.sub(r"\\\1", str(v)).replace("\n", "  \n")


def _clean(v):
    """'N'·빈칸·NaN은 정보 없음(None)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return None if s in ("", "N", "NA", "nan") else s


def short(v, n):
    v = _clean(v) or ""      # NaN·None·빈칸은 빈 글자로
    return v[:n] + "…" if len(v) > n else v


def clean_list(v):
    """목록 칸(check_needed 등). 문자열 하나·NaN·None도 받고, 빈 항목은 뺀다."""
    if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
        v = [v]
    return [x for x in (_clean(i) for i in v) if x]


def http_url(v):
    v = _clean(v)
    return v if v and re.match(r"https?://", v) else None


def contact_text(value):
    """Display the master's structured contacts without exposing JSON syntax."""
    try:
        rows = json.loads(value)
    except (TypeError, ValueError):
        return value
    if not isinstance(rows, list):
        return value
    lines = []
    for row in rows:
        if isinstance(row, dict):
            parts = (_clean(row.get(k)) for k in ("name", "content"))
            line = " · ".join(dict.fromkeys(part for part in parts if part))
            if line:
                lines.append(line)
    return "\n".join(dict.fromkeys(lines))


def service_detail_markdown(value):
    text = _clean(value)
    if not text:
        return ":gray[정보 확인 필요]"
    paragraphs = []
    for line in re.split(r"\n+|\s+(?=[○●□■]\s*)", text):
        line = line.strip()
        if not line:
            continue
        heading = re.match(r"^([○●□■•·-]\s*)?([^:：]{1,30})\s*[:：]\s*(.*)$", line)
        if heading:
            marker, label, body = heading.groups()
            paragraphs.append(f"{esc(marker or '')}**{esc(label.strip())}:** {esc(body)}")
        else:
            paragraphs.append(esc(line))
    return "\n\n".join(paragraphs)


def service_row(sid):
    """현재 마스터에서 servId 한 줄. 없으면 None. 정보 없음 칸은 None."""
    if sid not in pipe.services.index:
        return None
    row = pipe.services.loc[sid]
    return {c: _clean(row.get(c)) for c in DETAIL_COLS}


def with_service_info(cards):
    """카드의 servId로 services에서 요약·링크·부담 값을 꺼내 붙인다. 정보 없음은 None."""
    out = []
    for c in cards:
        row = pipe.services.loc[c["servId"]] if c["servId"] in pipe.services.index else {}
        info = {}
        for col in INFO_COLS:
            v = _clean(row.get(col) if hasattr(row, "get") else None)
            if col in BURDEN_COLS and v is not None:
                v = int(v) if v.isdigit() else None
            info[col] = v
        out.append({**c, **info})
    return out


def apply_detail_filters(cards, f):
    """세부 필터. 받은 카드(Top-N) 안에서만 거른다. 정보 없음(None)인 서비스는 빼지 않는다."""
    # TODO(통합): 백엔드에서 Top-5 전 후보 필요 (지금은 final_top_n으로 자른 뒤라 걸러지면 빈자리를 못 채움)
    def ok(card):
        for key, limit in (("contact_burden", f["contact"]), ("repeat_burden", f["repeat"])):
            v = card.get(key)
            if limit is not None and v is not None and v > limit:
                return False
        return not (f["no_interview"] and card.get("apply_interview") == "있음")
    return [c for c in cards if ok(c)]


def constraint_level(ch):
    """확정 제약 수준. 없으면 '없음'이라고 답했는지 보고, 그것도 아니면 None(모름)."""
    lv = next((c["level"] for c in profile().get("constraints", []) if c["channel"] == ch), None)
    return lv or ("없음" if ch in ss.no_burden else None)


def known(slot):
    p = profile()
    if slot == "activity_range":
        return p.get("activity_range") not in (None, "판단 불가")
    if slot == "phone":
        return constraint_level("전화") is not None
    return p.get(slot) is not None


# ─────────────────────────────────────────────
# 5. 대화 기록 · 백엔드 호출
# ─────────────────────────────────────────────
def say(role, text):
    """assistant 문구는 화면 고정 문구(+이스케이프한 값). user 문구는 표시할 때 이스케이프한다."""
    ss.log.append({"role": role, "text": text})


def bot():
    return st.chat_message("assistant", avatar="🌗")


def render_log():
    for i, m in enumerate(ss.log):
        if m["role"] == "assistant":
            st.html(f'<span id="chat-reply-{i}" aria-hidden="true"></span>')
        with (bot() if m["role"] == "assistant" else st.chat_message("user")):
            st.markdown(m["text"] if m["role"] == "assistant" else esc(m["text"]))
        if m.get("note"):       # 말풍선 아래 작은 글씨 (외부 AI가 만든 답 표시)
            st.caption(m["note"])


def render_scroll(page):
    """Only fixed UI state enters the script; conversation content stays out."""
    anchor = None
    if page == "confirm":
        anchor = "confirm-summary"
    elif page == "chat":
        if status() == "awaiting_answer" and current_question() is not None:
            anchor = "chat-followup"
        else:
            reply = next((i for i in range(len(ss.log) - 1, -1, -1)
                          if ss.log[i]["role"] == "assistant"), None)
            if reply is not None:
                anchor = f"chat-reply-{reply}"
    state = {"page": page, "anchor": anchor,
             "token": f"{page}:{ss.get('story_revision', 0)}:{ss.get('screen_revision', 0)}:{len(ss.log) if page == 'chat' else 0}"}
    script = (AI_DIR / "ui_scroll.js").read_text(encoding="utf-8")
    st.html("<script>" + script.replace("__SCROLL_STATE__", json.dumps(state)) + "</script>", unsafe_allow_javascript=True)


def safe(fn, *args, **kwargs):
    """파이프라인 호출. 예외가 나도 화면이 죽지 않게 오류 안내로 바꾼다."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        ss.last_error = f"{getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}"
        ss.outcome = "error"
        say("assistant", MSG["error"])
        return None


def resume(action):
    s = safe(pipe.resume, ss.session, {"target": None, "value": None, **action})
    if s is not None:
        ss.session = s
    return s is not None


def to_safety():
    """위기 중단. 추천 결과를 비우고 안전 화면으로. 안내는 search가 돌려준 정적 안내를 그대로 쓴다."""
    ss.out, ss.outcome, ss.slot = None, None, None
    ss.activity_cards = None
    out = safe(pipe.search, ss.session)
    ss.guidance = (out or {}).get("guidance")
    say("assistant", MSG["safety"])
    go("safety")


def after_backend():
    """resume 뒤 상태에 따라 다음 화면으로. ready면 바로 검색한다."""
    st_ = status()
    if st_ == "crisis_stop":
        to_safety()
    elif st_ == "explore_only":
        ss.slot = None
        say("assistant", MSG["explore"])
        go("unsure")
    elif st_ == "awaiting_answer":
        # 질문 말풍선은 채팅 화면이 선택지 바로 위에 그리고, 답할 때 기록에 남긴다.
        # 질문을 못 받았으면 빈 말풍선 대신 '질문을 불러오지 못했어요' 안내가 나온다.
        auto_skip_seen()
        if status() == "awaiting_answer":
            go("chat")
        else:
            after_backend()
    elif st_ == "ready":
        run_search()


def auto_skip_seen():
    """이미 보여 준 빈 칸(채팅에서 건너뜀·정리표 빈칸)을 백엔드가 또 물으면 화면에 띄우지 않고 건너뜀으로 답한다.
    필요한 도움(need)은 건너뛰지 않는다. 백엔드 질문 한도(max_total)는 그대로 지켜진다."""
    for _ in range(4):
        if status() != "awaiting_answer":
            return
        q = Q.pending(ss.session)
        if not q or q["slot"] == "need" or q["slot"] not in ss.skipped:
            return
        if not resume({"type": "answer", "target": q["slot"], "value": None, "answer_type": "건너뜀"}):
            return


def current_question():
    """백엔드가 기다리는 질문의 화면 표시값. 우리 백엔드에는 session['question'] 키가 없다:
    질문은 session['questions'][-1](답 전)에 있고, 문구·선택지는 Q.render()가 prompt·options·number_input·
    allow_free·skip_label로 만든다. 질문이 없거나 문구가 비었으면 None."""
    try:
        q = Q.pending(ss.session) if ss.session else None
        r = Q.render(q, labels(), pipe.domains) if q else None
    except Exception as e:
        ss.last_error = f"render: {type(e).__name__}: {e}"
        return None
    return r if r and str(r.get("prompt") or "").strip() else None


def care_stage_now():
    """지금 말투 단계 0~3. 모르면 3(가장 조심스럽게). tone_level(결과 화면 문구)과는 별개다."""
    return ss.care_stage if ss.get("care_stage") is not None else chitchat.UNKNOWN_STAGE


def update_care_stage(session):
    """추출·확인된 활동 범위가 있으면 대화에서 본 가장 고립된 단계를 기억한다."""
    ar = ((session or {}).get("confirmed_profile") or {}).get("activity_range")
    if ar in chitchat.ACTIVITY_STAGE:
        ss.care_stage = max(ss.care_stage if ss.get("care_stage") is not None else 0, chitchat.care_stage(ar))


def mark_chat(message):
    """일상 대화 말풍선 표시(새 이야기에서 앞 말풍선을 남기는 기준)."""
    message["chat"] = True
    return message


def smalltalk_reply(text):
    """신호 없는 일상 문장: Gemini(또는 단계별 기본 답). 보내는 건 이 한 문장 + 앞선 일상 대화 최근 3턴뿐."""
    with st.spinner("답을 쓰고 있어요."):
        r = chitchat.reply(text, secrets=st.secrets, stage=care_stage_now(), history=ss.chat_hist)
    ss.chitchat_status = r.source + (f":{r.error}" if r.error else "")
    if r.source == "gemini":
        ss.chat_hist = (ss.chat_hist + [[text, r.text]])[-chitchat.MAX_HISTORY:]
    mark_chat(ss.log[-1])                               # 방금 사용자 말풍선
    ss.log.append(mark_chat({"role": "assistant", "text": esc(r.text),
                             "note": CHITCHAT_AI_NOTE if r.source == "gemini" else None}))


def say_talk():
    """고정 공감 답 한 말풍선(단계별 문구를 돌아가며). 질문이 아니므로 '대답하지 않아도 괜찮아요'는 붙이지 않는다.
    이 말풍선 바로 앞의 사용자 말풍선도 일상 대화 말풍선으로 표시한다."""
    text = chitchat.talk_reply_text(care_stage_now(), ss.get("talk_n", 0))
    ss.talk_n = ss.get("talk_n", 0) + 1
    mark_chat(ss.log[-1])
    ss.log.append(mark_chat({"role": "assistant", "text": esc(text)}))


def offer_help(text, s):
    """힘듦·필요한 도움이 보이면 먼저 묻는다: 찾아볼까요? [찾아볼래요] [그냥 얘기할래요]. 고정 문구, Gemini 안 부름.
    질문이라서 말풍선 아래에 단계별 안내(3단계: 답하지 않아도 괜찮아요)를 붙인다."""
    ss.pending_offer = {"text": text, "session": s}
    mark_chat(ss.log[-1])
    ss.log.append(mark_chat({"role": "assistant", "text": chitchat.OFFER_QUESTION,
                             "note": chitchat.OFFER_NOTE[care_stage_now()]}))


def continue_flow(text, s):
    """기존 정보 수집 흐름. text가 첫 이야기다."""
    ss.session = s
    if s["status"] != "awaiting_confirm":
        after_backend()
        return
    if offer_small_activities(text, "start"):
        return
    names = [labels().get(n["domain"], n["domain"]) for n in profile().get("needs", [])]
    if names:
        say("assistant", f"말씀해 주신 내용에서 **{esc(', '.join(names))}** 쪽 도움을 살펴보면 좋겠다고 이해했어요. "
                         "마지막에 자유롭게 바꾸실 수 있어요.")
    ask_next()


def offer_accept(label=None):
    """[찾아볼래요] 또는 글자 답('네' 등): 제안이 나왔던 그 문장을 첫 이야기로 기존 흐름을 시작한다(다시 말하지 않아도 됨).
    label이 있으면 사용자 말풍선을 이미 그렸다(글자 답)."""
    offer = ss.pending_offer
    if not offer:
        return
    ss.pending_offer = None
    ss.chat_hist = []                                   # 정보 수집으로 넘어가면 일상 대화 기록을 비운다
    if label is None:
        say("user", chitchat.OFFER_BUTTONS[0])
    ss.last_text = offer["text"]
    continue_flow(offer["text"], offer["session"])


def offer_talk(label=None):
    """[그냥 얘기할래요] 또는 글자 답('아니요' 등): 단계별 고정 문구로만 답하고, 같은 막연한 힘듦으로 다시 묻기까지 3턴 쉰다."""
    ss.pending_offer = None
    ss.offer_cooldown = chitchat.OFFER_COOLDOWN
    if label is None:
        say("user", chitchat.OFFER_BUTTONS[1])
    say_talk()


def start(text, direct=False):
    """새 글 입력 → prepare(위기 확인이 맨 먼저) → 갈라진다.
    - 위기: 안전 안내
    - 제안 질문이 떠 있을 때 글자 답('네'/'아니요' 등): 버튼과 같게 처리 (위기 확인을 거친 뒤)
    - 분명한 요청('서비스 찾아', '추천해 줘', '신청' 등)·예시 버튼·탐색 안내 뒤 덧붙인 말: 제안 없이 바로 기존 흐름(그 문장이 첫 이야기)
    - 신호 없음: 일상 대화(Gemini, 단계별 말투)
    - 필요한 도움이 새로 보임(월세·일자리 등): 쉬는 중이어도 제안 질문
    - 막연한 힘듦: 제안 질문. 방금 [그냥 얘기할래요]를 골랐다면 3턴은 고정 공감 문구만
    탐색 안내(explore_only) 뒤 덧붙인 말은 앞 글과 합쳐서 다시 본다."""
    text = (text or "").strip()
    if not text:
        return
    pending = ss.get("pending_offer")
    prev = ss.session
    explore_prev = bool(prev and prev.get("status") == "explore_only")
    chat_ctx = prev is None or explore_prev                # 일상 대화·도움 제안을 받을 수 있는 상태
    keep_snapshot = ss.snapshot
    carry = {k: ss.get(k) for k in ("chat_hist", "care_stage", "offer_cooldown", "talk_n")} if chat_ctx else {}
    cooldown = carry.get("offer_cooldown") or 0
    if explore_prev:
        full = prev["original_text"] + " " + text
    else:
        kept = list(ss.log) if (prev is None and any(m.get("chat") for m in ss.log)) else None
        new_story()
        ss.log = kept or [{"role": "assistant", "text": GREETING}]
        for k, v in carry.items():
            ss[k] = v
        full = text
    ss.pending_offer = None
    ss.offer_cooldown = max(0, cooldown - 1)
    say("user", text)
    ss.last_text = text
    ss.out, ss.outcome, ss.last_error, ss.slot, ss.snapshot = None, None, None, None, None
    ss.asked, ss.chat_count, ss.no_burden, ss.skipped = [], 0, [], []
    go("chat")
    with st.spinner("이야기를 살펴보고 있어요. 잠시만 기다려 주세요."):
        s = safe(pipe.prepare, full)
    if s is None:
        ss.session = None
        return
    if s["status"] == "crisis_stop":                       # 위기는 항상 먼저. 글자 답이든 아니든 일상 대화 기록도 비운다
        ss.chat_hist = []
        ss.session = s
        to_safety()
        return
    update_care_stage(s)
    if pending:                                            # 제안 질문에 글자로 답했는지 (위기 확인을 이미 통과함)
        answer = chitchat.offer_answer(text)
        if answer == "yes":
            ss.pending_offer = pending
            offer_accept(label=text)                       # 첫 이야기는 제안이 나왔던 그 문장('네'가 아님)
            return
        if answer == "talk":
            ss.pending_offer = pending
            offer_talk(label=text)
            ss.snapshot = keep_snapshot
            return
    if chat_ctx and s["status"] in ("awaiting_confirm", "explore_only"):
        needs = bool((s.get("confirmed_profile") or {}).get("needs"))
        concrete = needs or chitchat.has_concrete_need(text)       # 구체적인 도움 주제(월세·일자리 등)
        vague = chitchat.has_signal(text) or should_offer(text)    # 막연한 힘듦·도움 표현
        explicit = chitchat.is_explicit_request(text)
        if not (concrete or vague or explicit):
            ss.session = prev
            smalltalk_reply(text)
            ss.snapshot = keep_snapshot
            return
        if explicit or direct or explore_prev:
            ss.offer_cooldown = 0
        else:
            ss.session = prev
            ss.snapshot = keep_snapshot
            if cooldown > 0 and not concrete:              # 쉬는 중에는 같은 막연한 힘듦에만 다시 묻지 않음
                say_talk()
            else:
                ss.offer_cooldown = 0                      # 새로 보인 도움 주제는 쉬는 중에도 다시 제안
                offer_help(text, s)
            return
    ss.chat_hist = []
    continue_flow(text, s)


def ask_next(ack=""):
    """채팅 질문을 하나 고른다. 이미 아는 항목·물은 항목은 건너뛰고, MAX_CHAT_Q개 뒤에는 정리표로."""
    if not profile().get("needs") and "need" not in ss.asked:
        slot = "need"
    else:
        slot = next((k for k in CHAT_ORDER if k not in ss.asked and not known(k)), None)
    if ss.chat_count >= MAX_CHAT_Q or slot is None:
        show_summary(ack)
        return
    ss.slot = slot
    ss.asked = ss.asked + [slot]
    ss.chat_count += 1
    lead = ""
    if slot == "activity_range" and re.search(r"밖|사람|만나", ss.session.get("original_text", "")):
        lead = "밖에 나가거나 사람을 만나는 부담도 함께 살펴볼게요. "
    elif slot == "phone":
        lead = "방문과 전화의 부담은 다를 수 있어서요. "
    say("assistant", (ack + "  \n" if ack else "") + lead + CHAT_SLOTS[slot])
    go("chat")


def show_summary(ack=""):
    ss.slot = None
    say("assistant", (ack + "  \n" if ack else "") + MSG["summary"])
    go_confirm()


def parse_reply(slot, raw):
    """자유 답을 조건 값으로. 정해진 짧은 표현만 받는다(규칙 기반, 모델 호출 아님). 못 읽으면 None."""
    t = raw.strip()
    if slot == "age":
        m = re.fullmatch(r"(?:만\s*)?(\d{1,2})(?:\s*(?:살|세)(?:이에요|예요|입니다)?)?[.!]?", t)
        return int(m.group(1)) if m and 10 <= int(m.group(1)) <= 99 else None
    if slot == "need":
        hit = next((d for d, pat in NEED_WORDS if re.fullmatch(pat, t)), None)
        if hit:
            return hit
        lab = labels()
        hits = [d for d in pipe.domains if t in (d, lab.get(d))]
        return hits[0] if len(hits) == 1 else None
    if slot == "region_sido":
        for merged in MERGED_SIDO:                          # 통합 이름 안의 '전남'·'광주'를 따로 세지 않음
            t = t.replace(merged, "")
        hits = {x for x in SIDO if x in t}
        return hits.pop() if len(hits) == 1 else None
    if slot == "activity_range":
        for pat, v in ((r"필요.*(외출|나가)", "필요하면 외출"), (r"집\s*근처|동네|편의점", "집 근처"),
                       (r"방\s*안|방에서", "방 안"), (r"집\s*안|집에만|밖.*못|안\s*나가", "집 안")):
            if re.search(pat, t):
                return v
        return None
    if slot == "phone":
        for pat, v in ((r"부담\s*(이|은|은\s*)?\s*없", "없음"), (r"불가|못|어려|안\s*돼", "불가"),
                       (r"부담", "부담"), (r"괜찮|가능|없음|좋아", "없음")):
            if re.search(pat, t):
                return v
    return None


CHAT_ACTION = {  # 채팅 질문 → 확인 조작
    "need": lambda v: {"type": "add_need", "value": v},
    "region_sido": lambda v: {"type": "set_region", "value": v},
    "activity_range": lambda v: {"type": "set_activity", "value": v},
    "age": lambda v: {"type": "set_age", "value": v},
    "phone": lambda v: {"type": "set_constraint", "target": "전화", "value": v},
}


def answer_chat(raw, value=None, skip=False):
    """채팅 질문(확인 단계)의 답. 버튼은 value를 함께 받고, 자유 입력은 위기 확인(memo) 뒤 규칙으로 읽는다."""
    slot = ss.slot
    if skip:
        say("user", SKIP_QUESTION)
        if slot:
            ss.skipped = list(dict.fromkeys(ss.skipped + [slot]))   # 찾기 뒤 백엔드가 같은 질문을 다시 보여 주지 않게 기억
        ask_next("답하지 않으셔도 괜찮아요.")
        return
    say("user", raw)
    if value is None:
        if not resume({"type": "memo", "value": raw}):   # 자유 입력은 먼저 위기 확인 (검색에는 안 씀)
            return
        if status() == "crisis_stop":
            to_safety()
            return
        if offer_small_activities(raw, "chat", slot):
            return
        value = parse_reply(slot, raw)
        if value is None:
            say("assistant", MSG["not_parsed"])
            return
    if not resume(CHAT_ACTION[slot](value)):
        return
    if slot == "phone" and value == "없음" and "전화" not in ss.no_burden:
        ss.no_burden = ss.no_burden + ["전화"]
    ack = {"need": f"**{esc(labels().get(value, value))}** 쪽부터 함께 살펴볼게요.",
           "region_sido": f"{region_label(value)} 기준으로 살펴볼게요.",
           "age": f"{value}세로 확인했어요."}.get(slot, "알려주셔서 고마워요.")
    ask_next(ack)


def answer_backend(r, label, value, answer_type):
    """백엔드 추가 질문(awaiting_answer)의 답. 질문 문구를 먼저 기록에 남긴다."""
    say("assistant", esc(r["prompt"]))
    say("user", label)
    if resume({"type": "answer", "target": r["slot"], "value": value, "answer_type": answer_type}):
        after_backend()


def answer_backend_free(r, raw):
    """백엔드 질문에 자유 입력. 직접 쓰기가 되는 질문(필요한 도움)은 백엔드가 위기 확인·후보 재확인을 한다.
    그 밖의 질문은 규칙으로 읽고, 못 읽으면 memo로 위기 확인만 하고 같은 질문을 유지한다."""
    if should_offer(raw, r["slot"]) and not ss.activity_declined:
        say("user", raw)
        if resume({"type": "memo", "value": raw}):
            if status() == "crisis_stop":
                to_safety()
            else:
                offer_small_activities(raw, "chat", r["slot"])
        return
    if r.get("allow_free"):
        answer_backend(r, raw, raw, "자유")
        return
    value = parse_reply(r["slot"], raw)
    if value is not None:
        answer_backend(r, raw, value, "선택")
        return
    say("assistant", esc(r["prompt"]))
    say("user", raw)
    if resume({"type": "memo", "value": raw}):
        if status() == "crisis_stop":
            to_safety()
        else:
            say("assistant", "정확히 이해하지 못했어요. 아래 답변 중 가까운 것을 골라 주셔도 괜찮아요.")


def submit_confirm(needs, age, sido, sgg, act, levels):
    """정리표 확인: 바뀐 값만 확인 조작으로 보내고 finish_confirm. 그 뒤는 백엔드가 정한다.
    필요한 도움이 하나도 없으면 바뀐 값만 반영하고 finish_confirm 없이 정보 부족 탐색으로 간다."""
    p = profile()
    cur = [n["domain"] for n in p.get("needs", [])]
    actions = [{"type": "add_need", "value": d} for d in needs if d not in cur]
    actions += [{"type": "delete_need", "target": d} for d in cur if d not in needs]
    if age is not None and int(age) != p.get("age"):
        actions.append({"type": "set_age", "value": int(age)})
    if sido and sido != p.get("region_sido"):
        actions.append({"type": "set_region", "value": sido})
    if sido and sgg != p.get("region_sgg"):
        actions.append({"type": "set_region_sgg", "value": sgg})
    if act and act != p.get("activity_range"):
        actions.append({"type": "set_activity", "value": act})
    for ch, v in levels.items():
        if v and v != constraint_level(ch):
            actions.append({"type": "set_constraint", "target": ch, "value": v})
        if v == "없음" and ch not in ss.no_burden:
            ss.no_burden = ss.no_burden + [ch]
    for a in actions:
        if not resume(a):
            return
    # 정리표에서 빈칸으로 둔 항목도 이미 본 것으로 기억 (찾기 뒤 다시 묻지 않음)
    ss.skipped = list(dict.fromkeys(ss.skipped + [k for k in ("region_sido", "activity_range", "age") if not known(k)]))
    if not needs:
        ss.snapshot = copy.deepcopy(ss.session)
        go("unsure")
        return
    say("user", "이 조건으로 찾아 주세요.")
    ss.snapshot = copy.deepcopy(ss.session)
    if resume({"type": "finish_confirm"}):
        after_backend()


def reopen_confirm():
    """Open another confirmation round without losing answers from the current round."""
    if not ss.session or status() == "crisis_stop":
        return False
    if status() == "awaiting_confirm":
        return True
    # The pre-search snapshot predates backend follow-up answers. Reopen the
    # current session instead so its accepted actions and conditions survive.
    if status() not in ("awaiting_answer", "explore_only", "ready"):
        return False
    s = copy.deepcopy(ss.session)
    s.update(status="awaiting_confirm", confirmation="none", questions=[],
             queries={}, query_expansion={})
    ss.session = s
    ss.out, ss.outcome, ss.snapshot = None, None, None
    return True


def go_confirm():
    """Synchronize the editable session before opening its summary screen."""
    if status() == "crisis_stop":
        go("safety")
        return
    if not reopen_confirm():
        go("chat" if ss.session else "home")
        return
    ss.slot = None
    ss.cv += 1
    go("confirm")


def run_search():
    with st.spinner("조건에 맞는 서비스를 찾고 있어요. 목록부터 보여 드릴게요..."):
        out = safe(pipe.search, ss.session, explain_mode="source_first")
    go("results")
    if out is None:
        return
    out["cards"] = with_service_info(out["cards"])
    ss.out = out
    rs = out["retrieval_status"]
    if rs == "error":
        ss.outcome = "error"
        say("assistant", MSG["error"])
    elif rs == "empty":
        ss.outcome = "empty"
        say("assistant", copy_for("empty"))
    elif not out["cards"]:
        ss.outcome = "all_excluded"
        say("assistant", copy_for("all_excluded"))
    else:                                    # ok·partial (partial은 결과 화면에 안내를 덧붙임)
        ss.outcome = "results"
        say("assistant", copy_for("to_result"))


# ─────────────────────────────────────────────
# 6. 공통 화면 조각
# ─────────────────────────────────────────────
def head(eyebrow, title, desc="", width="auto"):
    """고정 문구 전용 (사용자·원문 값을 넣지 않는다)."""
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'<div class="page-head">{eyebrow_html}<h1>{title}</h1><p>{desc}</p></div>',
                unsafe_allow_html=True, width=width)


def option_grid(options, key, n=3, primary=()):
    """[(값, 문구)] 버튼 묶음. 누른 (값, 문구)를 돌려준다."""
    cols = st.columns(n)
    hit = None
    for i, (v, lab) in enumerate(options):
        if cols[i % n].button(lab, key=f"{key}_{i}", width="stretch",
                              type="primary" if v in primary else "secondary"):
            hit = (v, lab)
    return hit


def result_tags():
    names = [labels().get(n["domain"], n["domain"]) for n in profile().get("needs", [])]
    region = profile().get("region_sido")
    if region:
        names.append(f"{region_label(region)}(지역)")
    if names:
        with st.container(key="result-tags", width="content"):
            st.markdown(" ".join(f":green-badge[{esc(x)}]" for x in names), width="content")


def contact_block(c):
    """연락처 한 건. 백엔드 guidance에 있는 항목만 표시하고 없는 항목은 비워 둔다. 전화는 사용자가 직접 거는 tel: 링크."""
    name, number, hours, desc = (c.get(k) for k in ("name", "number", "hours", "description"))
    line = " ".join(x for x in (f"**{esc(number)}**" if number else "", esc(name) if name else "",
                                f"({esc(hours)})" if hours else "") if x)
    st.markdown(line + (f"  \n{esc(desc)}" if desc else ""))
    digits = re.sub(r"\D", "", str(number or ""))
    if digits:
        st.link_button(f"{number} · 전화 연결 ↗", "tel:" + digits, width="stretch")


def render_guidance(g):
    """Reference layout; static contact links, no automatic call or transmission."""
    if status() == "crisis_stop" and g.get("message"):
        with bot():
            st.markdown(esc(g["message"]))
    st.markdown("""<div class="support-grid">
<section class="support-card urgent"><span class="tag">즉시 위험 · 긴급 도움</span><h2>신고·구조가 필요하다면</h2><p>지금 신변에 위협이 있거나 구조·구급이 필요할 때 연락할 수 있어요.</p><a class="support-link primary" href="tel:112">112 · 경찰 긴급신고 전화 연결 ↗</a><a class="support-link primary" href="tel:119">119 · 구조·구급 요청 전화 연결 ↗</a></section>
<section class="support-card"><span class="tag">전문 상담</span><h2>누군가와 이야기하고 싶을 때</h2><p>혼자 감당하기 힘들거나 이야기를 들어줄 사람이 필요할 때 전화할 수 있어요.</p><a class="support-link primary" href="tel:+82109?oai_link_source=model_response_hotline">109 · 상담전화 연결 ↗</a><a class="support-link" target="_blank" rel="noopener noreferrer" href="https://www.129.go.kr/109?oai_link_source=model_response_hotline">공식 상담 안내 보기 ↗</a></section></div>
<div class="connection-notice">버튼은 기기의 전화 앱 또는 공식 안내 페이지를 열어요. 자동 통화·신고·출동 요청·위치나 대화 전송은 하지 않아요.<br>PC에서 전화 앱이 없으면 버튼이 동작하지 않을 수 있어요. 표시된 번호로 직접 연락할 수 있어요.</div>""", unsafe_allow_html=True)



# ── 서비스 둘러보기 · 정보 부족 탐색 (화면이 가진 서비스 표 pipe.services만 사용, search 호출 없음) ──
def domain_tokens(v):
    """app_domain을 | 로 나눈 토큰 집합. 정보 없음(N·빈칸)은 빈 집합."""
    v = _clean(v)
    return {t.strip() for t in v.split("|")} if v else set()


def service_domains():
    """Use approved Q1 domains for browsing when the final addendum is connected."""
    df = pipe.services
    if "q1_core_domains" in df:
        return [(sid, set(v)) for sid, v in zip(df.index, df["q1_core_domains"])]
    col = df["app_domain"] if "app_domain" in df else [None] * len(df)
    return [(sid, domain_tokens(v)) for sid, v in zip(df.index, col)]


def explore_find(domain, query):
    """분야(None=전체)와 검색어(서비스명+요약에 모든 낱말이 글자로 들어 있는 것)에 맞는 servId. 순위 없이 표 순서."""
    df = pipe.services
    terms = [t.lower() for t in (query or "").split()]
    hit = []
    for (sid, toks), nm, sm in zip(service_domains(), df["servNm"], df["summary_text"]):
        if domain and domain not in toks:
            continue
        text = f"{_clean(nm) or ''} {_clean(sm) or ''}".lower()
        if all(t in text for t in terms):
            hit.append(sid)
    return hit


def domain_options():
    """[(분야, '라벨 · N개')] 맨 앞은 전체 보기(None)."""
    doms = service_domains()
    n = {d: sum(1 for _, t in doms if d in t) for d, _ in EXPLORE_DOMAINS}
    return [(None, f"전체 보기 · {len(doms)}개")] + [(d, f"{lab} · {n[d]}개") for d, lab in EXPLORE_DOMAINS]


def explore_label(domain):
    return dict(EXPLORE_DOMAINS).get(domain, "전체 보기")


def clearly_mismatch(sid):
    """알려진 나이·지역과 분명히 안 맞을 때만 True (모르면 False). 자격 판정이 아니라 탐색용 거름."""
    row, p = pipe.services.loc[sid], profile()
    age = p.get("age")
    if isinstance(age, int):
        (lk, lo), (hk, hi) = parse_bound(row.get("age_min")), parse_bound(row.get("age_max"))
        if (lk == "int" and age < lo) or (hk == "int" and age > hi):
            return True
    sido = norm_sido(p.get("region_sido")) - {NATIONWIDE}
    svc = norm_sido(_clean(row.get("region_sido")))
    return bool(_clean(row.get("source")) == "local" and sido and svc and NATIONWIDE not in svc and not sido & svc)


def unsure_picks():
    """분야마다 맞지 않는 게 아닌 첫 서비스 1개, 최대 UNSURE_MAX개."""
    doms, picks = service_domains(), []
    for d, _ in EXPLORE_DOMAINS:
        sid = next((i for i, t in doms if d in t and i not in picks and not clearly_mismatch(i)), None)
        if sid:
            picks.append(sid)
    return picks[:UNSURE_MAX]


def pick_domain(d):
    """정보 부족 탐색에서 고른 분야를 필요한 도움으로 넣고 정리표로."""
    if status() == "crisis_stop":
        go("safety")
        return
    reopen_confirm()
    if status() == "awaiting_confirm":
        if d not in [n["domain"] for n in profile().get("needs", [])] and not resume({"type": "add_need", "value": d}):
            return
        say("assistant", f"**{esc(explore_label(d))}** 쪽을 필요한 도움으로 넣었어요. 정리표에서 바꿀 수 있어요.")
        go_confirm()
    else:
        st.warning("지금은 필요한 도움을 넣을 수 없는 단계예요. 새로운 이야기를 시작해 주세요.")


def service_cards(sids, tag):
    for i, sid in enumerate(sids):
        s = service_row(sid)
        if s:
            show_card({"servId": sid, **s}, i, live=False, tag=tag)


def page_explore():
    head("", "어떤 서비스가 있는지 둘러봐요.",
         "이야기 없이도 볼 수 있어요.")
    ss.setdefault("ex_domain", None)
    ss.setdefault("ex_q", "")
    hit = None
    with st.container(key="browse-controls"):
        for i, (domain, label) in enumerate(domain_options()):
            if st.button(label, key=f"ex_dom_{i}", width="stretch",
                         type="primary" if domain == ss.ex_domain else "secondary"):
                hit = (domain, label)
        ss.ex_q = st.text_input("서비스 검색", value=ss.ex_q, max_chars=60,
                                placeholder="서비스 이름·요약 검색", key="ex_q_in", label_visibility="collapsed")
    if hit:
        ss.ex_domain = hit[0]
        st.rerun()
    found = explore_find(ss.ex_domain, ss.ex_q)
    st.caption(f"{esc(explore_label(ss.ex_domain))} · {len(found)}개 · 최대 {EXPLORE_MAX}개 표시 · 추천 순위 아님")
    if not found:
        st.info("검색 결과가 없어요. 다른 분야를 고르거나 다른 낱말로 찾아보세요.")
    service_cards(found[:EXPLORE_MAX], "둘러보기")


def offer_small_activities(text, return_page, slot=None):
    if status() == "crisis_stop" or ss.activity_declined or not should_offer(text, slot):
        return False
    ss.activity_return = return_page
    ss.activity_cards = None
    say("assistant", "지금 무엇부터 해야 할지 막막하군요. 서비스를 찾기 전에, 부담이 적은 작은 활동부터 살펴볼까요? 지금은 쉬어가도 괜찮아요.")
    go("activity_offer")
    return True


def continue_after_activities():
    ss.activity_declined = True
    if ss.activity_return == "start":
        ask_next()
    else:
        go(ss.activity_return)


def page_activity_offer():
    head("", "지금은 어떤 쪽이 편하세요?", "해야 할 일을 정하지 않아도 괜찮아요.")
    render_log()
    with st.container(key="panel-activity-offer"):
        st.markdown("##### 서비스보다 작은 활동부터 살펴볼까요?")
        st.caption("방 안에서 가볍게 해볼 수 있는 일부터, 직접 고른 범위 안에서만 보여 드려요. 먼저 동의하신 뒤에 활동을 골라요.")
        if st.button("네, 작은 활동부터 볼래요", key="activity_yes", type="primary"):
            go("activities")
            st.rerun()
        if st.button("서비스를 먼저 찾아볼래요", key="activity_no"):
            continue_after_activities()
            st.rerun()
        if st.button("지금은 쉬어갈래요", key="activity_rest", type="tertiary"):
            go("rest")
            st.rerun()


def activity_grid(cards):
    card_html = []
    for i, activity in enumerate(cards):
        meta = html.escape(activity["range"])             # 카드에는 활동 범위만. 상호작용 조건은 고르는 곳과 검색에서만 쓴다
        if activity.get("matched"):
            meta += f' · {html.escape(labels().get(activity["matched"], activity["matched"]))}'
        card_html.append(f'<article class="small-activity-card"><span class="activity-number">0{i + 1}</span><span class="activity-meta">{meta}</span><h3>{html.escape(activity["title"])}</h3></article>')
    st.markdown('<div class="small-activity-grid">' + ''.join(card_html) + '</div>', unsafe_allow_html=True)


def paired_activities(n_services):
    """추천 결과가 적거나 일상·여가 욕구가 있을 때 서비스 아래에 작은 활동을 함께 보여 줌."""
    if status() == "crisis_stop":
        return False
    p = profile()
    pairs = pair_activities(p.get("needs"), p.get("activity_range"), n_services,
                            constraint_level("방문") == "불가", ss.activity_declined)
    if not pairs:
        return False
    with st.container(key="panel-paired-activities"):
        st.markdown("##### 👣 서비스와 함께 해볼 수 있는 작은 활동")
        st.caption("지금 맞는 서비스가 적어서, 부담이 적은 작은 활동도 함께 보여 드려요." if n_services < 3 else
                   "생활 리듬은 서비스보다 작은 활동이 먼저 도움이 될 수 있어서 함께 보여 드려요.")
        activity_grid(pairs)
        st.caption("서비스를 대신하지 않아요. 알려 주신 활동 범위 안에서 사람을 만나지 않는 가벼운 활동만 골랐어요.")
    return True


def page_activities():
    head("", "지금 할 수 있는 만큼만.", "하나도 고르지 않아도 괜찮아요. 서비스는 나중에 이어서 살펴볼 수 있어요.")
    with st.container(key="panel-activity-options"):
        st.caption("팀 활동 초안 · 직접 고른 조건으로 최대 3개를 보여 드려요.")
        scopes = list(ACTIVITY)
        current = profile().get("activity_range")
        a, b, c = st.columns(3)
        scope = a.selectbox("지금 편한 활동 범위", scopes, index=scopes.index(current) if current in scopes else 0, key="activity_scope")
        energy = b.selectbox("오늘의 여유", [1, 2, 3], format_func=lambda v: {1: "아주 가볍게만", 2: "짧은 활동은 괜찮아요", 3: "조금 더 해봐도 괜찮아요"}[v], key="activity_energy")
        contact = c.selectbox("사람과의 상호작용", [0, 1], format_func=lambda v: "없는 활동만" if v == 0 else "짧은 연락·대화도 괜찮아요", key="activity_contact")
        if st.button("이 정도로 작은 활동 보기", key="activity_show", type="primary"):
            ss.activity_cards = select_activities(scope, energy, contact, constraint_level("방문") == "불가")
        # Re-filter on every render so changed limits never leave stale cards visible.
        cards = select_activities(scope, energy, contact, constraint_level("방문") == "불가") if ss.activity_cards is not None else None
    if cards is not None:
        if not cards:
            st.info("지금 고른 조건에 맞는 활동이 없어요. 범위를 무리해서 넓히지 않아도 괜찮아요.")
        activity_grid(cards)
    st.caption("활동 범위로 은둔 단계나 의욕을 판정하지 않으며, 치료나 회복 효과를 보장하지 않아요.")
    if st.button("서비스 이야기로 돌아가기", key="activity_continue"):
        continue_after_activities()
        st.rerun()
    if st.button("오늘은 여기까지 할래요", key="activity_done", type="tertiary"):
        go("rest")
        st.rerun()


def page_rest():
    head("", "지금은 쉬어가도 괜찮아요.", "아무것도 선택하거나 해내지 않아도 돼요. 원할 때 다시 이어가세요.")
    if st.button("작은 활동 살펴보기", key="rest_activities"):
        go("activities")
        st.rerun()
    if st.button("서비스 이야기 이어가기", key="rest_continue", type="tertiary"):
        continue_after_activities()
        st.rerun()


def page_unsure():
    head("", "아직 모르겠다면, 먼저 살펴봐도 괜찮아요.",
         "정보가 부족할 때 보는 탐색용 서비스예요. 맞춤 추천 순위가 아니고 지원 자격도 확정하지 않아요.")
    if st.button("서비스보다 작은 활동부터 살펴볼래요", key="unsure_activities"):
        ss.activity_return = "unsure"
        ss.activity_cards = None
        go("activity_offer")
        st.rerun()
    picks = unsure_picks()
    st.caption("탐색용 · 추천 순위 아님")
    if picks:
        service_cards(picks, "탐색용")
    else:
        st.info("지금 보여 드릴 서비스가 없어요. 전체 서비스를 둘러볼 수 있어요.")
    st.markdown("##### 이 중 조금이라도 눈에 들어오는 쪽이 있나요?")
    hit = option_grid([(d, lab) for d, lab in EXPLORE_DOMAINS], "un_dom", n=3)
    if hit:
        pick_domain(hit[0])
        st.rerun()
    a, c = st.columns(2)
    if a.button("전체 서비스 둘러보기", key="un_all", width="stretch"):
        ss.ex_domain = None
        go("explore")
        st.rerun()
    if status() == "explore_only" and c.button("한 가지만 더 적어 보기", key="un_chat", width="stretch"):
        go("chat")
        st.rerun()


# ─────────────────────────────────────────────
# 7. 화면
# ─────────────────────────────────────────────
def page_home():
    with st.container(key="home-layout"):
        main, aside = st.columns([3, 1.35], gap="large")
    with main:
        st.markdown(HERO_HTML, unsafe_allow_html=True)
        with bot():
            st.markdown(GREETING)
        with st.container(key="home-chat-actions"):
            st.caption("반 걸음은 복지 서비스를 찾아 드리는 도구예요.")
            if st.button("이야기 없이 서비스부터 둘러볼게요 →", key="home_explore"):
                go("explore")
                st.rerun()
    with aside:
        with st.container(key="aside-home"):
            st.markdown(ASIDE_HTML, unsafe_allow_html=True)
        with st.container(key="home-guide"):
            st.markdown(STEPS_HTML, unsafe_allow_html=True)
            st.caption(MSG["footer"])
    with st.bottom:             # 입력창 위: 예시 버튼만 (제목 없음)
        with st.container(key="home-examples"):
            for i, (icon, example) in enumerate(EXAMPLES):
                label = f"{icon}  " + example.replace("\n", "  \n")
                if st.button(label, key=f"ex_{i}", width="stretch"):
                    start(example.replace("\n", " "), direct=True)
                    st.rerun()
    text = st.chat_input("예를 들면, 일은 하고 싶은데 사람 만나는 게 부담스러워요.", max_chars=500, key="home_input")
    with st.bottom:             # 입력창 아래: 안내 한 줄
        with st.container(key="bottom-hint"):
            st.caption(HOME_HINT)
    if text:
        start(text)
        st.rerun()


def age_answer(key):
    with st.container(horizontal=True, horizontal_alignment="right"):
        with st.container(key=f"age-answer-{key}", width=300):
            field, action = st.columns([1.5, 1], gap="small", vertical_alignment="center")
            value = field.number_input("나이", min_value=10, max_value=99, value=None, step=1,
                                       placeholder="", key=f"{key}_num", label_visibility="collapsed")
            if action.button("다음으로", key=f"{key}_ok", type="primary", disabled=value is None, width="stretch"):
                return int(value), f"{int(value)}살이에요"
    return None


def question_actions(key):
    with st.container(horizontal=True, gap="xxsmall", vertical_alignment="center"):
        skip = st.button(SKIP_QUESTION, key=f"{key}_skip", type="tertiary")
        if st.button(UNSURE_BTN, key="chat_unsure", type="tertiary"):
            go("unsure")
            st.rerun()
        return skip


def chat_slot_widgets():
    """확인 단계 채팅 질문의 빠른 답변."""
    slot = ss.slot
    key = f"ui_{slot}_{ss.chat_count}"
    hit = None
    if slot == "need":
        lab = labels()
        suggested = [n["domain"] for n in confirm_view(ss.session)["needs"]]
        opts = [(d, lab[d]) for d in suggested if d in lab] + [(d, lab[d]) for d in pipe.domains if d not in suggested]
        hit = option_grid(opts, key, n=3, primary=suggested)
        if suggested:
            st.caption("색이 칠해진 것은 적어 주신 글에서 짐작한 도움이에요. 다른 것을 골라도 돼요.")
    elif slot == "region_sido":
        hit = option_grid([(area, region_label(area)) for area in REGION_CHOICES], key, n=6)
    elif slot == "activity_range":
        hit = option_grid(list(ACTIVITY.items()), key, n=2)
    elif slot == "phone":
        hit = option_grid(list(PHONE.items()), key, n=3)
    elif slot == "age":
        hit = age_answer(key)
    if question_actions(key):
        answer_chat("", skip=True)
        st.rerun()
    if hit:
        answer_chat(hit[1], value=hit[0])
        st.rerun()


def backend_question_widgets():
    """백엔드 추가 질문(awaiting_answer). 선택지·숫자 입력·건너뛰기는 백엔드 render 그대로."""
    r = current_question()
    if r is None:
        st.warning("질문을 불러오지 못했어요. 정리표로 돌아가서 다시 찾아볼 수 있어요.")
        if st.button("정리표로 돌아가기", key="bq_back", type="primary"):
            go_confirm()
            st.rerun()
        return None
    key = f"bq_{len(ss.session['questions'])}"
    st.html('<span id="chat-followup" aria-hidden="true"></span>')
    with bot():                          # 지금 질문을 선택지 바로 위에 다시 보여 준다
        st.markdown(esc(r["prompt"]))
    hit = age_answer(key) if r.get("number_input") else None
    if r["options"]:
        n = 6 if r["slot"] == "region_sido" else (2 if r["slot"] == "activity_range" else 3)
        opts = [(o["value"], ACTIVITY.get(o["value"], o["label"]) if r["slot"] == "activity_range" else o["label"])
                for o in r["options"]]
        if r["slot"] == "region_sido":
            opts = [(area, region_label(area)) for area in REGION_CHOICES]
        if r["slot"] in ("region_sido", "region_sgg"):
            opts.sort(key=lambda option: option[1])
        hit = option_grid(opts, key, n=n) or hit
    if question_actions(key):
        answer_backend(r, SKIP_QUESTION, None, "모름" if r["slot"] == "need" else "건너뜀")
        st.rerun()
    if hit:
        answer_backend(r, hit[1], hit[0], "선택")
        st.rerun()
    return r


def page_chat():
    render_log()
    if ss.get("pending_offer"):
        a, b, _ = st.columns([1, 1, 2])
        if a.button(chitchat.OFFER_BUTTONS[0], key="offer_yes", type="primary", width="stretch"):
            offer_accept()
            st.rerun()
        if b.button(chitchat.OFFER_BUTTONS[1], key="offer_talk", width="stretch"):
            offer_talk()
            st.rerun()
    st_ = status()
    on_text, placeholder = None, "편하게 답해 주세요. 직접 적어도 괜찮아요."
    if st_ is None:
        if ss.outcome == "error" and ss.last_text and st.button("다시 보내 보기", type="primary"):
            start(ss.last_text)
            st.rerun()
        on_text, placeholder = start, "천천히 적어도 돼요. 한 줄이어도 충분해요."
    elif st_ == "explore_only":
        if ss.snapshot and st.button("필요한 도움 직접 고르기", type="primary"):
            go_confirm()
            st.rerun()
        on_text, placeholder = start, "한 가지만 더 적어 주세요"
    elif st_ == "awaiting_answer":
        r = backend_question_widgets()
        on_text = (lambda t: answer_backend_free(r, t)) if r else None   # noqa: E731
    elif st_ == "awaiting_confirm" and ss.slot:
        chat_slot_widgets()
        on_text = answer_chat
    elif st_ == "awaiting_confirm":
        go_confirm()
        st.rerun()
    elif st_ == "ready":
        if ss.outcome and ss.outcome != "error":
            if st.button("찾은 결과 보기", type="primary"):
                go("results")
                st.rerun()
        elif st.button("다시 찾아보기", type="primary"):
            run_search()
            st.rerun()
    text = st.chat_input(placeholder if on_text else "위에서 골라 주세요", max_chars=500,
                         disabled=on_text is None, key="chat_input")
    if on_text is start:        # 일상 대화가 외부 AI로 갈 수 있는 입력일 때만 안내 (입력창 아래에 작게)
        with st.bottom:
            with st.container(key="bottom-hint"):
                st.caption(CHITCHAT_NOTICE)
    if text and on_text:
        on_text(text)
        st.rerun()


SUMMARY_ROWS = ["필요한 도움", "나이", "지역", "외출 범위", "전화 부담", "방문 부담"]


def summary_values():
    p = profile()
    lab = labels()
    sido = p.get("region_sido")
    return {
        "필요한 도움": ", ".join(lab.get(n["domain"], n["domain"]) for n in p.get("needs", [])),
        "나이": f"{p['age']}세" if p.get("age") is not None else "",
        "지역": " ".join(x for x in (region_label(sido), p.get("region_sgg")) if x),
        "외출 범위": ACTIVITY.get(p.get("activity_range"), ""),
        "전화 부담": PHONE.get(constraint_level("전화"), ""),
        "방문 부담": VISIT.get(constraint_level("방문"), ""),
    }


def page_confirm():
    # Existing browser sessions can retain the summary page after the backend
    # has advanced. Recover that state before rendering editable controls.
    if status() != "awaiting_confirm":
        go_confirm()
        st.rerun()
    head("", "지금까지의 이야기, 이렇게 모았어요.")
    render_log()
    p, lab, cv = profile(), labels(), ss.cv
    st.html('<span id="confirm-summary" aria-hidden="true"></span>')
    with st.container(key="panel-needs"):
        st.markdown("##### 도움 분야")
        needs = st.pills("필요한 도움", pipe.domains, selection_mode="multi", format_func=lambda d: lab.get(d, d),
                         default=[n["domain"] for n in p.get("needs", [])], key=f"c_need_{cv}",
                         label_visibility="collapsed")
    with st.container(key="panel-detail"):
        st.markdown("##### 검색 조건 수정")
        st.caption("나이와 지역은 서비스 조건을 확인하는 데 쓰여요. 답하지 않아도 진행할 수 있어요.")
        a, b = st.columns(2)
        age = a.number_input("나이", min_value=10, max_value=99, value=p.get("age"), step=1,
                             placeholder="", key=f"c_age_{cv}")
        regions = []
        for area in REGION_CHOICES:
            districts = set(pipe.sgg_map.get(area, ()))       # 광주·전남 모두 통합 데이터의 시군구를 합쳐서 보여 줌
            regions.extend((area, district) for district in [None, *sorted(districts)])
        previous_area = p.get("region_sido")
        current = (previous_area, p.get("region_sgg") or None)
        if current not in regions:
            current = (current[0], None)
        region = b.selectbox("사는 지역", regions, format_func=lambda r: " ".join(filter(None, (region_label(r[0]), r[1]))),
                             placeholder="시·도 또는 시·군·구 검색", key=f"c_region_{cv}",
                             index=regions.index(current) if current in regions else None)
        sido, sgg = region or (None, None)
        acts = list(ACTIVITY)
        act = a.selectbox("외출 범위", acts, format_func=ACTIVITY.get, placeholder="미입력",
                          index=acts.index(p["activity_range"]) if p.get("activity_range") in acts else None,
                          key=f"c_act_{cv}")
        levels = {}
        for col, ch in ((a, "전화"), (b, "방문")):
            opts = list(CHANNEL_LABELS[ch])
            cur = constraint_level(ch)
            levels[ch] = col.selectbox(f"{ch} 부담", opts, format_func=CHANNEL_LABELS[ch].get,
                                       index=opts.index(cur) if cur in opts else None,
                                       placeholder="미입력", key=f"c_{ch}_{cv}")
    if st.button(UNSURE_BTN, key="c_unsure", type="tertiary"):
        go("unsure")
        st.rerun()
    a, b, c = st.columns([1.4, 1, 1.6], vertical_alignment="center")
    if a.button("이 조건으로 도움 찾기 →", type="primary", key="c_submit", width="stretch"):
        submit_confirm(needs, age, sido, sgg, act, levels)   # 필요한 도움이 없으면 정보 부족 탐색으로
        st.rerun()
    if b.button("대화 이어가기", key="c_continue", width="stretch"):
        ss.chat_count = 0
        ask_next()
        st.rerun()
    if c.toggle("대화에서 확인한 조건", key=f"show_summary_{cv}"):
        with st.container(key="panel-summary"):
            vals = summary_values()
            for i, name in enumerate(SUMMARY_ROWS):
                with st.container(key=f"summary-row-{i}"):
                    st.caption(name)
                    st.markdown(esc(vals[name]) if vals[name] else f":gray[{UNKNOWN}]")
                    if vals[name]:
                        st.badge("확인했어요", color="green")
                    else:
                        st.badge("미입력", color="orange")


def detail_panel():
    f = ss.detail
    with st.expander("이용하기 편한 조건으로 좁히기"):
        st.caption("받은 결과 안에서만 걸러요. 정보가 없는 서비스는 빼지 않고 그대로 보여 드려요.")
        contact = st.radio("만남 형태", list(CONTACT), key="d_contact", horizontal=True,
                           index=list(CONTACT.values()).index(f["contact"]))
        repeat = st.radio("이용 기간", list(REPEAT), key="d_repeat", horizontal=True,
                          index=list(REPEAT.values()).index(f["repeat"]))
        no_interview = st.checkbox("면접 없이 신청할 수 있는 것만", f["no_interview"], key="d_iv")
    ss.detail = {"contact": CONTACT[contact], "repeat": REPEAT[repeat], "no_interview": no_interview}


def show_card(c, i, live=True, tag=None, visit_controls=False):
    with st.container(key=f"card-{i}-{c['servId']}"):
        first = live and i == 0 and tag is None
        tag = tag or (("먼저 살펴볼 도움" if i == 0 else "추천 서비스") if live else "담은 서비스")
        region = _clean(c.get("region_sido")) or ("전국" if _clean(c.get("source")) == "central" else "지역 확인 필요")
        color = "blue" if first else "orange"
        badge = f":{color}-badge[{tag}] :gray[{esc(region)}]"
        if visit_controls:
            with st.container(key=f"saved-heading-{i}"):
                st.checkbox("이 서비스 선택", key=f"visit_select_{c['servId']}",
                            on_change=sync_visit_selection, label_visibility="collapsed")
                st.markdown(badge)
        else:
            st.markdown(badge)
        st.markdown(f"#### {esc(_clean(c.get('servNm')) or '서비스 이름 정보 없음')}")
        body = _clean(c.get("reason") if live else c.get("summary_text"))
        if not live or c.get("explanation_pending") or c.get("explanation_kind") == "source":
            body = friendly_summary(body)
        with st.container(key=f"service-preview-{i}-{c['servId']}"):
            st.markdown(esc(short(body, 180) or "자세한 내용을 확인해 주세요."))
        with st.container(key=f"service-actions-{i}-{c['servId']}"):
            if live:
                st.caption("신청 자격은 추가 확인이 필요해요.")
            if st.button("자세히 보기 ↗", key=f"open_{i}_{c['servId']}", width="stretch"):
                ss.selected, ss.detail_from = c["servId"], ss.page
                service_dialog()


def page_results():
    with st.container(key="results-heading", horizontal=True, gap="xsmall", vertical_alignment="center"):
        head("", "지금의 나에게 맞는 도움", width="content")
        result_tags()
    out = ss.out
    paired = False
    if ss.outcome == "error" or out is None:
        with st.container(key="panel-error"):
            st.markdown("#### 잠시 연결이 원활하지 않아요")
            st.write(MSG["error"])
            if status() == "ready" and st.button("다시 찾아보기", type="primary"):
                run_search()
                st.rerun()
    elif ss.outcome in ("empty", "all_excluded"):
        with st.container(key="panel-empty"):
            st.markdown("#### 관련 서비스를 찾지 못했어요" if ss.outcome == "empty" else "#### 지금 조건으로는 결과가 없어요")
            st.write(copy_for(ss.outcome))
        paired = paired_activities(0)
    else:
        if out["retrieval_status"] == "partial":
            st.warning(MSG["partial"])
        detail_panel()
        st.write("천천히 보셔도 돼요. 집에서 할 수 있는 것부터 보면 충분해요.")
        cards = apply_detail_filters(out["cards"], ss.detail)[:TOP_CARDS]   # 백엔드 순서 그대로 유지
        if not cards:
            st.write(MSG["detail_none"])
        else:
            st.caption(f"추천 {len(cards)}개 · 지원 확정이 아니에요")
        for i, c in enumerate(cards):
            show_card(c, i)
        paired = paired_activities(len(cards))
    a, b, c = st.columns([1.3, 1.3, 1.5], vertical_alignment="center")
    if a.button("조건을 바꿔 다시 찾기", key="r_confirm", width="stretch"):
        go_confirm()
        st.rerun()
    if b.button("새로운 이야기 시작하기", key="r_new", width="stretch"):
        new_story()
        st.rerun()
    if c.toggle("지금까지 나눈 대화", key="r_show_chat"):
        render_log()
    if st.button("다른 조건으로 작은 활동 보기" if paired else "작은 활동도 살펴볼까요?", key="results_activities", type="tertiary"):
        ss.activity_return = "results"
        ss.activity_cards = None
        go("activity_offer")
        st.rerun()
    if ss.dev and out:
        dev_panel(out)


@st.dialog("서비스 자세히 보기", width="large")
def service_dialog():
    # Keep the list mounted behind the dialog, including filters and scroll position.
    if status() == "crisis_stop":
        st.warning("지금은 안전을 위한 도움을 먼저 확인해 주세요.")
        return
    page_detail(in_dialog=True)


def page_detail(in_dialog=False):
    sid = ss.selected
    s = service_row(sid) if sid else None
    c = next((x for x in (ss.out or {}).get("cards", []) if x["servId"] == sid), None)
    review = getattr(pipe, "review", None)
    review_fields = review.card_fields(sid) if review is not None and sid in pipe.services.index else {}
    if not in_dialog and st.button("← 목록으로 돌아가기", type="tertiary"):
        go(ss.detail_from)
        st.rerun()
    if s is None:
        st.markdown("#### 상세 자료를 찾지 못했어요")
        st.write("이 버전의 서비스 목록에서 찾을 수 없어요. 다시 목록에서 골라 주세요.")
        return
    kind = {"central": "중앙부처 서비스", "local": "지역 서비스"}.get(_clean(s["source"]), "서비스")
    st.markdown(f":green-badge[{kind}] :gray[{esc(s['region_sido'] or '지역 정보 확인 필요')}]")
    st.markdown(f"## {esc(s['servNm'] or sid)}")
    if s["summary_text"]:
        st.markdown(service_detail_markdown(s["summary_text"]))
    left, right = st.columns([2.6, 1], gap="large")
    with left:
        if c:
            with st.container(key="panel-reason"):
                pending = c.get("explanation_pending", False)
                is_source = pending or c.get("explanation_kind") == "source" or c.get("fallback")
                st.markdown("**서비스 소개**" if is_source else "**반 걸음이 설명해 드려요**")
                reason = _clean(c.get("reason"))
                st.markdown(service_detail_markdown(reason))
                if pending:
                    st.caption("목록은 먼저 보여 드렸어요. 원하시면 이 서비스 하나의 AI 설명을 요청할 수 있어요.")
                    if st.button("AI로 이 서비스 설명 보기", key=f"explain_{sid}", type="primary"):
                        with st.spinner("이 서비스의 원문을 읽고 설명하고 있어요..."):
                            explained = safe(pipe.explain_service, ss.session, ss.out, sid)
                        if explained is not None:
                            ss.out["cards"] = [explained if x["servId"] == sid else x for x in ss.out["cards"]]
                            st.rerun(scope="fragment" if in_dialog else "app")
                else:
                    st.caption("생성 설명 · 아래 원문과 구분해서 확인해 주세요." if not is_source else "AI 설명을 확인하지 못해 서비스 원문 안내를 보여 드려요.")
        for label, col in (("지원 대상", "target_text"), ("선정 기준", "criteria_text"),
                           ("지원 내용", "benefit_text"), ("신청 방법", "apply_text")):
            value = s[col]
            if col == "apply_text" and value:
                value = "\n".join(line for line in value.splitlines()
                                  if not re.match(r"^\s*거주지\s+읍\s*/\s*면\s*/\s*동", line))
                if not value.strip():
                    continue
            st.divider()
            st.markdown(f"##### {label}")
            st.markdown(service_detail_markdown(value))
        st.divider()
        st.markdown("##### 신청과 이용은 나눠서 확인해요")
        for label, col in (("신청 경로", "apply_channel"), ("면접 여부", "apply_interview"), ("문의 정보", "contact_json")):
            if col == "apply_channel":
                if s[col] is not None and s[col] not in APPLY_CHANNEL:   # 뜻을 모르는 값(예: 숫자만)은 줄을 숨김
                    continue
                value = APPLY_CHANNEL.get(s[col])
            else:
                value = contact_text(s[col]) if col == "contact_json" else s[col]
            if col == "contact_json":
                st.markdown(f"**{label}**")
                st.markdown("\n".join(f"- {esc(line)}" for line in value.splitlines()) if value
                            else ":gray[정보 확인 필요]")
            else:
                st.markdown(f"**{label}** · {esc(value) if value else '정보 확인 필요'}")
        st.caption("신청이 온라인이어도 실제 이용에는 방문이 필요할 수 있어요.")
        q2 = (c or {}).get("q2_source_comparison") or review_fields.get("q2_source_comparison") or {}
        if q2.get("same_program_confirmed"):
            st.divider()
            st.markdown("##### 공식 사이트 비교")
            st.caption("같은 사업으로 검수한 자료예요. 현재 모집 여부는 공식 안내에서 다시 확인해 주세요.")
            item_labels = {"age": "나이", "income": "소득", "channel": "신청 경로",
                           "period": "신청 기간", "documents": "구비 서류", "contact": "문의처"}
            for k, v in (q2.get("items") or {}).items():
                status_label = str(v.get("status_label") or v.get("status")).replace("SSIS", "복지로")
                with st.expander(f"{item_labels.get(k, k)} · {status_label}"):
                    for side, label in (("ssis", "복지로 수집 원문"), ("gov24", "정부24 수집 원문")):
                        evidence = v.get(side) or {}
                        st.markdown(f"**{label}**")
                        fields = evidence.get("source_fields") or {}
                        st.markdown(esc("\n".join(str(value) for value in fields.values())) if fields else "확인된 원문 정보가 없어요.")
                        link = http_url(evidence.get("url"))
                        if link:
                            st.link_button(f"{item_labels.get(k, k)} · {label} 확인 ↗", link)
    with right:
        with st.container(key="aside-step"):
            st.markdown(":orange-badge[여기서 반 걸음]")
            st.markdown(copy_for("half_step"))
            link = http_url(s["detail_url"])
            if link:
                st.link_button("공식 안내 열기 ↗", link, type="primary", width="stretch")
            else:
                st.caption("확인된 공식 링크가 없어요.")
            saved = sid in ss.visit
            if st.button("담았어요 ✓" if saved else "담기", key=f"save_{sid}",
                         disabled=saved, width="stretch"):
                ss.visit = ss.visit + [sid]
                st.toast("담은 서비스에 넣었어요.")
                st.rerun(scope="fragment" if in_dialog else "app")
            st.caption("이 창을 열어 둔 동안만 기억해요.  \n새로고침하면 사라져요.")
            items = clean_list((c or {}).get("check_needed"))
            if items:
                with st.expander("신청 전 확인할 점"):
                    st.markdown("\n".join(f"- {esc(x)}" for x in items))


def sync_visit_selection():
    ss.visit_select_all = bool(ss.visit) and all(ss.get(f"visit_select_{sid}", False) for sid in ss.visit)


def select_all_visits():
    for sid in ss.visit:
        ss[f"visit_select_{sid}"] = ss.visit_select_all


def remove_visits(ids):
    removed = set(ids)
    ss.visit = [sid for sid in ss.visit if sid not in removed]
    for sid in removed:
        ss.pop(f"visit_select_{sid}", None)
    sync_visit_selection()


def page_visit():
    head("", "오늘 담은 서비스", "이 창을 열어 둔 동안만 기억해요. 다음 추천에 자동 반영하지 않아요.")
    if not ss.visit:
        with st.container(key="panel-visit-empty"):
            st.markdown('<div class="empty-symbol" aria-hidden="true">◌</div><h2 class="empty-title">아직 담은 서비스가 없어요</h2><p class="empty-copy">지금 마음에 들어오는 서비스가 있다면 담아 둘 수 있어요.</p>', unsafe_allow_html=True)
            if st.button("둘러보기", key="visit_browse"):
                go("explore")
                st.rerun()
        return
    selected = [sid for sid in ss.visit if ss.get(f"visit_select_{sid}", False)]
    with st.container(key="visit-management"):
        st.checkbox("전체 선택", key="visit_select_all", on_change=select_all_visits)
        st.button(f"삭제 ({len(selected)})" if selected else "삭제", key="visit_remove_selected", disabled=not selected,
                  on_click=remove_visits, args=(selected,))
    for i, sid in enumerate(ss.visit):
        s = service_row(sid)
        if s is None:
            st.caption(f"이 버전에서 조회할 수 없는 서비스예요 ({esc(sid)})")
            st.checkbox("이 서비스 선택", key=f"visit_select_{sid}", on_change=sync_visit_selection,
                        label_visibility="collapsed")
            continue
        show_card({"servId": sid, **s}, i, live=False, visit_controls=True)


def page_safety():
    head("", "지금 바로 연락할 수 있는 도움",
         "지금 위험하다면 긴급 도움을 요청해 주세요. 혼자 감당하지 않아도 괜찮아요.")
    render_guidance(ss.guidance or crisis_guidance())
    if st.button("새로운 이야기 시작하기", key="s_new"):
        new_story()
        st.rerun()


def page_about():
    head("", "나의 속도로, 필요한 만큼만.")
    mode = {"시연용 가짜": "화면 확인을 위한 시연 모드예요. 예시 서비스와 가짜 응답을 사용하며 실제 추천 품질을 확인하는 실행이 아니에요.",
            "실제 흐름(가짜 모델)": "실제 서비스 데이터와 Python Pipeline을 연결한 모드예요. 모델은 테스트용 대체 모델이며 추천 품질을 검증하는 실행이 아니에요.",
            "실제 모델": "실제 서비스 데이터와 Python Pipeline을 연결한 모드예요. 설정된 모델이 추출·검색·설명을 처리해요."}[ss.pipe_choice]
    st.markdown(f"""<article class="reference-panel reference-prose">
<h2>이 화면에서 할 수 있는 것</h2><p>이야기를 적고, 필요한 도움과 부담 조건을 확인한 뒤 서비스별 신청 방법과 원문을 살펴볼 수 있어요. 서비스 둘러보기에서는 이야기 없이도 도움을 먼저 살펴볼 수 있어요. 검색 결과는 목록과 원문 안내를 먼저 보여 드리고, AI 설명은 상세 화면에서 요청할 때 작성해요.</p>
<h2>지금의 실행 모드</h2><p>{html.escape(mode)}</p>
<h2>추천은 지원 확정이 아니에요</h2><p>원문의 정보가 없거나 오래되었을 수 있어요. 소득·대상·모집 기간 등 최종 조건과 신청 가능 여부는 기관의 최신 안내로 확인해 주세요. 온라인 신청과 비대면 이용도 각각 확인해야 해요.</p>
</article><article class="reference-panel reference-prose">
<h2>개인정보와 담은 서비스</h2><p>이름·연락처·주민등록번호를 적을 필요가 없어요. 나이와 지역은 알려 주지 않아도 되고, 질문은 건너뛸 수 있어요.</p>
<p>대화와 담은 서비스는 접속한 Streamlit 세션의 메모리에서 관리해요. 새로운 이야기를 시작하면 대화가 초기화되고, 담은 서비스는 같은 세션에 남아요. 새로고침이나 접속 종료 시 유지되지 않을 수 있어요.</p>
<h2>👣 선택형 작은 활동</h2><p>막막하거나 의욕이 없다는 답변에는 작은 활동부터 살펴볼지 먼저 물어요. 동의하면 19개 팀 초안에서 직접 고른 범위·여유·상호작용 조건에 맞춰 최대 3개를 보여 드려요. 서비스를 먼저 찾거나 쉬어갈 수도 있어요.</p>
<h2>일상 대화와 외부 AI</h2>{CHITCHAT_ABOUT}
<h2>모델 처리와 기록</h2><p>실제 모델의 처리 위치와 외부 전송·로그 보관 여부는 실행 환경 설정에 따라 달라요. 운영 환경의 처리·보관 정책은 별도로 확인해야 해요. 화면에서 대화를 초기화하는 동작이 서버 로그 삭제를 뜻하지는 않아요.</p>
<p>화면의 글꼴을 불러올 때 Google Fonts로 외부 요청이 생겨요. 이 화면이 자동으로 기관에 대화를 공유하거나 신고·예약을 제출하지는 않아요.</p>
</article>""", unsafe_allow_html=True)


def page_dev():
    st.title("개발자")
    st.caption("팀 시연용 설정이에요. 사용자에게는 보이지 않게 할 메뉴예요.")
    review = getattr(pipe, "review", None)
    if review is not None:
        st.json({"models": pipe.cfg["models"], "services": len(pipe.services),
                 "q1_reviewed_services": len(review.q1), "q2_confirmed_pairs": len(review.q2),
                 "dense_profile": pipe.cfg["retrieval"].get("dense_profile"),
                 "rrf_weights": pipe.cfg["retrieval"].get("rrf_weights", {"bm25": 1, "dense": 1})})
        if not review.q1 or not review.q2:
            st.warning("최종 검수 데이터 일부가 연결되지 않았어요. prepare_runtime.py 실행 후 다시 시작해 주세요.")
    choice = st.selectbox("파이프라인", PIPE_CHOICES, index=PIPE_CHOICES.index(ss.pipe_choice), key="w_pipe")
    if choice != ss.pipe_choice:        # 바꾸면 대화를 처음부터
        ss.pipe_choice, ss.pipe_error = choice, None
        reset_all()
        go("dev")
        st.rerun()
    scenario = st.selectbox("시나리오 (시연용 가짜에서만)", SCENARIOS, index=SCENARIOS.index(ss.scenario),
                            key="w_scenario", disabled=ss.pipe_choice != "시연용 가짜")
    if scenario != ss.scenario:
        ss.scenario = scenario
        reset_all()
        go("dev")
        st.rerun()
    ss.dev = st.toggle("개발자 보기", value=ss.dev, key="w_dev")
    st.caption("파이프라인이나 시나리오를 바꾸면 대화가 처음부터 다시 시작돼요.")
    if ss.pipe_error:
        st.error(ss.pipe_error)
    if ss.get("last_error"):
        st.warning("대화 중 오류: " + ss.last_error)
    if ss.get("chitchat_status"):
        st.caption("일상 대화 응답: " + ss.chitchat_status + " (gemini = 외부 AI 응답, fallback:이유 = 기본 답)")
    if ss.session:
        with st.expander("현재 세션 (status · confirmed_profile · questions)"):
            st.json({"status": status(), "confirmed_profile": profile(), "questions": ss.session.get("questions"),
                     "snapshot": bool(ss.snapshot), "asked(채팅)": ss.asked})


def dev_panel(out):
    st.divider()
    st.caption("개발자 보기 (사용자에게는 안 보임)")
    ranks = out.get("ranks") or {}
    rrf = {r["servId"]: r for r in ranks.get("rrf", [])}
    st.dataframe([{"servId": c["servId"], "RRF": rrf.get(c["servId"], {}).get("rrf_rank"),
                   "BM25": rrf.get(c["servId"], {}).get("ranks", {}).get("bm25"),
                   "Dense": rrf.get(c["servId"], {}).get("ranks", {}).get("dense"),
                   **{k: c.get(k) for k in BURDEN_COLS + ["apply_interview"]}} for c in out["cards"]])
    with st.expander("queries"):
        st.json(out.get("queries") or {})
    with st.expander("rules_view"):
        st.json(out.get("rules_view") or {})
    with st.expander("ranks (앞 10개)"):
        st.json({"bm25": ranks.get("bm25", [])[:10], "dense": ranks.get("dense", [])[:10],
                 "fused": ranks.get("fused", [])[:10], "status": ranks.get("status"),
                 "errors": ranks.get("errors")})
    with st.expander(f"excluded ({len(out.get('excluded') or [])}건)"):
        st.json(out.get("excluded") or [])
    st.json({"status": out["status"], "retrieval_status": out["retrieval_status"],
             "영역 원문": [(n["domain"], n["level"]) for n in profile().get("needs", [])],
             # 백엔드 위기 판정에는 점수가 없고 crisis·by·evidence만 있다.
             "위기 판정": out.get("crisis"), "세부 필터": ss.detail, "errors": out["retrieval_errors"]})


# ─────────────────────────────────────────────
# 8. 사이드바(좌측 레일) · 상단 · 화면 고르기
# ─────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
st.html(AI_DIR / "ui_responsive.css")

NAV = [("conv", "대화로 찾아보기"), ("explore", "서비스 둘러보기"), ("visit", "담은 서비스"), ("activities", "👣 작은 활동"), ("about", "이용 안내·개인정보"),
       ("safety", "지금 도움이 필요하다면"), ("dev", "개발자")]

with st.sidebar:
    st.markdown(BRAND_HTML, unsafe_allow_html=True)
    if st.button("＋ 새로운 이야기", key="nav_new", type="primary", width="stretch"):
        new_story()
        st.rerun()
    with st.container(key="rail-nav"):
        for k, label in NAV[:4]:
            active = (ss.page in CONV_PAGES + ("detail", "unsure") and k == "conv") or ss.page == k
            if st.button(label, key=f"nav_{k}", type="secondary" if active else "tertiary", width="stretch"):
                if k == "activities" and ss.page not in ("activity_offer", "activities", "rest"):
                    ss.activity_return = ss.page
                go(ss.conv_page if k == "conv" else k)
                st.rerun()
    with st.container(key="rail-bottom"):
        if st.button("이용 안내 · 개인정보", key="nav_about", type="tertiary"):
            go("about")
            st.rerun()
        if st.button("지금 도움이 필요하다면 ↗", key="nav_safety", type="tertiary"):
            go("safety")
            st.rerun()
        st.markdown('<p class="rail-copyright">© 반 걸음 · 함께, 천천히</p>', unsafe_allow_html=True)


if st.query_params.get("dev") == "1" and not DEMO:
    with st.popover("개발자 설정", icon=":material/tune:"):
        choice = st.selectbox("파이프라인", PIPE_CHOICES, index=PIPE_CHOICES.index(ss.pipe_choice), key="quick_pipe")
        if choice != ss.pipe_choice:
            ss.pipe_choice, ss.pipe_error = choice, None
            reset_all()
            st.rerun()
        if st.button("개발자 화면 열기", key="open_dev"):
            go("dev")
            st.rerun()
with st.container(key="compact-nav"):
    if st.button("대화", key="compact_chat", width="stretch"):
        go(ss.conv_page)
        st.rerun()
    if st.button("서비스", key="compact_explore", width="stretch"):
        go("explore")
        st.rerun()
    if st.button("담은 서비스", key="compact_visit", width="stretch"):
        go("visit")
        st.rerun()
    with st.popover("메뉴", width="stretch"):
        if st.button("👣 작은 활동", key="compact_activities", width="stretch"):
            if ss.page not in ("activity_offer", "activities", "rest"):
                ss.activity_return = ss.page
            go("activities")
            st.rerun()
        if st.button("새 이야기", key="compact_new", width="stretch"):
            new_story()
            st.rerun()
        if st.button("이용 안내 · 개인정보", key="compact_about", width="stretch"):
            go("about")
            st.rerun()
        if st.button("지금 도움이 필요하다면", key="compact_safety", width="stretch"):
            go("safety")
            st.rerun()
pipe = get_pipeline()

page = ss.page
if DEMO and page == "dev":
    page = "home"                       # 시연 모드에서는 개발자 화면을 열지 않는다
if status() == "crisis_stop" and page in ("home", "chat", "confirm", "results", "detail", "explore", "unsure", "visit", "activity_offer", "activities", "rest"):
    page = "safety"                     # 위기 중단 세션은 결과·둘러보기·탐색·입력 화면으로 가지 않는다
st.markdown(f'<style>:root{{--visit-count:"{len(ss.visit)}"}}</style>' , unsafe_allow_html=True)
if page == "chat":              # 대화 화면은 위쪽 바 없이 대화가 바로 시작되게 함 (개발자 설정은 ?dev=1 주소로 계속 가능)
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
else:
    badge = (f'<span class="mode">{MODE_BADGE[ss.pipe_choice]}</span>' if DEMO else
             f'<a class="mode" href="?dev=1" target="_self" aria-label="개발자 설정">{MODE_BADGE[ss.pipe_choice]}</a>')
    st.markdown(f'<div class="topbar"><div class="breadcrumb">반 걸음<span>/</span><b>{PAGE_NAMES[page]}</b></div>{badge}</div>',
                unsafe_allow_html=True)

{"home": page_home, "chat": page_chat, "confirm": page_confirm, "results": page_results, "detail": page_detail,
 "activity_offer": page_activity_offer, "activities": page_activities, "rest": page_rest,
 "visit": page_visit, "explore": page_explore, "unsure": page_unsure, "about": page_about, "safety": page_safety, "dev": page_dev}[page]()

if page != "home":
    st.markdown(f'<div class="footer">반 걸음은 정보 탐색을 도와드려요. {MSG["footer"]}</div>', unsafe_allow_html=True)
render_scroll(page)
