"""Streamlit Community Cloud 진입점. ai/app.py를 그대로 실행한다 (Main file path: deploy/streamlit_app.py).
이 폴더의 requirements.txt가 배포용 가벼운 설치 목록이다. DEMO_MODE=true Secret과 함께 쓴다."""
import runpy
import sys
from pathlib import Path

AI = Path(__file__).resolve().parent.parent / "ai"
sys.path.insert(0, str(AI))
runpy.run_path(str(AI / "app.py"), run_name="__main__")
