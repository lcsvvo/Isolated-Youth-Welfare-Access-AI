"""Launch the current Streamlit app behind the authenticated Colab notebook proxy.

In a Colab cell:
    from pathlib import Path
    import runpy
    root = Path('/content/bangeol-real-ai')
    launch = runpy.run_path(str(root / 'ai/scripts/colab_launch.py'))
    streamlit_process = launch['start_preview'](root, globals().get('streamlit_process'))
"""
import html
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _stop_previous(root, process):
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait(timeout=30)
        return
    pid_path = root / "streamlit.pid"
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
        proc = Path("/proc") / str(pid)
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
        cwd = (proc / "cwd").resolve()
        if b"streamlit" not in cmd or b"app.py" not in cmd or cwd != (root / "ai").resolve():
            raise RuntimeError("기록된 PID가 이 앱이 아닙니다. 기존 프로세스를 확인해 주세요.")
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not proc.exists():
                return
            try:
                state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
                if state == "Z":
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                    return
            except FileNotFoundError:
                return
            time.sleep(0.25)
        raise RuntimeError("기존 Streamlit 프로세스 종료를 확인해 주세요.")
    except (FileNotFoundError, ProcessLookupError):
        return


def start_preview(root=Path("/content/bangeol-real-ai"), process=None):
    from google.colab import output
    from google.colab.output import eval_js
    from IPython.display import display, HTML
    root = Path(root).resolve()
    if not (root / "ai/app.py").exists():
        raise FileNotFoundError("이 런타임에 앱 파일이 없습니다. GitHub 받기 또는 ZIP 업로드 셀부터 실행해 주세요.")
    proxy_url = eval_js("google.colab.kernel.proxyPort(8501)").rstrip("/")
    host = urlparse(proxy_url).hostname
    if not host or not host.endswith(".colab.dev"):
        raise ValueError("Colab proxy 주소를 확인해 주세요")
    env = os.environ.copy()
    env["BANGEOL_PIPELINE_MODE"] = "실제 모델"
    env["BANGEOL_REQUIRE_REVIEW_DATA"] = "1"
    env.setdefault("BANGEOL_EMBED_BATCH_SIZE", "4")
    _stop_previous(root, process)
    with (root / "streamlit.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0",
             "--server.port=8501", "--server.headless=true", "--browser.serverAddress=" + host,
             "--server.enableWebsocketCompression=false", "--server.corsAllowedOrigins=" + proxy_url,
             "--browser.gatherUsageStats=false"],
            cwd=root / "ai", env=env, stdout=log, stderr=subprocess.STDOUT)
    (root / "streamlit.pid").write_text(str(process.pid))
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((root / "streamlit.log").read_text(encoding="utf-8"))
        try:
            with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("Streamlit 시작 시간 초과: streamlit.log 확인 필요")
    display(HTML(f'<p><a href="{html.escape(proxy_url, quote=True)}" target="_blank" rel="noopener">반 걸음 실제 AI 화면을 새 탭으로 열기 ↗</a></p>'))
    output.serve_kernel_port_as_iframe(8501, height=900)
    print("Colab 노트북과 T4 연결을 유지해 주세요. health 응답은 모델 적재 완료를 뜻하지 않습니다.")
    return process
