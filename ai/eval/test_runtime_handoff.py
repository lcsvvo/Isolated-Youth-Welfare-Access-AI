"""Runtime packaging and proxy settings without a cloud account or network."""
import json
import runpy
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

AI = Path(__file__).resolve().parents[1]
BUNDLE = runpy.run_path(str(AI / "scripts/build_runtime_bundle.py"))
LAUNCH = runpy.run_path(str(AI / "scripts/colab_launch.py"))


class RuntimeHandoffTests(unittest.TestCase):
    def test_bundle_keeps_runtime_and_excludes_credentials_and_caches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = {"ai/app.py": "print('ui')", "ai/ui_scroll.js": "// fixed scroll controller",
                       "ai/ui_responsive.css": "main {min-width: 0}",
                       "ai/.streamlit/config.toml": "[theme]\nbase='light'",
                       "ai/.streamlit/secrets.toml": "private='do-not-package'", "ai/.env": "TOKEN=private",
                       "ai/data/small_activities.json": "{}", "ai/data/service_emb_core.npz": "cache",
                       "ai/data/dev_corpus.jsonl": "{}", "ai/eval/test_private.py": "private",
                       "data/processed/services_master.csv": "servId,servNm\n1,example"}
            for name, text in fixture.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            target = root / "runtime.zip"
            result = BUNDLE["build_bundle"](target, root)
            with zipfile.ZipFile(target) as z:
                names = z.namelist()
                self.assertIn("ai/app.py", names)
                self.assertIn("ai/ui_scroll.js", names)
                self.assertIn("ai/ui_responsive.css", names)
                self.assertIn("ai/data/small_activities.json", names)
                self.assertFalse(any("secret" in n or ".env" in n or ".npz" in n or "/eval/" in n for n in names))
                self.assertEqual(json.loads(z.read("bundle-manifest.json")), result)

    def test_notebook_code_cells_parse(self):
        import ast
        for path in (AI / "notebooks").glob("streamlit*.ipynb"):
            nb = json.loads(path.read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    self.assertFalse(cell["outputs"])

    def test_launch_uses_current_proxy_without_disabling_security(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "ai/.streamlit").mkdir(parents=True)
            (root / "ai/app.py").write_text("print('ui')")
            cfg = root / "ai/.streamlit/config.toml"
            cfg.write_text("[theme]\nbase='light'\n[server]\ncorsAllowedOrigins=['https://old.colab.dev']\n")
            output = types.ModuleType("google.colab.output")
            output.eval_js = lambda code: "https://8501-current.colab.dev/"
            output.serve_kernel_port_as_iframe = MagicMock()
            colab = types.ModuleType("google.colab"); colab.output = output
            google = types.ModuleType("google"); google.colab = colab
            display = types.ModuleType("IPython.display")
            display.display = MagicMock(); display.HTML = lambda value: value
            ipython = types.ModuleType("IPython"); ipython.display = display
            modules = {"google": google, "google.colab": colab, "google.colab.output": output,
                       "IPython": ipython, "IPython.display": display}
            process = MagicMock(); process.poll.return_value = None; process.pid = 1234
            response = MagicMock(); response.__enter__.return_value.status = 200
            with patch.dict(sys.modules, modules), patch("subprocess.Popen", return_value=process) as popen, patch("urllib.request.urlopen", return_value=response):
                result = LAUNCH["start_preview"](root)
            self.assertIs(result, process)
            content = cfg.read_text()
            self.assertIn('https://old.colab.dev', content)  # tracked file stays unchanged for git pull
            command = popen.call_args.args[0]
            self.assertIn('--server.corsAllowedOrigins=https://8501-current.colab.dev', command)
            self.assertIn('--server.enableWebsocketCompression=false', command)
            self.assertNotIn('enableXsrfProtection = false', content)
            self.assertEqual(popen.call_args.kwargs['env']['BANGEOL_PIPELINE_MODE'], '실제 모델')
            output.serve_kernel_port_as_iframe.assert_called_once_with(8501, height=900)


if __name__ == "__main__":
    unittest.main()
