"""단계별 HTML 요약(outputs/<stage_id>/summary/<stage_id>_summary_final.html) 생성기.

docs/RESEARCH_OUTPUT_STANDARD.md F절: hero → 좌측 sticky TOC → 본문 section → footer.
표는 tables/ CSV를 읽은 DataFrame으로, 그림은 figures/ PNG를 data URI로 넣어 오프라인 단일 파일로 만든다.

blocks 형식
  ("p", html)                         문단 (HTML 허용)
  ("box", "result"|"warning"|"good"|"dark", html)
  ("table", DataFrame, caption, source_path)
  ("fig", png_path, caption, alt)
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

CSS = """
:root{--navy:#0f172a;--blue:#2563eb;--bg:#f8fafc;--line:#e2e8f0;--ink:#1e293b;--muted:#475569;
--warn-bg:#fff7ed;--warn:#9a3412;--good-bg:#f0fdf4;--good:#166534}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 "Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif}
.hero{background:var(--navy);color:#f8fafc;padding:40px 24px}
.hero .in{max-width:1180px;margin:0 auto}
.hero h1{margin:0 0 8px;font-size:28px;line-height:1.3}
.hero .lead{font-size:19px;font-weight:600;color:#bfdbfe;margin:0 0 12px}
.hero p{max-width:900px;margin:0 0 12px;color:#e2e8f0}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;padding:0;list-style:none}
.meta li{background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:13px}
.shell{max-width:1180px;margin:0 auto;display:grid;grid-template-columns:240px minmax(0,1fr);gap:32px;padding:24px}
nav.toc{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:14px}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc a{display:block;padding:6px 10px;border-left:3px solid var(--line);color:var(--muted);text-decoration:none}
nav.toc a:hover,nav.toc a:focus{border-left-color:var(--blue);color:var(--navy);outline:none;background:#eef2ff}
main{min-width:0}
section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:24px;margin-bottom:24px}
section h2{margin:0 0 12px;font-size:22px;color:var(--navy)}
.result-box,.box{border-radius:8px;padding:14px 16px;margin:12px 0}
.result-box{background:#eff6ff;border-left:4px solid var(--blue)}
.warning{background:var(--warn-bg);border-left:4px solid var(--warn);color:#431407}
.good{background:var(--good-bg);border-left:4px solid var(--good);color:#052e16}
.dark{background:var(--navy);color:#f1f5f9}
.tablewrap{overflow-x:auto;margin:12px 0;border:1px solid var(--line);border-radius:8px}
.tablewrap:focus{outline:2px solid var(--blue)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#f1f5f9;color:var(--navy);position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.cap{font-size:13px;color:var(--muted);margin:4px 0 16px}
figure{margin:16px 0}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
footer{max-width:1180px;margin:0 auto;padding:16px 24px 48px;color:var(--muted);font-size:13px}
@media (max-width:900px){.shell{grid-template-columns:1fr;padding:16px}
nav.toc{position:static;max-height:none;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}}
@media print{nav.toc{display:none}.shell{display:block}section{break-inside:avoid-page;border:none}
.tablewrap{overflow:visible}table{font-size:11px}.dark{background:#fff;color:#000;border:1px solid #000}
.hero{background:#fff;color:#000}.hero .lead,.hero p{color:#000}.meta li{background:#fff;color:#000}}
"""


def _fmt(v) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:,.1f}" if abs(v) < 1000 or v != int(v) else f"{int(v):,}"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    return html.escape(str(v)) if v is not None else ""


def _table(df: pd.DataFrame, caption: str, source: str, max_rows: int = 60) -> str:
    head = "".join(f"<th scope='col'>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for v in r:
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            cells.append(f"<td class='num'>{_fmt(v)}</td>" if num else f"<td>{_fmt(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = f" 화면에는 상위 {max_rows}행만 표시 (전체 {len(df):,}행)." if len(df) > max_rows else ""
    return (f"<div class='tablewrap' tabindex='0'><table><caption class='cap' style='caption-side:top;text-align:left;padding:8px 10px'>"
            f"{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class='cap'>원본: <code>{html.escape(source)}</code>.{more}</p>")


def _fig(path: Path, caption: str, alt: str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(alt)}'>"
            f"<figcaption>{html.escape(caption)} (원본: <code>{html.escape(Path(path).as_posix().split('outputs/')[-1] if 'outputs' in Path(path).as_posix() else str(path))}</code>)</figcaption></figure>")


def build(out: Path, *, title: str, lead: str, summary: str, meta: list[str], sections: list[dict], footer: str) -> Path:
    toc = "".join(f"<li><a href='#{s['id']}'>{html.escape(s['title'])}</a></li>" for s in sections)
    body = []
    for s in sections:
        parts = [f"<section id='{s['id']}' aria-labelledby='{s['id']}-h'><h2 id='{s['id']}-h'>{html.escape(s['title'])}</h2>"]
        for b in s["blocks"]:
            kind = b[0]
            if kind == "p":
                parts.append(f"<p>{b[1]}</p>")
            elif kind == "box":
                cls = "result-box" if b[1] == "result" else f"box {b[1]}"
                parts.append(f"<div class='{cls}'>{b[2]}</div>")
            elif kind == "table":
                parts.append(_table(b[1], b[2], b[3]))
            elif kind == "fig":
                parts.append(_fig(b[1], b[2], b[3]))
        parts.append("</section>")
        body.append("".join(parts))
    meta_html = "".join(f"<li>{html.escape(m)}</li>" for m in meta)
    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
           f"<style>{CSS}</style></head><body>"
           f"<header class='hero'><div class='in'><h1>{html.escape(title)}</h1><p class='lead'>{lead}</p>"
           f"<p>{summary}</p><ul class='meta'>{meta_html}</ul></div></header>"
           f"<div class='shell'><nav class='toc' aria-label='목차'><ol>{toc}</ol></nav><main>{''.join(body)}</main></div>"
           f"<footer>{footer}</footer></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
