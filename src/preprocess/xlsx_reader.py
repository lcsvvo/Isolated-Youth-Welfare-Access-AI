"""표준 라이브러리만으로 xlsx 시트를 문자열 DataFrame으로 읽는다.

openpyxl이 requirements.txt에 없어서 둔 최소 구현이다. 값은 모두 str로 읽고(dtype=str과 같은 취지),
수식·서식·병합셀은 다루지 않는다. 첫 행을 헤더로 쓴다.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_COL = re.compile(r"([A-Z]+)(\d+)")


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")) for si in root.findall("m:si", _NS)]


def sheet_names(path: Path) -> dict[str, str]:
    """시트 이름 → zip 내부 XML 경로."""
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for s in wb.find("m:sheets", _NS):
        t = target[s.get(_REL)].lstrip("/")
        out[s.get("name")] = t if t.startswith("xl/") else f"xl/{t}"
    return out


def read_sheet(path: Path, name: str, header: bool = True) -> pd.DataFrame:
    xml_path = sheet_names(path)[name]
    with zipfile.ZipFile(path) as z:
        sst = _shared_strings(z)
        rows: list[dict[int, str]] = []
        with z.open(xml_path) as f:
            for _, el in ET.iterparse(f):
                if el.tag != f"{{{_NS['m']}}}row":
                    continue
                row = {}
                for c in el.findall("m:c", _NS):
                    col = _col_index(_COL.match(c.get("r")).group(1))
                    t = c.get("t")
                    if t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
                    else:
                        v = c.find("m:v", _NS)
                        if v is None or v.text is None:
                            continue
                        val = sst[int(v.text)] if t == "s" else v.text
                    row[col] = val
                rows.append(row)
                el.clear()
    width = max((max(r) + 1 for r in rows if r), default=0)
    grid = [[r.get(i) for i in range(width)] for r in rows]
    if not header:
        return pd.DataFrame(grid, dtype=str)
    cols = [c if c is not None else f"_col{i}" for i, c in enumerate(grid[0])]
    return pd.DataFrame(grid[1:], columns=cols, dtype=str)
