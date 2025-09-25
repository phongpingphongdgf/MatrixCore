# ============================
# dashboard.py  (FastAPI API only: write+read+tests+csv+pdf+limits)
# VERSION surface comes from matrix_core
# ============================
from __future__ import annotations

import os, socket, argparse, io, csv, json as _json
from typing import Dict

from fastapi import FastAPI, Request, Query, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

import uvicorn
from .matrix_core import (
    MatrixCore,
    __version__ as CORE_VERSION,
    __app_name__ as APP_NAME,
    MAX_STRUCTURE_UNITS,
    StructureUnitLimitExceeded,
)

# optional PDF support
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

CORE = MatrixCore()
app = FastAPI(title=f"{APP_NAME} API v{CORE_VERSION}")

# ----------- exception handlers -----------
@app.exception_handler(StructureUnitLimitExceeded)
async def su_limit_handler(request: Request, exc: StructureUnitLimitExceeded):
    return JSONResponse(
        status_code=409,
        content={
            "detail": "StructureUnit limit exceeded",
            "level": exc.level_name,
            "requested": exc.requested,
            "limit": exc.limit,
        },
    )

# ---------------- System & version ----------------
@app.get("/api/version")
async def api_version():
    return {"name": APP_NAME, "version": CORE_VERSION}

@app.get("/api/limits")
async def api_limits():
    return {"max_structure_units": MAX_STRUCTURE_UNITS}

@app.get("/api/snapshot.json")
async def api_snapshot():
    return PlainTextResponse(_json.dumps(CORE.to_dict(), ensure_ascii=False), media_type="application/json")

# ---------------- Ingest (text & PDF) ----------------
@app.post("/api/ingest")
async def api_ingest(payload: Dict[str, str]):
    text = payload.get("text", "")
    CORE.encode_message(text)  # may raise StructureUnitLimitExceeded -> handled above
    return {"ok": True}

@app.post("/api/ingest_pdf")
async def api_ingest_pdf(file: UploadFile = File(...)):
    if PdfReader is None:
        raise HTTPException(status_code=400, detail="pypdf not installed. pip install pypdf")
    empty_pages = 0
    total_pages = 0
    try:
        reader = PdfReader(file.file)
        total_pages = len(reader.pages)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip():
                empty_pages += 1
            CORE.encode_message(text)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    return {"ok": True, "total_pages": total_pages, "empty_pages": empty_pages}

# ---------------- Banks browsing (pairs, groups, CSV) ----------------
def _get_lu(level: str):
    if level == "word":
        return CORE.word_logic
    if level == "sentence":
        return CORE.sent_logic
    return CORE.msg_logic

@app.get("/api/unit/{level}/stats")
async def api_unit_stats(level: str):
    st = _get_lu(level).stats
    return {
        "bytes_processed": st.bytes_processed,
        "units_processed": st.units_processed,
        "bank_total_pairs": st.bank_total_pairs,
        "bank_pairs_by_su": st.bank_pairs_by_su,
        "xs_processed": st.xs_processed,
        "ys_bank_size": st.ys_bank_size,
    }

@app.get("/api/unit/{level}/su/{su_id}/pairs")
async def api_su_pairs(level: str, su_id: int, page: int = Query(1, ge=1), size: int = Query(100, ge=1, le=1000)):
    lu = _get_lu(level)
    if su_id < 0 or su_id >= len(lu.structur_chain):
        return {"page": page, "size": size, "total": 0, "rows": []}
    su = lu.structur_chain[su_id]
    return su.snapshot_pairs_page(page=page, size=size)

@app.get("/api/unit/{level}/groups")
async def api_groups(level: str):
    lu = _get_lu(level)
    arr = []
    for i, su in enumerate(lu.structur_chain):
        arr.append({"su_id": i, "groups": su.snapshot_groups()})
    return arr

@app.get("/api/unit/{level}/group/{prim}/sec")
async def api_group_secs(level: str, prim: int, su_id: int = 0, page: int = Query(1, ge=1), size: int = Query(100, ge=1, le=1000)):
    lu = _get_lu(level)
    if su_id < 0 or su_id >= len(lu.structur_chain):
        return {"prim": prim, "page": page, "size": size, "total": 0, "rows": []}
    su = lu.structur_chain[su_id]
    return su.snapshot_group_secs_page(prim=prim, page=page, size=size)

@app.get("/api/unit/{level}/su/{su_id}/pairs/search")
async def api_su_pairs_search(level: str, su_id: int,
                              page: int = Query(1, ge=1),
                              size: int = Query(100, ge=1, le=5000),
                              prim: int | None = None,
                              sec: int | None = None,
                              min_hits: int = 0):
    lu = _get_lu(level)
    if su_id < 0 or su_id >= len(lu.structur_chain):
        return {"page": page, "size": size, "total": 0, "rows": []}
    su = lu.structur_chain[su_id]
    rows = []
    for idx in range(1, len(su.pair_index_map)):  # skip reserved 0
        p, s = su.pair_index_map[idx]
        h = su.pair_hits[idx] if idx < len(su.pair_hits) else 0
        if prim is not None and p != prim: continue
        if sec is not None and s != sec: continue
        if h < min_hits: continue
        rows.append({"pair_index": idx, "prim": p, "sec": s, "hits": h,
                     "prim_is_pair": su.prim_is_pair[idx], "sec_is_pair": su.sec_is_pair[idx]})
    total = len(rows)
    start = (page - 1) * size
    end = min(total, start + size)
    return {"page": page, "size": size, "total": total, "rows": rows[start:end]}

@app.get("/api/unit/{level}/su/{su_id}/pairs.csv")
async def api_su_pairs_csv(level: str, su_id: int,
                           prim: int | None = None,
                           sec: int | None = None,
                           min_hits: int = 0):
    j = await api_su_pairs_search(level, su_id, page=1, size=10_000_000, prim=prim, sec=sec, min_hits=min_hits)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["pair_index","prim","sec","hits","prim_is_pair","sec_is_pair"])
    w.writeheader(); w.writerows(j["rows"])
    return StreamingResponse(iter([buf.getvalue()]),
                             media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=su_{su_id}_pairs.csv"})

# ---------------- Decode (read-path) ----------------
@app.get("/api/decode/word")
async def api_decode_word(level: int, idx: int):
    return CORE.decode_word_iv((level, idx))

@app.get("/api/decode/sentence")
async def api_decode_sentence(level: int, idx: int):
    return CORE.decode_sentence_iv((level, idx))

@app.get("/api/decode/message")
async def api_decode_message(level: int, idx: int):
    return CORE.decode_message_iv((level, idx))

# ---------------- Test suite via API ----------------
class TestReport:
    def __init__(self):
        self.results: list[dict] = []
    def add(self, name: str, ok: bool, details: str = ""):
        self.results.append({"name": name, "ok": ok, "details": details})
    def json(self):
        total = len(self.results); passed = sum(1 for r in self.results if r["ok"])
        return {"total": total, "passed": passed, "failed": total - passed, "results": self.results}

def _test_ab_mirror(tr: TestReport):
    ok = True; details = []
    for lu_name in ("word", "sent", "msg"):
        lu = getattr(CORE, f"{lu_name}_logic")
        for lvl, su in enumerate(lu.structur_chain):
            for prim, sec, pair_idx in su._a_sorted:
                if pair_idx >= len(su.pair_index_map) or su.pair_index_map[pair_idx] != (prim, sec):
                    ok = False; details.append(f"{lu_name}[{lvl}] A→B mismatch at {pair_idx}"); break
            aset = {(p, s, idx) for (p, s, idx) in su._a_sorted}
            for idx in range(1, len(su.pair_index_map)):
                p, s = su.pair_index_map[idx]
                if (p, s, idx) not in aset:
                    ok = False; details.append(f"{lu_name}[{lvl}] B→A missing ({p},{s})@{idx}"); break
    tr.add("ab_mirror", ok, "; ".join(details[:4]))

def _test_word_roundtrip(tr: TestReport):
    words = ["a", "ab", "abc", "Привет", "CAN", "BUS", "TCP", "IP"]
    ok = True; bad: list[str] = []
    for w in words:
        iv = CORE.encode_word_text(w)
        w2 = CORE.decode_word_iv(iv)["text"]
        if w != w2: ok = False; bad.append(f"{w}->{w2}")
    tr.add("word_roundtrip", ok, ", ".join(bad))

def _test_sentence_roundtrip(tr: TestReport):
    sent = "hello TCP IP"; iv = CORE.encode_sentence_text(sent); text = CORE.decode_sentence_iv(iv)["text"]
    ok = text.split() == sent.split()
    tr.add("sentence_roundtrip", ok, f"'{sent}' -> '{text}'")

def _test_message_roundtrip(tr: TestReport):
    msg = "Hello world! Привет мир!"; iv = CORE.encode_message(msg); text = CORE.decode_message_iv(iv)["text"]
    ok = set(text.split()) >= set(["Hello", "world", "Привет", "мир"])
    tr.add("message_roundtrip", ok, f"decoded='{text}'")

def _test_hits_counters(tr: TestReport):
    CORE.encode_message("aaa aaa"); CORE.encode_message("aaa")
    lu = CORE.word_logic; ok = True
    if not lu.structur_chain: ok = False
    else:
        su0 = lu.structur_chain[0]; pa = (ord('a'), ord('a'))
        idx = next((idx for idx, t in enumerate(su0.pair_index_map) if t == pa), None)
        if not idx or su0.pair_hits[idx] <= 0: ok = False
    tr.add("hits_counters", ok, "" if ok else "pair ('a','a') not found / no hits")

def _test_limit_enforced(tr: TestReport):
    CORE.reset()
    ok = False; details = ""
    try:
        CORE.encode_word_text("a" * 300)  # n=300 -> needs 299 SUs > 256
    except StructureUnitLimitExceeded as e:
        ok = True; details = f"caught: {e.level_name}, requested={e.requested}, limit={e.limit}"
    tr.add("limit_enforced", ok, details)

TEST_REGISTRY = {
    "ab_mirror": _test_ab_mirror,
    "word_roundtrip": _test_word_roundtrip,
    "sentence_roundtrip": _test_sentence_roundtrip,
    "message_roundtrip": _test_message_roundtrip,
    "hits_counters": _test_hits_counters,
    "limit_enforced": _test_limit_enforced,
}

@app.post("/api/test/reset")
async def api_test_reset():
    CORE.reset()
    return {"ok": True}

@app.get("/api/test/run")
async def api_test_run(name: str):
    if name not in TEST_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown test '{name}'")
    tr = TestReport(); TEST_REGISTRY[name](tr)
    return tr.json()

@app.get("/api/test/run_all")
async def api_test_run_all():
    tr = TestReport()
    for name in TEST_REGISTRY:
        TEST_REGISTRY[name](tr)
    return tr.json()

# ---------------- autostart helpers (optional) ----------------
def _is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def start_server_in_thread(host: str = "127.0.0.1", port: int = 8000) -> None:
    def _target():
        uvicorn.run(app, host=host, port=port, log_level="info")
    import threading
    threading.Thread(target=_target, daemon=True).start()

def ensure_server(host: str = "127.0.0.1", port: int = 8000) -> bool:
    if _is_port_open(host, port):
        return False
    start_server_in_thread(host, port)
    return True

# ---------------- CLI entry ----------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{APP_NAME} API")
    p.add_argument("--host", default=os.environ.get("MATRIX_HOST", "0.0.0.0"))
    p.add_argument("--port", default=int(os.environ.get("MATRIX_PORT", "8000")), type=int)
    p.add_argument("--reload", action="store_true", help="Enable autoreload (dev).")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run("api.dashboard:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
