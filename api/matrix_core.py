# ============================
# api/matrix_core.py
# VERSION: 0.4.7
# ============================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import bisect
import logging
import re
import json
import threading

# ---- versioning ----
__app_name__ = "Matrix"
__version__ = "0.4.7"

# ---- limits (configurable) ----
MAX_STRUCTURE_UNITS = 256  # hard cap per LogicUnit chain length

# ---- custom exceptions ----
class MatrixError(Exception):
    """Base class for Matrix-specific errors."""

class StructureUnitLimitExceeded(MatrixError):
    """Raised when required SU chain length exceeds MAX_STRUCTURE_UNITS."""
    def __init__(self, level_name: str, requested: int, limit: int = MAX_STRUCTURE_UNITS):
        super().__init__(f"{level_name}: structure units required={requested} exceeds limit={limit}")
        self.level_name = level_name
        self.requested = requested
        self.limit = limit


# -----------------------------
# Types
# -----------------------------
InitVector = Tuple[int, int]  # (st_unit_num, pair_index)

# Telemetry of a LogicUnit
@dataclass
class LUStats:
    level_name: str  # "word" | "sentence" | "message"
    bytes_processed: int = 0
    units_processed: int = 0
    bank_total_pairs: int = 0  # total across all StructureUnits at this level
    bank_pairs_by_su: Dict[int, int] = field(default_factory=dict)  # su_id -> count
    last_init_vectors: List[Tuple[InitVector, str]] = field(default_factory=list)  # [(IV, human_label)]
    xs_processed: List[int] = field(default_factory=list)  # X: total processed (bytes)
    ys_bank_size: List[int] = field(default_factory=list)  # Y: size (pairs)
    # Per-SU traces for mini charts: su_id -> (xs, ys)
    su_traces: Dict[int, Tuple[List[int], List[int]]] = field(default_factory=dict)

@dataclass
class CoreUpdate:
    lu_stats: Dict[str, LUStats]  # {level: LUStats}
    total_bytes: int
    total_bank_pairs: int

# -----------------------------
# Text preprocessor (simplified)
# -----------------------------
class TextPreprocessor:
    WORD_RE = re.compile(r"\w+", re.UNICODE)

    def split_into_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        parts = re.split(r"([.!?])", text)
        sents: List[str] = []
        cur = ""
        for chunk in parts:
            if chunk in ".!?":
                cur += chunk
                if cur.strip():
                    sents.append(cur.strip())
                cur = ""
            else:
                cur += chunk
        if cur.strip():
            sents.append(cur.strip())
        return sents

    def tokenize_words(self, sent: str) -> List[str]:
        return self.WORD_RE.findall(sent)

# -----------------------------
# Primary & Secondary groups for A-index analytics
# -----------------------------
@dataclass
class SecondaryEntry:
    sec: int
    pair_index: int
    hits: int = 0

@dataclass
class PrimaryGroup:
    prim: int
    hits: int = 0
    sec_map: Dict[int, SecondaryEntry] = field(default_factory=dict)

# -----------------------------
# StructureUnit: stores pairs for one fold level
# -----------------------------
class StructureUnit:
    """One folding level. Maintains two mirrored tables A & B and per-cell counters.

    A (write/search):  *sorted* list of (prim, sec, pair_index) for determinism/scan
                        + primary/secondary analytics (groups with hit counters)
    B (read):          append-only list  pair_index -> (prim, sec)
    Flags:             for each pair_index we store whether prim/sec are *pairs* of prev level.

    Invariant: index 0 in all data tables is **reserved** (zeros) for future stats.
    """

    def __init__(self, st_unit_num: int):
        self.st_unit_num = st_unit_num
        # --- B table (read): index -> pair ---
        self.pair_index_map: List[Tuple[int, int]] = [(0, 0)]  # reserve cell 0
        self.pair_hits: List[int] = [0]
        self.prim_is_pair: List[bool] = [False]
        self.sec_is_pair: List[bool] = [False]

        # --- A table (write/search): sorted triples ---
        self._a_sorted: List[Tuple[int, int, int]] = []  # (prim, sec, pair_index)
        # TODO (post-MVP): заменить A на dict {(prim, sec) -> pair_index} для O(1) вставок.

        # Primary/secondary analytics
        self.primary_groups: Dict[int, PrimaryGroup] = {}
        self._sentinel_group = PrimaryGroup(prim=0)

    # ---------- A: helpers ----------
    def _a_find(self, prim: int, sec: int) -> Optional[int]:
        i = bisect.bisect_left(self._a_sorted, (prim, sec, -1))
        if i < len(self._a_sorted):
            p, s, idx = self._a_sorted[i]
            if p == prim and s == sec:
                return idx
        return None

    def _a_insert(self, prim: int, sec: int, pair_index: int) -> None:
        i = bisect.bisect_left(self._a_sorted, (prim, sec, -1))
        self._a_sorted.insert(i, (prim, sec, pair_index))

    # ---------- groups & counters ----------
    def _ensure_group(self, prim: int) -> PrimaryGroup:
        if prim == 0:
            return self._sentinel_group
        g = self.primary_groups.get(prim)
        if g is None:
            g = PrimaryGroup(prim=prim)
            self.primary_groups[prim] = g
        return g

    def _count_hit(self, prim: int, sec: int, pair_index: int) -> None:
        while pair_index >= len(self.pair_hits):
            self.pair_hits.append(0)
            self.prim_is_pair.append(False)
            self.sec_is_pair.append(False)
        self.pair_hits[pair_index] += 1
        g = self._ensure_group(prim)
        g.hits += 1
        se = g.sec_map.get(sec)
        if se is None:
            se = SecondaryEntry(sec=sec, pair_index=pair_index, hits=0)
            g.sec_map[sec] = se
        se.hits += 1

    # ---------- public API ----------
    def find_or_add_pair(self, prim: int, sec: int, prim_is_pair: bool, sec_is_pair: bool) -> int:
        found = self._a_find(prim, sec)
        if found is not None:
            self._count_hit(prim, sec, found)
            return found
        pair_index = len(self.pair_index_map)
        self.pair_index_map.append((prim, sec))
        self.pair_hits.append(0)
        self.prim_is_pair.append(prim_is_pair)
        self.sec_is_pair.append(sec_is_pair)
        self._a_insert(prim, sec, pair_index)
        self._count_hit(prim, sec, pair_index)
        return pair_index

    def bank_size(self) -> int:
        return max(0, len(self.pair_index_map) - 1)

    # --------- snapshots for API ---------
    def snapshot_pairs_page(self, page: int = 1, size: int = 100) -> Dict:
        start = max(1, (page - 1) * size + 1)
        end = min(len(self.pair_index_map), start + size)
        rows = []
        for idx in range(start, end):
            prim, sec = self.pair_index_map[idx]
            rows.append({
                "pair_index": idx,
                "prim": prim,
                "sec": sec,
                "hits": self.pair_hits[idx] if idx < len(self.pair_hits) else 0,
                "prim_is_pair": self.prim_is_pair[idx],
                "sec_is_pair": self.sec_is_pair[idx],
            })
        return {
            "page": page,
            "size": size,
            "total": max(0, len(self.pair_index_map) - 1),
            "rows": rows,
        }

    def snapshot_groups(self) -> List[Dict]:
        out = []
        out.append({"prim": 0, "hits": self._sentinel_group.hits, "sec_count": len(self._sentinel_group.sec_map)})
        for prim, g in sorted(self.primary_groups.items()):
            out.append({"prim": prim, "hits": g.hits, "sec_count": len(g.sec_map)})
        return out

    def snapshot_group_secs_page(self, prim: int, page: int = 1, size: int = 100) -> Dict:
        g = self._ensure_group(prim)
        secs = sorted(g.sec_map.values(), key=lambda e: e.sec)
        start = (page - 1) * size
        end = min(len(secs), start + size)
        rows = [{"sec": e.sec, "pair_index": e.pair_index, "hits": e.hits} for e in secs[start:end]]
        return {"prim": prim, "page": page, "size": size, "total": len(secs), "rows": rows}

# -----------------------------
# LogicUnit + concrete levels
# -----------------------------
class LogicUnit:
    def __init__(self, level_name: str):
        self.level_name = level_name
        self.structur_chain: List[StructureUnit] = []
        self.stats = LUStats(level_name=level_name)

    def ensure_structur_chain(self, needed_len: int) -> None:
        if needed_len > MAX_STRUCTURE_UNITS:
            raise StructureUnitLimitExceeded(self.level_name, requested=needed_len, limit=MAX_STRUCTURE_UNITS)
        while len(self.structur_chain) < max(0, needed_len):
            self.structur_chain.append(StructureUnit(st_unit_num=len(self.structur_chain)))

    def build_base_symbols(self, unit_input) -> List[int]:
        raise NotImplementedError

    def label_for_unit(self, unit_input) -> str:
        return str(unit_input)[:60]

    def encode_unit(self, unit_input) -> InitVector:
        symbols = self.build_base_symbols(unit_input)
        n = len(symbols)
        if n == 0:
            self.ensure_structur_chain(1)
            return (0, 0)
        if n == 1:
            self.ensure_structur_chain(1)
            su = self.structur_chain[0]
            idx = su.find_or_add_pair(symbols[0], 0, prim_is_pair=False, sec_is_pair=False)
            return (0, idx)

        self.ensure_structur_chain(n - 1)
        cur = symbols
        cur_is_pair = [False] * len(symbols)

        for level in range(n - 1):
            su = self.structur_chain[level]
            next_level: List[int] = []
            next_is_pair: List[bool] = []
            for i in range(len(cur) - 1):
                pair_index = su.find_or_add_pair(
                    cur[i], cur[i + 1],
                    prim_is_pair=cur_is_pair[i],
                    sec_is_pair=cur_is_pair[i + 1],
                )
                next_level.append(pair_index)
                next_is_pair.append(True)
            cur = next_level
            cur_is_pair = next_is_pair

        return (n - 2, cur[0])

    def bump_stats_after_block(self, bytes_added: int) -> None:
        total_bank = sum(su.bank_size() for su in self.structur_chain)
        self.stats.bank_total_pairs = total_bank
        self.stats.bank_pairs_by_su = {i: su.bank_size() for i, su in enumerate(self.structur_chain)}
        self.stats.bytes_processed += bytes_added
        self.stats.xs_processed.append(self.stats.bytes_processed)
        self.stats.ys_bank_size.append(total_bank)
        for i, su in enumerate(self.structur_chain):
            xs, ys = self.stats.su_traces.setdefault(i, ([], []))
            xs.append(self.stats.bytes_processed)
            ys.append(su.bank_size())

class WordLogicUnit(LogicUnit):
    def __init__(self):
        super().__init__("word")
    def build_base_symbols(self, word: str) -> List[int]:
        return [ord(ch) for ch in word]

class SentenceLogicUnit(LogicUnit):
    def __init__(self):
        super().__init__("sentence")
    @staticmethod
    def iv_pack(iv: InitVector) -> int:
        return (iv[0] << 32) | (iv[1] & 0xFFFFFFFF)
    @staticmethod
    def iv_unpack(token: int) -> InitVector:
        return ((token >> 32) & 0xFFFFFFFF, token & 0xFFFFFFFF)
    def build_base_symbols(self, word_vectors: List[InitVector]) -> List[int]:
        return [self.iv_pack(iv) for iv in word_vectors]

class MessageLogicUnit(LogicUnit):
    def __init__(self):
        super().__init__("message")
    @staticmethod
    def iv_pack(iv: InitVector) -> int:
        return (iv[0] << 32) | (iv[1] & 0xFFFFFFFF)
    @staticmethod
    def iv_unpack(token: int) -> InitVector:
        return ((token >> 32) & 0xFFFFFFFF, token & 0xFFFFFFFF)
    def build_base_symbols(self, sent_vectors: List[InitVector]) -> List[int]:
        return [self.iv_pack(iv) for iv in sent_vectors]

# -----------------------------
# MatrixCore: write+read + telemetry + autosave + helpers
# -----------------------------
class MatrixCore:
    def __init__(self):
        self.pre = TextPreprocessor()
        self.word_logic = WordLogicUnit()
        self.sent_logic = SentenceLogicUnit()
        self.msg_logic = MessageLogicUnit()
        self.on_block_processed: Optional[Callable[[CoreUpdate], None]] = None
        self.log = logging.getLogger("matrix")

        # autosave settings
        self.autosave_path: Optional[str] = "matrix_autosave.json"
        self.autosave_enabled: bool = True
        self._autosave_lock = threading.Lock()

    # ---------- WRITE ----------
    def encode_message(self, text: str) -> InitVector:
        sentences = self.pre.split_into_sentences(text)
        bytes_len = len(text.encode("utf-8"))
        word_vectors: List[InitVector] = []
        sent_vectors: List[InitVector] = []
        for sent in sentences:
            words = self.pre.tokenize_words(sent)
            for w in words:
                iv = self.word_logic.encode_unit(w)
                self.word_logic.stats.units_processed += 1
                self.word_logic.stats.last_init_vectors.append((iv, w))
                self.word_logic.stats.last_init_vectors = self.word_logic.stats.last_init_vectors[-200:]
                word_vectors.append(iv)
            if words:
                start_idx = len(word_vectors) - len(words)
                sent_iv = self.sent_logic.encode_unit(word_vectors[start_idx:])
                self.sent_logic.stats.units_processed += 1
                self.sent_logic.stats.last_init_vectors.append((sent_iv, sent[:60]))
                self.sent_logic.stats.last_init_vectors = self.sent_logic.stats.last_init_vectors[-200:]
                sent_vectors.append(sent_iv)
        if sent_vectors:
            msg_iv = self.msg_logic.encode_unit(sent_vectors)
        else:
            msg_iv = (0, 0)
        self.msg_logic.stats.units_processed += 1
        self.msg_logic.stats.last_init_vectors.append((msg_iv, text[:60]))
        self.msg_logic.stats.last_init_vectors = self.msg_logic.stats.last_init_vectors[-200:]

        if self.autosave_enabled and self.autosave_path:
            self._autosave_background(self.autosave_path)

        self.word_logic.bump_stats_after_block(bytes_len)
        self.sent_logic.bump_stats_after_block(bytes_len)
        self.msg_logic.bump_stats_after_block(bytes_len)

        if self.on_block_processed:
            lu = {
                "word": self.word_logic.stats,
                "sentence": self.sent_logic.stats,
                "message": self.msg_logic.stats,
            }
            total_bytes = sum(s.bytes_processed for s in lu.values())
            total_bank = sum(s.bank_total_pairs for s in lu.values())
            self.on_block_processed(CoreUpdate(lu, total_bytes, total_bank))
        return msg_iv

    # ---------- READ (decoding) ----------
    def _expand_pair(self, lu: LogicUnit, level: int, idx: int) -> List[int]:
        su = lu.structur_chain[level]
        prim, sec = su.pair_index_map[idx]
        pflag = su.prim_is_pair[idx]
        sflag = su.sec_is_pair[idx]

        left  = self._expand_symbol(lu, level - 1, prim, pflag)
        right = self._expand_symbol(lu, level - 1, sec,  sflag)

        # Перекрытие у соседних детей уровня (level-1) равно level.
        # Но у базовой пары (оба флага False) перекрытия нет → trim = 0.
        trim = level if (pflag and sflag) else 0

        if left and right:
            return left + right[trim:]
        return left + right

    def _expand_symbol(self, lu: LogicUnit, level: int, sym: int, is_pair: bool) -> List[int]:
        if level < 0 or not is_pair:
            return [sym]
        return self._expand_pair(lu, level, sym)

    def decode_word_iv(self, iv: InitVector) -> Dict:
        level, idx = iv
        if not self.word_logic.structur_chain or level < 0 or level >= len(self.word_logic.structur_chain):
            return {"codes": [], "text": ""}
        codes = self._expand_pair(self.word_logic, level, idx) if level >= 0 else [idx]
        # спец-кейс n=1: (x,0) -> [x]
        if level == 0 and len(codes) == 2 and codes[1] == 0:
            codes = [codes[0]]
        # игнорируем «нулевые» заполнители
        codes = [c for c in codes if c != 0]
        chars = []
        for c in codes:
            try:
                chars.append(chr(c))
            except (ValueError, OverflowError):
                chars.append("�")
        return {"codes": codes, "text": "".join(chars)}

    def decode_sentence_iv(self, iv: InitVector) -> Dict:
        level, idx = iv
        if not self.sent_logic.structur_chain:
            return {"tokens": [], "text": ""}
        tokens = self._expand_pair(self.sent_logic, level, idx) if level >= 0 else [idx]
        tokens = [t for t in tokens if t != 0]  # убрать «заполнитель» нули
        words: List[str] = []
        for t in tokens:
            w_iv = self.sent_logic.iv_unpack(t)
            words.append(self.decode_word_iv(w_iv)["text"])
        return {"tokens": words, "text": " ".join([w for w in words if w])}

    def decode_message_iv(self, iv: InitVector) -> Dict:
        level, idx = iv
        if not self.msg_logic.structur_chain:
            return {"sentences": [], "text": ""}
        tokens = self._expand_pair(self.msg_logic, level, idx) if level >= 0 else [idx]
        tokens = [t for t in tokens if t != 0]  # убрать «заполнитель» нули
        sents: List[str] = []
        for t in tokens:
            s_iv = self.msg_logic.iv_unpack(t)
            sents.append(self.decode_sentence_iv(s_iv)["text"])
        return {"sentences": sents, "text": " ".join([s for s in sents if s])}

    # ---------- helpers ----------
    def to_dict(self) -> Dict:
        def su_to_dict(su: StructureUnit) -> Dict:
            return {
                "st_unit_num": su.st_unit_num,
                "pair_index_map": su.pair_index_map,
                "pair_hits": su.pair_hits,
                "prim_is_pair": su.prim_is_pair,
                "sec_is_pair": su.sec_is_pair,
                "a_sorted": su._a_sorted,
                "primary_groups": {
                    prim: {
                        "hits": g.hits,
                        "sec_map": {sec: {"pair_index": e.pair_index, "hits": e.hits} for sec, e in g.sec_map.items()}
                    } for prim, g in su.primary_groups.items()
                },
            }
        def lu_to_dict(lu: LogicUnit) -> Dict:
            return {
                "level": lu.level_name,
                "stats": lu.stats.__dict__,
                "structur_chain": [su_to_dict(su) for su in lu.structur_chain],
            }
        return {
            "word": lu_to_dict(self.word_logic),
            "sentence": lu_to_dict(self.sent_logic),
            "message": lu_to_dict(self.msg_logic),
        }

    def _autosave_background(self, path: str) -> None:
        def _save():
            with self._autosave_lock:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(self.to_dict(), f, ensure_ascii=False)
                except Exception as e:
                    self.log.error(f"Autosave failed: {e}")
        threading.Thread(target=_save, daemon=True).start()

    # --- testing helpers ---
    def encode_word_text(self, word: str) -> InitVector:
        return self.word_logic.encode_unit(word)

    def encode_sentence_text(self, sentence: str) -> InitVector:
        words = self.pre.tokenize_words(sentence)
        ivs = [self.word_logic.encode_unit(w) for w in words]
        return self.sent_logic.encode_unit(ivs)

    def reset(self) -> None:
        """Полный сброс банков/статистики до исходного состояния.
        Сохраняем настройки автосейва и обратный вызов (если он был назначен из dashboard)."""
        on_cb = self.on_block_processed
        autosave_path = self.autosave_path
        autosave_enabled = self.autosave_enabled

        # Полная переинициализация
        self.__init__()

        # Вернуть полезные «крючки»
        self.on_block_processed = on_cb
        self.autosave_path = autosave_path
        self.autosave_enabled = autosave_enabled


# default logger
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
