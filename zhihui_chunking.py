"""知汇 AI 切块与索引进度。

main.py 只负责展示进度，不要在里面做切块。
索引线程调用 chunk_file()，并通过 IndexStatus 汇报阶段。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

TEXT_SUFFIXES = {".txt", ".md"}
OFFICE_SUFFIXES = {".docx", ".xlsx", ".xlsm", ".pptx"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# 中文技术文档按字符切。800 字大约落在多数 embedding 的安全区间内，
# 重叠 120 用来保住跨段的工序号、尺寸和条款。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TABLE_ROW_GROUP = 20

SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", "。", "；", "，", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    section: str = ""
    kind: str = "text"  # text / table / image


@dataclass
class IndexStatus:
    phase: str = "idle"  # idle / scanning / parsing / chunking / embedding / ready / failed
    done: int = 0
    total: int = 0
    current_file: str = ""
    message: str = ""
    error: str = ""
    started_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs: object) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = 0
            if self.started_at and self.phase not in {"ready", "idle", "failed"}:
                elapsed = max(0, int(time.monotonic() - self.started_at))
            percent = int(self.done / self.total * 100) if self.total else 0
            return {
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "current_file": self.current_file,
                "message": self.message,
                "error": self.error,
                "ready": self.phase == "ready",
                "elapsed_seconds": elapsed,
                "percent": percent,
            }


STATUS = IndexStatus()


def split_text(
    text: str,
    source: str,
    page: int | None = None,
    section: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """按中文分隔符递归切。表格块和图片不要走这里。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(text, source, page, section)]

    pieces = _recursive_split(text, chunk_size)
    pieces = [
        limited_piece
        for piece in pieces
        for limited_piece in _enforce_chunk_limit(piece, chunk_size, overlap)
    ]
    merged = _merge_with_overlap(pieces, chunk_size, overlap)
    return [Chunk(piece, source, page, section) for piece in merged if piece.strip()]


def chunk_file(
    path: Path,
    extract_page,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """extract_page(path) 由你现有解析器提供，返回 [(page_no, text, kind)]。

    kind 为 text / table / image。图片不切块，只保留路径和说明，交给视觉模型。
    chunk_size 与 overlap 只作用于正文切分，供重建索引时覆盖默认值。
    """
    suffix = path.suffix.lower()
    chunks: list[Chunk] = []
    for page_no, text, kind in extract_page(path):
        if kind == "image" or suffix in IMAGE_SUFFIXES:
            caption = text.strip() or f"图像文件：{path.name}"
            chunks.append(Chunk(caption, str(path), page_no, kind="image"))
            continue
        if kind == "table":
            chunks.extend(_chunk_table(text, str(path), page_no))
            continue
        chunks.extend(split_text(text, str(path), page_no, chunk_size=chunk_size, overlap=overlap))
    return chunks


def _chunk_table(
    text: str, source: str, page: int | None, section: str = ""
) -> list[Chunk]:
    rows = [row for row in text.splitlines() if row.strip()]
    if len(rows) <= TABLE_ROW_GROUP + 1:
        return [Chunk(text, source, page, section, kind="table")]
    header = rows[0]
    body = rows[1:]
    chunks = []
    for start in range(0, len(body), TABLE_ROW_GROUP):
        group = [header, *body[start:start + TABLE_ROW_GROUP]]
        chunks.append(Chunk("\n".join(group), source, page, section, kind="table"))
    return chunks


def chunk_content(
    text: str,
    source: str,
    page: int | None = None,
    section: str = "",
    kind: str = "text",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """按内容类型创建可检索块，保留页码、工作表或投影片等来源信息。"""
    if kind == "image":
        caption = text.strip() or f"图像文件：{Path(source).name}"
        return [Chunk(caption, source, page, section, kind="image")]
    if kind == "table":
        return _chunk_table(text, source, page, section)
    return split_text(text, source, page, section, chunk_size=chunk_size, overlap=overlap)


def _recursive_split(text: str, chunk_size: int, separators: Iterable[str] = SEPARATORS) -> list[str]:
    separators = list(separators)
    if len(text) <= chunk_size or not separators:
        return [text]
    sep = separators[0]
    if sep and sep in text:
        parts = text.split(sep)
        out: list[str] = []
        buf = ""
        for part in parts:
            piece = part if not buf else buf + sep + part
            if len(piece) <= chunk_size:
                buf = piece
                continue
            if buf:
                out.append(buf)
            if len(part) > chunk_size:
                out.extend(_recursive_split(part, chunk_size, separators[1:]))
                buf = ""
            else:
                buf = part
        if buf:
            out.append(buf)
        return out
    return _recursive_split(text, chunk_size, separators[1:])


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    if not pieces:
        return []
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if len(merged[-1]) + len(piece) <= chunk_size:
            merged[-1] = merged[-1] + piece
        else:
            available_overlap = max(0, chunk_size - len(piece))
            tail_length = min(overlap, available_overlap)
            tail = merged[-1][-tail_length:] if tail_length else ""
            merged.append(tail + piece)
    return merged


def _enforce_chunk_limit(text: str, chunk_size: int, overlap: int) -> list[str]:
    """递归分隔无效时，以带重叠的硬切分保证 embedding 输入不超长。"""
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    return [text[start:start + chunk_size] for start in range(0, len(text), step)]


def render_progress(status: IndexStatus) -> str:
    snap = status.snapshot()
    phase = {
        "idle": "等待",
        "scanning": "扫描",
        "parsing": "解析",
        "chunking": "切块",
        "embedding": "向量化",
        "ready": "完成",
        "failed": "失败",
    }.get(snap["phase"], snap["phase"])
    total = snap["total"] or 0
    done = snap["done"]
    width = 24
    filled = int(width * done / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    name = Path(snap["current_file"]).name if snap["current_file"] else ""
    extra = f" {name}" if name else ""
    if snap["error"]:
        return f"[{phase}] {snap['error']}"
    return f"[{phase}] [{bar}] {done}/{total}{extra}"


def watch_index_status(status: IndexStatus, interval: float = 0.4) -> threading.Thread:
    """给 main.py 用：服务起来后单行刷新索引进度，完成或失败后停。"""

    def _run() -> None:
        last = ""
        while True:
            line = render_progress(status)
            if line != last:
                print("\r" + line + " " * 8, end="", flush=True)
                last = line
            phase = status.snapshot()["phase"]
            if phase in {"ready", "failed"}:
                print()
                break
            time.sleep(interval)

    thread = threading.Thread(target=_run, name="zhihui-index-status", daemon=True)
    thread.start()
    return thread


# 检索后交给对话模型的提示。上下文里每段已经带来源。
QA_SYSTEM_PROMPT = """你是知匯 AI，只根據給定資料回答。
規則：
- 只用資料中的內容，不補充資料裡沒有的工藝參數、尺寸或條款。
- 資料不足時直接說「資料中沒有這一項」，不要猜測。
- 回答用繁體中文。涉及尺寸、編號、頁碼時原樣保留。
- 每條關鍵結論後標註來源，格式為（檔名，第N頁）。
- 表格數據不要改寫成沒有依據的結論。"""

QA_USER_TEMPLATE = """資料：
{context}

問題：{question}"""

# 塊寫入索引前可選用，短文檔不必跑這步。
CHUNK_SUMMARY_PROMPT = """用一兩句繁體中文概括下面片段，保留工序名、零件號、尺寸和專有名詞。不要添加片段裡沒有的資訊。

片段：
{chunk}"""
