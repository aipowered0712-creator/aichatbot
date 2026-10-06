"""知匯 AI 單一啟動入口：執行後啟動 Web 服務並開啟瀏覽器。"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


def configure_console_output() -> None:
    """确保 Windows 终端能输出中文启动状态，而不会因活动代码页退出。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="啟動知匯 AI 通用知識檢索平台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    """读取 .env，已存在的环境变量不被覆盖。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key.startswith("export "):
            key = key[7:].strip()
        if key and key not in os.environ:
            os.environ[key] = value


def browser_host(bind_host: str) -> str:
    return "127.0.0.1" if bind_host in WILDCARD_HOSTS else bind_host


def probe_running_app(host: str, port: int) -> bool:
    url = f"http://{browser_host(host)}:{port}/api/health"
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def find_free_port(host: str, start: int, attempts: int = 20) -> int:
    bind_host = "0.0.0.0" if host in WILDCARD_HOSTS else host
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((bind_host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"連接埠 {start}-{start + attempts - 1} 均被占用，請用 --port 指定其他連接埠")


def warn_missing_config() -> None:
    local_path = os.environ.get("LOCAL_MODEL_PATH", r"D:\A_model\Kimi-VL-A3B-Instruct")
    if Path(local_path).exists():
        print(f"本地模型：已找到 {local_path}")
    else:
        print(f"提示：本地模型目錄不存在（LOCAL_MODEL_PATH={local_path}），本地模式將不可用。")
    if os.environ.get("HKPC_LLM_API_KEY"):
        print("線上模型：已讀取 HKPC_LLM_API_KEY")
    else:
        print("提示：未設定 HKPC_LLM_API_KEY，線上模式將不可用。")
    knowledge_dir = Path(os.environ.get("KNOWLEDGE_BASE_DIR", APP_DIR / "knowledge_base" / "moldpdf"))
    if knowledge_dir.name != "knowledge_base" and knowledge_dir.parent.name == "knowledge_base":
        knowledge_dir = knowledge_dir.parent
    if knowledge_dir.is_dir():
        docs = [
            path for path in knowledge_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {
                ".pdf", ".docx", ".txt", ".md", ".xlsx", ".xlsm", ".pptx",
                ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
            } and path.name.lower() != "readme.md"
        ]
        print(f"知識庫：{len(docs)} 份文檔，啟動後會在後台建立檢索索引。")
        print(f"FAISS 索引：{APP_DIR / 'faiss'}")
        print("請使用上面列印的網址開啟頁面；關閉本視窗後網頁會連不上。")
    else:
        print(f"提示：知識庫目錄不存在（{knowledge_dir}）。")


def announce_when_ready(server, url: str, open_browser: bool) -> None:
    while not getattr(server, "started", False):
        time.sleep(0.05)
    print(f"\n知匯 AI 已啟動：{url}")
    print("按 Ctrl+C 停止服務。\n")
    if open_browser:
        webbrowser.open(url)


def main() -> None:
    configure_console_output()
    args = parse_args()
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    load_dotenv(APP_DIR / ".env")

    try:
        import uvicorn
        from api_server import app
    except ImportError as error:
        raise SystemExit("缺少執行依賴，請先執行：python -m pip install -r requirements.txt") from error

    preferred = f"http://{browser_host(args.host)}:{args.port}"
    if probe_running_app(args.host, args.port):
        print(f"知匯 AI 已在運行：{preferred}")
        print("無需重複啟動。請用這個網址開啟頁面，不要使用 8001。")
        if not args.no_browser:
            webbrowser.open(preferred)
        return

    port = find_free_port(args.host, args.port)
    if port != args.port:
        print(f"連接埠 {args.port} 已被占用，改用 {port}。")
        print(f"請開啟 http://{browser_host(args.host)}:{port} ，不要繼續使用 {preferred}")

    url = f"http://{browser_host(args.host)}:{port}"
    print(f"正在啟動知匯 AI：{url}")
    warn_missing_config()

    config = uvicorn.Config(app, host=args.host, port=port, reload=False)
    server = uvicorn.Server(config)
    threading.Thread(
        target=announce_when_ready,
        args=(server, url, not args.no_browser),
        name="zhihui-ready",
        daemon=True,
    ).start()

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n知匯 AI 已停止。")


if __name__ == "__main__":
    main()
