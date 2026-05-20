from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import build_knowledge_base, configure_console_encoding, initialize_agent
from core.rag_eval import RAGEvalRunner
from core.safety_eval import SafetyEvalRunner
from core.retriever import HybridRetriever


class WebState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.agent = None
        self.memory_manager = None
        self.status = "starting"
        self.last_error = ""
        self.initialize()

    def initialize(self) -> None:
        with self.lock:
            self.status = "loading"
            try:
                self.agent, self.memory_manager = initialize_agent()
                self.status = "ready"
                self.last_error = ""
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                raise

    def clear_conversation(self) -> None:
        with self.lock:
            if self.memory_manager is None:
                return
            short_term = getattr(self.memory_manager, "short_term", None)
            if short_term is not None:
                getattr(short_term, "window", []).clear()
                if hasattr(short_term, "summary"):
                    short_term.summary = ""


STATE: WebState | None = None


class TCMRAGHandler(BaseHTTPRequestHandler):
    server_version = "TCMRAGWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": STATE is not None and STATE.status == "ready",
                    "status": STATE.status if STATE else "starting",
                    "error": STATE.last_error if STATE else "",
                }
            )
            return

        path = "/index.html" if parsed.path == "/" else parsed.path
        self._serve_static(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        routes = {
            "/api/chat": self._handle_chat,
            "/api/new": self._handle_new,
            "/api/rebuild": self._handle_rebuild,
            "/api/eval": self._handle_eval,
            "/api/safetyeval": self._handle_safety_eval,
        }
        handler = routes.get(parsed.path)
        if handler is None:
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return
        try:
            handler()
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _handle_chat(self) -> None:
        payload = self._read_json()
        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json({"ok": False, "error": "message is required"}, status=400)
            return

        state = self._require_state()
        with state.lock:
            result = state.agent.run(message)

        self._send_json(
            {
                "ok": True,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0),
                "retrieval_iterations": result.get("retrieval_iterations", 0),
                "evidence_count": result.get("evidence_count", 0),
                "safety_action": result.get("safety_action", "proceed"),
                "safety_risk_level": result.get("safety_risk_level", "low"),
                "safety_flags": result.get("safety_flags", []),
            }
        )

    def _handle_new(self) -> None:
        state = self._require_state()
        state.clear_conversation()
        self._send_json({"ok": True, "message": "conversation cleared"})

    def _handle_rebuild(self) -> None:
        state = self._require_state()
        with state.lock:
            retriever = HybridRetriever()
            build_knowledge_base("data/raw", retriever)
            state.initialize()
        self._send_json({"ok": True, "message": "knowledge base rebuilt"})

    def _handle_eval(self) -> None:
        state = self._require_state()
        with state.lock:
            summary = RAGEvalRunner(state.agent).run()
        self._send_json({"ok": True, "summary": summary.__dict__})

    def _handle_safety_eval(self) -> None:
        result = SafetyEvalRunner().run()
        self._send_json({"ok": True, "summary": result.__dict__})

    def _serve_static(self, request_path: str) -> None:
        relative = request_path.lstrip("/")
        file_path = (STATIC_DIR / relative).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _require_state(self) -> WebState:
        if STATE is None or STATE.status != "ready" or STATE.agent is None:
            raise RuntimeError("Agent is not ready")
        return STATE

    def log_message(self, fmt: str, *args: Tuple[Any, ...]) -> None:
        sys.stderr.write("[web] " + fmt % args + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TCM-RAG web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    configure_console_encoding()

    global STATE
    STATE = WebState()

    server = ThreadingHTTPServer((args.host, args.port), TCMRAGHandler)
    print(f"TCM-RAG web demo running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
