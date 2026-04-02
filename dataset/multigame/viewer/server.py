from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .backend import DatasetViewerBackend

_STATIC_DIR = Path(__file__).parent / "static"
# dataset/multigame/viewer → dataset/multigame → dataset → tile_ims
_TILE_IMS_DIR = Path(__file__).parent.parent.parent / "tile_ims"


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class _ViewerHandler(BaseHTTPRequestHandler):
    backend: DatasetViewerBackend

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if path == "/static/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
            return
        if path.startswith("/tile_ims/"):
            self._serve_tile_image(path)
            return
        if path == "/api/games":
            self._send_json({"games": self.backend.games_with_counts()})
            return
        if path == "/api/sample":
            self._handle_sample_api(parsed.query)
            return
        if path == "/api/mapping":
            self._handle_mapping_api(parsed.query)
            return

        if path.startswith("/api/"):
            self._send_json({"error": f"api not found: {path}"}, status=HTTPStatus.NOT_FOUND)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/reload":
            self._handle_reload_api()
            return

        self._send_json({"error": f"api not found: {path}"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        # Keep terminal output compact while browsing quickly.
        return

    def _handle_reload_api(self) -> None:
        """데이터셋 + tile_mapping.json 을 프로세스 재시작 없이 다시 로드한다."""
        import time
        t0 = time.time()
        try:
            result = self.backend.reload()
            elapsed = round(time.time() - t0, 2)
            self._send_json({**result, "elapsed_sec": elapsed})
            # 서버 콘솔에도 출력
            print(f"[viewer] reloaded in {elapsed}s — games: "
                  + ", ".join(f"{r['game']}({r['count']})" for r in result["games"]))
        except Exception as exc:  # noqa: BLE001
            self._send_json({"status": "error", "error": str(exc)},
                            status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_static(self, file_name: str, content_type: str) -> None:
        path = _STATIC_DIR / file_name
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_sample_api(self, query: str) -> None:
        params = parse_qs(query)
        game = (params.get("game") or [None])[0]
        index_raw = (params.get("index") or ["0"])[0]

        if not game:
            self._send_json({"error": "missing required query param: game"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            index = int(index_raw)
            payload = self.backend.get_sample(game=game, index=index)
            self._send_json(payload)
        except ValueError:
            self._send_json({"error": "index must be an integer"}, status=HTTPStatus.BAD_REQUEST)
        except (KeyError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except IndexError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)

    def _handle_mapping_api(self, query: str) -> None:
        params = parse_qs(query)
        game = (params.get("game") or [None])[0]

        if not game:
            self._send_json({"error": "missing required query param: game"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = self.backend.get_game_mapping(game=game)
            self._send_json(payload)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _serve_tile_image(self, req_path: str) -> None:
        # Serve only files under dataset/tile_ims (subdirectories allowed).
        name = req_path[len("/tile_ims/"):].strip()
        if not name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid tile image path")
            return

        path = (_TILE_IMS_DIR / name).resolve()
        # path traversal 방지: tile_ims 디렉토리 밖으로 나가는 경로 차단
        if not str(path).startswith(str(_TILE_IMS_DIR.resolve())):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid tile image path")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Tile image not found")
            return

        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    dungeon_root: str | None = None,
    pokemon_root: str | None = None,
    boxoban_root: str | None = None,
    doom_root: str | None = None,
) -> None:
    kwargs = {}
    if dungeon_root:
        kwargs["dungeon_root"] = dungeon_root
    if pokemon_root:
        kwargs["pokemon_root"] = pokemon_root
    if boxoban_root:
        kwargs["boxoban_root"] = boxoban_root
    if doom_root:
        kwargs["doom_root"] = doom_root

    backend = DatasetViewerBackend(**kwargs) if kwargs else DatasetViewerBackend()

    class Handler(_ViewerHandler):
        pass

    Handler.backend = backend

    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = ReusableServer((host, port), Handler)  # type: ignore[arg-type]

    actual_port = server.server_address[1]
    print(f"[viewer] Serving on http://{host}:{actual_port}")
    print("[viewer] Available games and counts:")
    for row in backend.games_with_counts():
        print(f"  - {row['game']}: {row['count']}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] Stopped by user")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset browser viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--dungeon-root", help="Dungeon dataset root path")
    parser.add_argument("--pokemon-root", help="Pokemon dataset root path")
    parser.add_argument("--boxoban-root", help="Boxoban dataset root path")
    parser.add_argument("--doom-root", help="DOOM dataset root path")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        dungeon_root=args.dungeon_root,
        pokemon_root=args.pokemon_root,
        boxoban_root=args.boxoban_root,
        doom_root=args.doom_root,
    )


if __name__ == "__main__":
    main()

